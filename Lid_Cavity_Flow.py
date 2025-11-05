import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import json
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RectBivariateSpline

# ==============================================================================
# 0. 配置中心 (Configuration Hub)
# ==============================================================================
CONFIG = {
    # 物理问题参数
    "Re_target": 100,  # 目标雷诺数
    "x_domain": [0.0, 1.0],
    "y_domain": [0.0, 1.0],
    
    # 标准PINN参数
    "pinn_layers": [2, 64, 64, 64, 64, 64, 3],
    "pinn_epochs": 20000,
    "pinn_learning_rate": 1e-3,
    "pinn_lr_decay_rate": 0.9,
    "pinn_lr_decay_steps": 5000,
    
    # Curriculum PINN参数
    "curriculum_layers": [2, 64, 64, 64, 64, 64, 3],
    "curriculum_total_epochs": 20000,
    "curriculum_learning_rate": 1e-3,
    "curriculum_lr_decay_rate": 0.9,
    "curriculum_lr_decay_steps": 5000,
    
    # 课程学习阶段
    "curriculum_stages": [
        {"Re": 10, "epochs": 2000},
        {"Re": 30, "epochs": 2000},
        {"Re": 60, "epochs": 2000},
        {"Re": 100, "epochs": 14000},
    ],
    
    # 训练数据参数
    "N_bc": 200,
    "N_pde": 10000,
    "lambda_bc": 10.0,
    "lambda_pde": 1.0,
    "lambda_cont": 1.0,
    
    # 数值求解参数
    "fdm_grid_size": 64,
    "fdm_max_iter": 10000,
    "fdm_tolerance": 1e-6,
    "fdm_dt": 0.0001,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("\n" + "="*80)
print("EXPERIMENTAL CONFIGURATION")
print("="*80)
print(json.dumps(CONFIG, indent=2))
print("="*80 + "\n")


# ==============================================================================
# 1. 有限差分数值求解器 (Ground Truth)
# ==============================================================================
class FDMSolver:
    """
    使用有限差分法求解2D顶盖驱动空腔流
    方法：涡量-流函数格式 (Vorticity-Stream Function Formulation)
    """
    
    def __init__(self, Re, nx=64, ny=64, dt=0.0001, max_iter=10000, tol=1e-6):
        self.Re = Re
        self.nu = 1.0 / Re
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.max_iter = max_iter
        self.tol = tol
        
        self.Lx = 1.0
        self.Ly = 1.0
        self.dx = self.Lx / (nx - 1)
        self.dy = self.Ly / (ny - 1)
        
        self.x = np.linspace(0, self.Lx, nx)
        self.y = np.linspace(0, self.Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        self.psi = np.zeros((ny, nx))
        self.omega = np.zeros((ny, nx))
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))
        
        self._check_stability()
    
    def _check_stability(self):
        """CFL条件稳定性检查"""
        print("\n" + "-"*60)
        print("Evaluating FDM Stability Conditions (CFL)...")
        print("-"*60)
        print(f"  Space step (dx): {self.dx:.6f}")
        print(f"  Time step (dt): {self.dt:.6f}")
        print("-"*20)
        
        u_max = 1.0
        cfl_advective = u_max * self.dt / self.dx
        print(f"  Advective CFL number = |u|*dt/dx = {cfl_advective:.4f} (Requirement: <= 1.0)")
        
        cfl_diffusive = 2 * self.nu * self.dt / (self.dx**2)
        print(f"  Diffusive CFL number = 2*nu*dt/dx^2 = {cfl_diffusive:.4f} (Requirement: <= 0.5)")
        print("-"*20)
        
        if cfl_advective > 1.0 or cfl_diffusive > 0.5:
            print("  [WARNING] FDM parameters may be UNSTABLE!")
        else:
            print("  [SUCCESS] FDM parameters appear to be STABLE.")
        print("-"*60 + "\n")
    
    def solve_poisson_psi(self):
        """求解泊松方程: ∇²ψ = -ω"""
        nx, ny = self.nx, self.ny
        dx2, dy2 = self.dx**2, self.dy**2
        
        N = (nx - 2) * (ny - 2)
        A = sparse.lil_matrix((N, N))
        b = np.zeros(N)
        
        def idx(i, j):
            return i * (nx - 2) + j
        
        for i in range(ny - 2):
            for j in range(nx - 2):
                k = idx(i, j)
                A[k, k] = -2.0 / dx2 - 2.0 / dy2
                b[k] = -self.omega[i+1, j+1]
                
                if j > 0:
                    A[k, idx(i, j-1)] = 1.0 / dx2
                else:
                    b[k] -= self.psi[i+1, 0] / dx2
                
                if j < nx - 3:
                    A[k, idx(i, j+1)] = 1.0 / dx2
                else:
                    b[k] -= self.psi[i+1, nx-1] / dx2
                
                if i > 0:
                    A[k, idx(i-1, j)] = 1.0 / dy2
                else:
                    b[k] -= self.psi[0, j+1] / dy2
                
                if i < ny - 3:
                    A[k, idx(i+1, j)] = 1.0 / dy2
                else:
                    b[k] -= self.psi[ny-1, j+1] / dy2
        
        A = A.tocsr()
        psi_inner = spsolve(A, b)
        
        for i in range(ny - 2):
            for j in range(nx - 2):
                self.psi[i+1, j+1] = psi_inner[idx(i, j)]
    
    def compute_velocity_from_psi(self):
        """从流函数计算速度"""
        self.u[1:-1, 1:-1] = (self.psi[2:, 1:-1] - self.psi[:-2, 1:-1]) / (2 * self.dy)
        self.v[1:-1, 1:-1] = -(self.psi[1:-1, 2:] - self.psi[1:-1, :-2]) / (2 * self.dx)
        
        self.u[:, 0] = 0.0
        self.u[:, -1] = 0.0
        self.u[0, :] = 0.0
        self.u[-1, :] = 1.0
        
        self.v[:, 0] = 0.0
        self.v[:, -1] = 0.0
        self.v[0, :] = 0.0
        self.v[-1, :] = 0.0
    
    def update_omega_boundary(self):
        """更新边界涡量"""
        dx2, dy2 = self.dx**2, self.dy**2
        
        self.omega[:, 0] = -2.0 * self.psi[:, 1] / dx2
        self.omega[:, -1] = -2.0 * self.psi[:, -2] / dx2
        self.omega[0, :] = -2.0 * self.psi[1, :] / dy2
        self.omega[-1, :] = -2.0 * self.psi[-2, :] / dy2 - 2.0 * 1.0 / self.dy
    
    def solve(self):
        """主求解循环"""
        print(f"\nStarting FDM solver (Ground Truth) for Re = {self.Re}")
        print(f"Grid: {self.nx} x {self.ny}, dt = {self.dt}")
        print("-" * 60)
        
        start_time = time.time()
        
        for iteration in range(self.max_iter):
            omega_old = self.omega.copy()
            
            self.solve_poisson_psi()
            self.compute_velocity_from_psi()
            self._update_omega()
            self.update_omega_boundary()
            
            residual = np.max(np.abs(self.omega - omega_old))
            
            if (iteration + 1) % 500 == 0:
                elapsed = time.time() - start_time
                print(f"Iter {iteration+1:5d}: Residual = {residual:.2e}, Time = {elapsed:.1f}s")
            
            if residual < self.tol:
                elapsed = time.time() - start_time
                print("-" * 60)
                print(f"✓ FDM Converged at iteration {iteration+1}")
                print(f"  Final residual: {residual:.2e}")
                print(f"  Total time: {elapsed:.1f}s")
                print("-" * 60)
                break
        else:
            print(f"\n⚠ Warning: Maximum iterations ({self.max_iter}) reached")
        
        self._compute_pressure()
        return self.u, self.v, self.p
    
    def _update_omega(self):
        """更新涡量"""
        dx, dy, dt = self.dx, self.dy, self.dt
        
        du_omega = np.zeros_like(self.omega)
        dv_omega = np.zeros_like(self.omega)
        
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                if self.u[i, j] > 0:
                    du_omega[i, j] = self.u[i, j] * (self.omega[i, j] - self.omega[i, j-1]) / dx
                else:
                    du_omega[i, j] = self.u[i, j] * (self.omega[i, j+1] - self.omega[i, j]) / dx
                
                if self.v[i, j] > 0:
                    dv_omega[i, j] = self.v[i, j] * (self.omega[i, j] - self.omega[i-1, j]) / dy
                else:
                    dv_omega[i, j] = self.v[i, j] * (self.omega[i+1, j] - self.omega[i, j]) / dy
        
        d2omega = (
            (self.omega[1:-1, 2:] - 2*self.omega[1:-1, 1:-1] + self.omega[1:-1, :-2]) / dx**2 +
            (self.omega[2:, 1:-1] - 2*self.omega[1:-1, 1:-1] + self.omega[:-2, 1:-1]) / dy**2
        )
        
        self.omega[1:-1, 1:-1] += dt * (
            -du_omega[1:-1, 1:-1] - dv_omega[1:-1, 1:-1] + self.nu * d2omega
        )
    
    def _compute_pressure(self):
        """简化压力计算"""
        self.p = -self.psi


# ==============================================================================
# 2. PINN 网络定义
# ==============================================================================
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers_list = nn.ModuleList()
        
        for i in range(len(layers) - 1):
            self.layers_list.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.layers_list.append(nn.Tanh())
        
        for m in self.layers_list:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        for layer in self.layers_list:
            x = layer(x)
        return x


# ==============================================================================
# 3. 标准 PINN 求解器
# ==============================================================================
class StandardPINN:
    def __init__(self, config):
        self.config = config
        self.Re = config["Re_target"]
        self.model = PINN(config["pinn_layers"]).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config["pinn_learning_rate"]
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config["pinn_lr_decay_steps"], 
            gamma=config["pinn_lr_decay_rate"]
        )
        
        self._generate_training_data()
        self.loss_history = []
    
    def _generate_training_data(self):
        """生成训练数据"""
        N_bc = self.config["N_bc"]
        N_pde = self.config["N_pde"]
        
        # 边界点
        x_left = torch.zeros(N_bc, 1)
        y_left = torch.linspace(0, 1, N_bc).reshape(-1, 1)
        
        x_right = torch.ones(N_bc, 1)
        y_right = torch.linspace(0, 1, N_bc).reshape(-1, 1)
        
        x_bottom = torch.linspace(0, 1, N_bc).reshape(-1, 1)
        y_bottom = torch.zeros(N_bc, 1)
        
        x_top = torch.linspace(0, 1, N_bc).reshape(-1, 1)
        y_top = torch.ones(N_bc, 1)
        
        self.X_bc = torch.cat([
            torch.cat([x_left, y_left], dim=1),
            torch.cat([x_right, y_right], dim=1),
            torch.cat([x_bottom, y_bottom], dim=1),
            torch.cat([x_top, y_top], dim=1),
        ], dim=0).to(device).requires_grad_(True)
        
        u_bc = torch.cat([
            torch.zeros(N_bc, 1),
            torch.zeros(N_bc, 1),
            torch.zeros(N_bc, 1),
            torch.ones(N_bc, 1),
        ], dim=0).to(device)
        
        v_bc = torch.zeros(4*N_bc, 1).to(device)
        self.UV_bc = torch.cat([u_bc, v_bc], dim=1)
        
        # PDE配置点
        x_pde = torch.rand(N_pde, 1)
        y_pde = torch.rand(N_pde, 1)
        self.X_pde = torch.cat([x_pde, y_pde], dim=1).to(device).requires_grad_(True)
    
    def pde_loss(self, X):
        """计算PDE残差"""
        X.requires_grad_(True)
        UVP = self.model(X)
        u, v, p = UVP[:, 0:1], UVP[:, 1:2], UVP[:, 2:3]
        
        u_x = torch.autograd.grad(u, X, torch.ones_like(u), create_graph=True)[0][:, 0:1]
        u_y = torch.autograd.grad(u, X, torch.ones_like(u), create_graph=True)[0][:, 1:2]
        v_x = torch.autograd.grad(v, X, torch.ones_like(v), create_graph=True)[0][:, 0:1]
        v_y = torch.autograd.grad(v, X, torch.ones_like(v), create_graph=True)[0][:, 1:2]
        p_x = torch.autograd.grad(p, X, torch.ones_like(p), create_graph=True)[0][:, 0:1]
        p_y = torch.autograd.grad(p, X, torch.ones_like(p), create_graph=True)[0][:, 1:2]
        
        u_xx = torch.autograd.grad(u_x, X, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, X, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
        v_xx = torch.autograd.grad(v_x, X, torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y, X, torch.ones_like(v_y), create_graph=True)[0][:, 1:2]
        
        nu = 1.0 / self.Re
        
        f_u = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        f_v = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        f_cont = u_x + v_y
        
        return f_u, f_v, f_cont
    
    def train(self):
        """训练标准PINN"""
        print(f"\n{'='*80}")
        print(f"EXPERIMENT 1: Standard PINN Training (Re = {self.Re})")
        print(f"{'='*80}")
        print(f"Total epochs: {self.config['pinn_epochs']}")
        print("-" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.config['pinn_epochs']):
            self.optimizer.zero_grad()
            
            UVP_bc = self.model(self.X_bc)
            loss_bc = torch.mean((UVP_bc[:, :2] - self.UV_bc)**2)
            
            f_u, f_v, f_cont = self.pde_loss(self.X_pde)
            loss_pde = torch.mean(f_u**2 + f_v**2)
            loss_cont = torch.mean(f_cont**2)
            
            loss = (self.config["lambda_bc"] * loss_bc + 
                   self.config["lambda_pde"] * loss_pde + 
                   self.config["lambda_cont"] * loss_cont)
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            self.loss_history.append(loss.item())
            
            if (epoch + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1:5d}: Loss = {loss.item():.4e}, "
                      f"BC = {loss_bc.item():.2e}, PDE = {loss_pde.item():.2e}, "
                      f"Cont = {loss_cont.item():.2e}, Time = {elapsed:.1f}s")
        
        elapsed = time.time() - start_time
        print("-" * 60)
        print(f"✓ Standard PINN training completed in {elapsed:.1f}s")
        print(f"{'='*80}\n")
    
    def predict(self, X):
        """预测"""
        with torch.no_grad():
            return self.model(X).cpu().numpy()


# ==============================================================================
# 4. Curriculum PINN 求解器
# ==============================================================================
class CurriculumPINN:
    def __init__(self, config):
        self.config = config
        self.model = PINN(config["curriculum_layers"]).to(device)
        self.loss_history = []
        self.Re_history = []
        
        self._generate_training_data()
    
    def _generate_training_data(self):
        """生成训练数据"""
        N_bc = self.config["N_bc"]
        N_pde = self.config["N_pde"]
        
        x_left = torch.zeros(N_bc, 1)
        y_left = torch.linspace(0, 1, N_bc).reshape(-1, 1)
        
        x_right = torch.ones(N_bc, 1)
        y_right = torch.linspace(0, 1, N_bc).reshape(-1, 1)
        
        x_bottom = torch.linspace(0, 1, N_bc).reshape(-1, 1)
        y_bottom = torch.zeros(N_bc, 1)
        
        x_top = torch.linspace(0, 1, N_bc).reshape(-1, 1)
        y_top = torch.ones(N_bc, 1)
        
        self.X_bc = torch.cat([
            torch.cat([x_left, y_left], dim=1),
            torch.cat([x_right, y_right], dim=1),
            torch.cat([x_bottom, y_bottom], dim=1),
            torch.cat([x_top, y_top], dim=1),
        ], dim=0).to(device).requires_grad_(True)
        
        u_bc = torch.cat([
            torch.zeros(N_bc, 1),
            torch.zeros(N_bc, 1),
            torch.zeros(N_bc, 1),
            torch.ones(N_bc, 1),
        ], dim=0).to(device)
        
        v_bc = torch.zeros(4*N_bc, 1).to(device)
        self.UV_bc = torch.cat([u_bc, v_bc], dim=1)
        
        x_pde = torch.rand(N_pde, 1)
        y_pde = torch.rand(N_pde, 1)
        self.X_pde = torch.cat([x_pde, y_pde], dim=1).to(device).requires_grad_(True)
    
    def pde_loss(self, X, Re):
        """计算PDE残差（参数化雷诺数）"""
        X.requires_grad_(True)
        UVP = self.model(X)
        u, v, p = UVP[:, 0:1], UVP[:, 1:2], UVP[:, 2:3]
        
        u_x = torch.autograd.grad(u, X, torch.ones_like(u), create_graph=True)[0][:, 0:1]
        u_y = torch.autograd.grad(u, X, torch.ones_like(u), create_graph=True)[0][:, 1:2]
        v_x = torch.autograd.grad(v, X, torch.ones_like(v), create_graph=True)[0][:, 0:1]
        v_y = torch.autograd.grad(v, X, torch.ones_like(v), create_graph=True)[0][:, 1:2]
        p_x = torch.autograd.grad(p, X, torch.ones_like(p), create_graph=True)[0][:, 0:1]
        p_y = torch.autograd.grad(p, X, torch.ones_like(p), create_graph=True)[0][:, 1:2]
        
        u_xx = torch.autograd.grad(u_x, X, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, X, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
        v_xx = torch.autograd.grad(v_x, X, torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y, X, torch.ones_like(v_y), create_graph=True)[0][:, 1:2]
        
        nu = 1.0 / Re
        
        f_u = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        f_v = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        f_cont = u_x + v_y
        
        return f_u, f_v, f_cont
    
    def train(self):
        """使用课程学习训练PINN"""
        print(f"\n{'='*80}")
        print(f"EXPERIMENT 2: Curriculum PINN Training")
        print(f"{'='*80}")
        print(f"Total stages: {len(self.config['curriculum_stages'])}")
        print(f"Total epochs: {self.config['curriculum_total_epochs']}")
        print("-" * 60)
        
        global_start_time = time.time()
        
        for stage_idx, stage in enumerate(self.config['curriculum_stages']):
            Re_current = stage["Re"]
            epochs = stage["epochs"]
            
            print(f"\n{'─'*60}")
            print(f"STAGE {stage_idx + 1}/{len(self.config['curriculum_stages'])}: Re = {Re_current}, Epochs = {epochs}")
            print(f"{'─'*60}")
            
            # 为每个阶段创建新的优化器
            optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.config["curriculum_learning_rate"]
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=self.config["curriculum_lr_decay_steps"], 
                gamma=self.config["curriculum_lr_decay_rate"]
            )
            
            stage_start_time = time.time()
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                UVP_bc = self.model(self.X_bc)
                loss_bc = torch.mean((UVP_bc[:, :2] - self.UV_bc)**2)
                
                f_u, f_v, f_cont = self.pde_loss(self.X_pde, Re_current)
                loss_pde = torch.mean(f_u**2 + f_v**2)
                loss_cont = torch.mean(f_cont**2)
                
                loss = (self.config["lambda_bc"] * loss_bc + 
                       self.config["lambda_pde"] * loss_pde + 
                       self.config["lambda_cont"] * loss_cont)
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                self.loss_history.append(loss.item())
                self.Re_history.append(Re_current)
                
                if (epoch + 1) % 500 == 0:
                    elapsed = time.time() - stage_start_time
                    print(f"  Epoch {epoch+1:5d}/{epochs}: Loss = {loss.item():.4e}, "
                          f"BC = {loss_bc.item():.2e}, PDE = {loss_pde.item():.2e}, "
                          f"Cont = {loss_cont.item():.2e}, Time = {elapsed:.1f}s")
            
            stage_elapsed = time.time() - stage_start_time
            print(f"✓ Stage {stage_idx + 1} completed in {stage_elapsed:.1f}s")
        
        total_elapsed = time.time() - global_start_time
        print(f"\n{'-'*60}")
        print(f"✓ Curriculum PINN training completed in {total_elapsed:.1f}s")
        print(f"{'='*80}\n")
    
    def predict(self, X):
        """预测"""
        with torch.no_grad():
            return self.model(X).cpu().numpy()


# ==============================================================================
# 5. 实验结果对比与可视化
# ==============================================================================
def compare_all_methods(fdm_solver, standard_pinn, curriculum_pinn, resolution=100):
    """对比FDM、标准PINN和Curriculum PINN的结果"""
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE COMPARISON")
    print("="*80)
    
    # 创建预测网格
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X_grid, Y_grid = np.meshgrid(x, y)
    
    # FDM解 (Ground Truth)
    fdm_u_interp = RectBivariateSpline(fdm_solver.y, fdm_solver.x, fdm_solver.u)
    fdm_v_interp = RectBivariateSpline(fdm_solver.y, fdm_solver.x, fdm_solver.v)
    fdm_p_interp = RectBivariateSpline(fdm_solver.y, fdm_solver.x, fdm_solver.p)
    
    U_fdm = fdm_u_interp(y, x)
    V_fdm = fdm_v_interp(y, x)
    P_fdm = fdm_p_interp(y, x)
    
    # 标准PINN解
    X_flat = torch.tensor(np.c_[X_grid.ravel(), Y_grid.ravel()], dtype=torch.float32).to(device)
    UVP_standard = standard_pinn.predict(X_flat)
    
    U_standard = UVP_standard[:, 0].reshape(resolution, resolution)
    V_standard = UVP_standard[:, 1].reshape(resolution, resolution)
    P_standard = UVP_standard[:, 2].reshape(resolution, resolution)
    
    # Curriculum PINN解
    UVP_curriculum = curriculum_pinn.predict(X_flat)
    
    U_curriculum = UVP_curriculum[:, 0].reshape(resolution, resolution)
    V_curriculum = UVP_curriculum[:, 1].reshape(resolution, resolution)
    P_curriculum = UVP_curriculum[:, 2].reshape(resolution, resolution)
    
    # 计算误差
    error_u_standard = np.abs(U_fdm - U_standard)
    error_v_standard = np.abs(V_fdm - V_standard)
    
    error_u_curriculum = np.abs(U_fdm - U_curriculum)
    error_v_curriculum = np.abs(V_fdm - V_curriculum)
    
    # ========== 可视化 ==========
    fig = plt.figure(figsize=(20, 12))
    
    # 定义子图布局
    titles = [
        'Ground Truth (FDM)\nu velocity',
        'Standard PINN\nu velocity',
        'Curriculum PINN\nu velocity',
        'Standard PINN\nError in u',
        'Curriculum PINN\nError in u',
        
        'Ground Truth (FDM)\nv velocity',
        'Standard PINN\nv velocity',
        'Curriculum PINN\nv velocity',
        'Standard PINN\nError in v',
        'Curriculum PINN\nError in v',
        
        'Ground Truth (FDM)\nPressure',
        'Standard PINN\nPressure',
        'Curriculum PINN\nPressure',
        'Velocity Magnitude\n(Ground Truth)',
        'Training Loss\nComparison',
    ]
    
    data = [
        U_fdm, U_standard, U_curriculum, error_u_standard, error_u_curriculum,
        V_fdm, V_standard, V_curriculum, error_v_standard, error_v_curriculum,
        P_fdm, P_standard, P_curriculum, np.sqrt(U_fdm**2 + V_fdm**2), None
    ]
    
    for i in range(15):
        ax = plt.subplot(3, 5, i+1)
        
        if i == 14:  # 损失曲线
            ax.semilogy(standard_pinn.loss_history, label='Standard PINN', linewidth=2)
            ax.semilogy(curriculum_pinn.loss_history, label='Curriculum PINN', linewidth=2)
            
            # 标记课程学习阶段
            epoch_count = 0
            for stage in CONFIG['curriculum_stages']:
                epoch_count += stage['epochs']
                ax.axvline(epoch_count, color='red', linestyle='--', alpha=0.5)
                ax.text(epoch_count, ax.get_ylim()[1]*0.5, 
                       f"Re={stage['Re']}", rotation=90, fontsize=8)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss Comparison', fontsize=10, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            im = ax.contourf(X_grid, Y_grid, data[i], levels=20, cmap='jet')
            ax.set_title(titles[i], fontsize=9, fontweight='bold')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # 添加流线（仅对速度场）
            if i < 3 or (5 <= i < 8):
                row = i // 5
                if i % 5 < 3:
                    if row == 0:
                        U_plot, V_plot = data[i], data[i+5]
                    else:
                        U_plot, V_plot = data[i], data[i]  # 简化处理
                    ax.streamplot(X_grid, Y_grid, U_plot, V_fdm if row == 0 else V_fdm, 
                                 color='white', linewidth=0.5, density=1.2, arrowsize=0.6)
    
    plt.suptitle(f'Comprehensive Comparison: FDM vs Standard PINN vs Curriculum PINN (Re = {CONFIG["Re_target"]})', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('comprehensive_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Comparison plot saved as 'comprehensive_comparison.png'")
    
    # ========== 定量分析 ==========
    print("\n" + "="*80)
    print("QUANTITATIVE ERROR ANALYSIS")
    print("="*80)
    
    print("\n--- Standard PINN ---")
    l2_u_std = np.linalg.norm(error_u_standard) / np.linalg.norm(U_fdm)
    l2_v_std = np.linalg.norm(error_v_standard) / np.linalg.norm(V_fdm)
    max_u_std = np.max(error_u_standard)
    max_v_std = np.max(error_v_standard)
    
    print(f"L2 Relative Error in u: {l2_u_std:.6f}")
    print(f"L2 Relative Error in v: {l2_v_std:.6f}")
    print(f"Max Absolute Error in u: {max_u_std:.6f}")
    print(f"Max Absolute Error in v: {max_v_std:.6f}")
    
    print("\n--- Curriculum PINN ---")
    l2_u_cur = np.linalg.norm(error_u_curriculum) / np.linalg.norm(U_fdm)
    l2_v_cur = np.linalg.norm(error_v_curriculum) / np.linalg.norm(V_fdm)
    max_u_cur = np.max(error_u_curriculum)
    max_v_cur = np.max(error_v_curriculum)
    
    print(f"L2 Relative Error in u: {l2_u_cur:.6f}")
    print(f"L2 Relative Error in v: {l2_v_cur:.6f}")
    print(f"Max Absolute Error in u: {max_u_cur:.6f}")
    print(f"Max Absolute Error in v: {max_v_cur:.6f}")
    
    print("\n--- Improvement by Curriculum Learning ---")
    improvement_l2_u = (l2_u_std - l2_u_cur) / l2_u_std * 100
    improvement_l2_v = (l2_v_std - l2_v_cur) / l2_v_std * 100
    improvement_max_u = (max_u_std - max_u_cur) / max_u_std * 100
    improvement_max_v = (max_v_std - max_v_cur) / max_v_std * 100
    
    print(f"L2 Error Reduction in u: {improvement_l2_u:+.2f}%")
    print(f"L2 Error Reduction in v: {improvement_l2_v:+.2f}%")
    print(f"Max Error Reduction in u: {improvement_max_u:+.2f}%")
    print(f"Max Error Reduction in v: {improvement_max_v:+.2f}%")
    
    print("="*80 + "\n")
    
    # ========== 沿中心线的对比 ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 水平中心线 (y=0.5)
    y_mid_idx = resolution // 2
    axes[0].plot(x, U_fdm[y_mid_idx, :], 'k-', linewidth=2, label='FDM (Ground Truth)')
    axes[0].plot(x, U_standard[y_mid_idx, :], 'b--', linewidth=2, label='Standard PINN')
    axes[0].plot(x, U_curriculum[y_mid_idx, :], 'r:', linewidth=2, label='Curriculum PINN')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u velocity')
    axes[0].set_title('Horizontal Centerline (y = 0.5)', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 垂直中心线 (x=0.5)
    x_mid_idx = resolution // 2
    axes[1].plot(V_fdm[:, x_mid_idx], y, 'k-', linewidth=2, label='FDM (Ground Truth)')
    axes[1].plot(V_standard[:, x_mid_idx], y, 'b--', linewidth=2, label='Standard PINN')
    axes[1].plot(V_curriculum[:, x_mid_idx], y, 'r:', linewidth=2, label='Curriculum PINN')
    axes[1].set_xlabel('v velocity')
    axes[1].set_ylabel('y')
    axes[1].set_title('Vertical Centerline (x = 0.5)', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('centerline_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Centerline comparison saved as 'centerline_comparison.png'")
    
    plt.show()


# ==============================================================================
# 6. 主程序
# ==============================================================================
def main():
    print("\n" + "="*80)
    print("EXPERIMENTAL STUDY: FDM vs Standard PINN vs Curriculum PINN")
    print("="*80 + "\n")
    
    # STEP 1: FDM求解 (Ground Truth)
    print(f"{'='*80}")
    print("STEP 1: Solving with Finite Difference Method (Ground Truth)")
    print(f"{'='*80}")
    fdm_solver = FDMSolver(
        Re=CONFIG["Re_target"],
        nx=CONFIG["fdm_grid_size"],
        ny=CONFIG["fdm_grid_size"],
        dt=CONFIG["fdm_dt"],
        max_iter=CONFIG["fdm_max_iter"],
        tol=CONFIG["fdm_tolerance"]
    )
    fdm_solver.solve()
    
    # STEP 2: 标准PINN训练
    standard_pinn = StandardPINN(CONFIG)
    standard_pinn.train()
    
    # STEP 3: Curriculum PINN训练
    curriculum_pinn = CurriculumPINN(CONFIG)
    curriculum_pinn.train()
    
    # STEP 4: 综合对比
    print(f"\n{'='*80}")
    print("STEP 4: Comprehensive Comparison and Visualization")
    print(f"{'='*80}")
    compare_all_methods(fdm_solver, standard_pinn, curriculum_pinn, resolution=100)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()