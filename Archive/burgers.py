import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import sys

# ==============================================================================
# 0. 配置中心 (Configuration Hub)
# ==============================================================================
# 将所有超参数集中在此，方便管理和复现实验
CONFIG = {
    # 物理问题参数
    "nu": 0.002,  # 粘性系数 (尝试 0.01/np.pi, 0.005, 0.001)
    "x_domain": [-1.0, 1.0],
    "t_domain": [0.0, 1.0],

    # 神经网络结构
    "layers": [2, 32, 32, 32, 32, 32, 1],

    # 训练参数
    "epochs": 20000,
    "learning_rate": 5e-4,
    
    # 课程学习参数
    "curriculum_ramp_ratio": 0.4, # beta从0到1的爬坡阶段占总训练时长的比例

    # 数据点数量
    "N_ic": 100,  # 初始条件点
    "N_bc": 100,  # 边界条件点
    "N_pde": 10000, # PDE配置点

    # 数值解参数 (用于生成基准真相)
    "NX": 512, # 空间网格点数
    "NT": 10000, # 时间步数
    
    # 可视化参数
    "plot_t_final": 0.01, # 最终用于绘图的时间点
    "plot_x_points": 512, # 绘图时的空间分辨率
}

# 设置设备 (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("\nExperiment Configuration:")
print(json.dumps(CONFIG, indent=2))
print("-" * 50)


# ==============================================================================
# 1. FDM 参数预评估 (FDM Parameter Pre-evaluation)
# ==============================================================================
def evaluate_fdm_params(config):
    """
    在运行FDM求解器之前，评估 dx 和 dt 是否满足CFL稳定性条件。
    """
    print("Evaluating FDM stability conditions (CFL)...")
    nx = config["NX"]
    nt = config["NT"]
    x_domain = config["x_domain"]
    t_domain = config["t_domain"]
    nu = config["nu"]

    # 计算 dx 和 dt
    dx = (x_domain[1] - x_domain[0]) / (nx - 1)
    dt = (t_domain[1] - t_domain[0]) / (nt - 1)
    
    # 估计最大速度 |u|_max。对于初始条件 u(x,0) = -sin(pi*x)，最大绝对值为 1.0。
    # 这是一个合理的保守估计。
    u_max_est = 1.0

    # 计算对流和扩散CFL数
    cfl_adv = u_max_est * dt / dx
    cfl_diff = 2 * nu * dt / (dx**2)

    print(f"  Space step (dx): {dx:.6f}")
    print(f"  Time step (dt): {dt:.6f}")
    print("-" * 20)
    print(f"  Advective CFL number = |u|*dt/dx = {cfl_adv:.4f} (Requirement: <= 1.0)")
    print(f"  Diffusive CFL number = 2*nu*dt/dx^2 = {cfl_diff:.4f} (Requirement: <= 0.5)")
    print("-" * 20)

    # 检查稳定性
    is_stable = True
    if cfl_adv > 1.0:
        print("  [STABILITY WARNING] Advective CFL condition not met! Solution may be unstable.")
        is_stable = False
    if cfl_diff > 0.5:
        print("  [STABILITY WARNING] Diffusive CFL condition not met! Solution may be unstable.")
        is_stable = False
    
    if is_stable:
        print("  [SUCCESS] FDM parameters appear to be STABLE.")
    else:
        print("  [FAILURE] FDM parameters are UNSTABLE. Please increase NT or decrease NX.")
    
    print("-" * 50)
    return is_stable


# ==============================================================================
# 1.5. 数值求解器 (Finite Difference Method for Ground Truth)
# ==============================================================================
def solve_burgers_fdm(config):
    """
    使用有限差分法求解一维伯格斯方程，作为基准解。
    u_t + u * u_x = nu * u_xx
    """
    nx = config["NX"]
    nt = config["NT"]
    x_domain = config["x_domain"]
    t_domain = config["t_domain"]
    nu = config["nu"]

    x = np.linspace(x_domain[0], x_domain[1], nx)
    t = np.linspace(t_domain[0], t_domain[1], nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # 初始化解数组
    u = np.zeros((nx, nt))

    # 初始条件: u(x, 0) = -sin(pi*x)
    u[:, 0] = -np.sin(np.pi * x)

    # 边界条件: u(-1, t) = 0, u(1, t) = 0 (已在初始化时满足)
    
    print("Generating numerical solution using FDM...")
    start_time = time.time()
    # 时间步进 (显式格式)
    for n in range(nt - 1):
        un = u[:, n].copy()
        for i in range(1, nx - 1):
            # 使用中心差分计算空间导数
            u_x = (un[i+1] - un[i-1]) / (2 * dx)
            u_xx = (un[i+1] - 2 * un[i] + un[i-1]) / (dx**2)
            u[i, n+1] = un[i] - dt * (un[i] * u_x - nu * u_xx)
    
    print(f"FDM solver finished in {time.time() - start_time:.2f} seconds.")
    print("-" * 50)
    return u, x, t


# ==============================================================================
# 2. 神经网络模型定义
# ==============================================================================
class PINN_Net(nn.Module):
    """一个简单的多层感知机 (MLP)"""
    def __init__(self, layers):
        super(PINN_Net, self).__init__()
        self.activation = nn.Tanh()
        self.layers_list = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers_list.append(nn.Linear(layers[i], layers[i+1]))

    def forward(self, x):
        if x.dim() == 1:
            x = x.view(-1, x.shape[0])
        for i in range(len(self.layers_list) - 1):
            x = self.activation(self.layers_list[i](x))
        return self.layers_list[-1](x)


# ==============================================================================
# 3. PINN 求解器类
# ==============================================================================
class PINNSolver:
    def __init__(self, config, curriculum=False):
        self.config = config
        self.nu = self.config["nu"]
        self.epochs = self.config["epochs"]
        
        self.curriculum = curriculum
        self.ramp_epochs = int(self.epochs * self.config["curriculum_ramp_ratio"]) if curriculum else 0
        self.beta = 0.0 if curriculum else 1.0
        
        self.model = PINN_Net(self.config["layers"]).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        
        self.loss_history = []
        self.beta_history = []

        self._generate_data()

    def _generate_data(self):
        """根据配置生成训练数据点"""
        x_domain = self.config["x_domain"]
        t_domain = self.config["t_domain"]
        
        # 初始条件 (t=0)
        x_ic = torch.linspace(x_domain[0], x_domain[1], self.config["N_ic"]).view(-1, 1)
        t_ic = torch.zeros_like(x_ic)
        self.X_ic = torch.cat([x_ic, t_ic], dim=1).to(device)
        self.u_ic = -torch.sin(np.pi * x_ic).to(device)

        # 边界条件 (x=-1 and x=1)
        t_bc = torch.linspace(t_domain[0], t_domain[1], self.config["N_bc"]).view(-1, 1)
        x_bc_left = torch.full_like(t_bc, x_domain[0])
        x_bc_right = torch.full_like(t_bc, x_domain[1])
        self.X_bc_left = torch.cat([x_bc_left, t_bc], dim=1).to(device)
        self.X_bc_right = torch.cat([x_bc_right, t_bc], dim=1).to(device)
        self.u_bc = torch.zeros_like(t_bc).to(device)

        # PDE残差的配置点
        t_pde = torch.rand(self.config["N_pde"], 1, device=device) * t_domain[1]
        x_pde = torch.rand(self.config["N_pde"], 1, device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
        self.X_pde = torch.cat([x_pde, t_pde], dim=1)
        self.X_pde.requires_grad = True

    def pde_residual(self, X):
        """计算PDE残差 f = u_t + beta * u * u_x - nu * u_xx"""
        x, t = X[:, 0:1], X[:, 1:2]
        u = self.model(torch.cat([x, t], dim=1))

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        residual = u_t + self.beta * (u * u_x) - self.nu * u_xx
        return residual

    def loss_fn(self):
        """计算总损失"""
        u_pred_ic = self.model(self.X_ic)
        loss_ic = torch.mean((u_pred_ic - self.u_ic) ** 2)

        u_pred_bc_left = self.model(self.X_bc_left)
        u_pred_bc_right = self.model(self.X_bc_right)
        loss_bc = torch.mean((u_pred_bc_left - self.u_bc) ** 2) + torch.mean((u_pred_bc_right - self.u_bc) ** 2)

        pde_res = self.pde_residual(self.X_pde)
        loss_pde = torch.mean(pde_res ** 2)

        total_loss = loss_pde + loss_ic + loss_bc
        return total_loss, loss_ic, loss_bc, loss_pde

    def train(self):
        """训练循环"""
        self.model.train()
        start_time = time.time()

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            total_loss, loss_ic, loss_bc, loss_pde = self.loss_fn()
            total_loss.backward()
            self.optimizer.step()
            
            self.loss_history.append(total_loss.item())
            self.beta_history.append(self.beta)

            # 更新课程学习参数 beta
            if self.curriculum and epoch < self.ramp_epochs:
                self.beta = min(1.0, (epoch + 1) / self.ramp_epochs)

            if (epoch + 1) % 2000 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss.item():.4e}, Beta: {self.beta:.3f}")
        
        elapsed_time = time.time() - start_time
        print(f"Training finished in {elapsed_time:.2f} seconds.")

    def predict(self, X):
        """用训练好的模型进行预测"""
        self.model.eval()
        with torch.no_grad():
            u_pred = self.model(X)
        return u_pred.cpu().numpy()


# ==============================================================================
# 4. 可视化工具
# ==============================================================================
def plot_comprehensive_results(results):
    """
    一个功能强大的绘图函数，用于生成和展示所有对比结果。
    `results` 是一个包含所有实验数据的字典。
    """
    config = results['config']
    nu = config['nu']
    t_final = config['plot_t_final']
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'PINN Performance for 1D Burgers Equation (ν={nu:.2e})', fontsize=16)

    # --- 1. 损失曲线对比 ---
    ax = axes[0, 0]
    ax.plot(results['standard']['solver'].loss_history, label=results['standard']['label'], alpha=0.7)
    ax.plot(results['curriculum']['solver'].loss_history, label=results['curriculum']['label'], linewidth=2)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Loss Curve Comparison')
    ax.legend()
    ax.grid(True, which="both", ls="--")

    # --- 2. Beta 演变过程 ---
    ax = axes[0, 1]
    ax.plot(results['curriculum']['solver'].beta_history, color='green', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('β (Non-linearity factor)')
    ax.set_title('Curriculum Learning: β Annealing Schedule')
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, ls="--")

    # --- 3. 最终解的对比 ---
    ax = axes[1, 0]
    x_test = torch.linspace(config['x_domain'][0], config['x_domain'][1], config['plot_x_points']).view(-1, 1)
    t_test = torch.full_like(x_test, t_final)
    X_test_tensor = torch.cat([x_test, t_test], dim=1).to(device)
    
    u_pred_std = results['standard']['solver'].predict(X_test_tensor)
    u_pred_curr = results['curriculum']['solver'].predict(X_test_tensor)
    
    # 从数值解中提取对应时间片的解
    exact_data = results['exact']
    t_idx = np.argmin(np.abs(exact_data['t'] - t_final))
    u_exact_final = exact_data['solution'][:, t_idx]

    ax.plot(exact_data['x'], u_exact_final, 'k-', label=exact_data['label'], linewidth=3, alpha=0.8)
    ax.plot(x_test.numpy(), u_pred_std, 'b--', label=results['standard']['label'])
    ax.plot(x_test.numpy(), u_pred_curr, 'r-.', label=results['curriculum']['label'], linewidth=2)
    
    ax.set_xlabel('x')
    ax.set_ylabel(f'u(x, t={t_final})')
    ax.set_title(f'Solution Comparison at t={t_final}')
    ax.legend()
    ax.grid(True, ls="--")

    # --- 4. 绝对误差对比 ---
    ax = axes[1, 1]
    # 需要将数值解插值到与PINN预测相同的点上
    u_exact_interp = np.interp(x_test.numpy().flatten(), exact_data['x'], u_exact_final)
    
    error_std = np.abs(u_pred_std.flatten() - u_exact_interp)
    error_curr = np.abs(u_pred_curr.flatten() - u_exact_interp)
    
    ax.plot(x_test.numpy(), error_std, 'b--', label=f"Error ({results['standard']['label']})")
    ax.plot(x_test.numpy(), error_curr, 'r-.', label=f"Error ({results['curriculum']['label']})", linewidth=2)
    
    ax.set_yscale('log')
    ax.set_xlabel('x')
    ax.set_ylabel('Absolute Error')
    ax.set_title(f'Absolute Error vs. Numerical Solution at t={t_final}')
    ax.legend()
    ax.grid(True, which="both", ls="--")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('pinn_comprehensive_comparison.png', dpi=300)
    plt.show()


# ==============================================================================
# 5. 主执行流程
# ==============================================================================
if __name__ == "__main__":
    # 0. 在开始任何昂贵的计算之前，首先评估FDM参数的稳定性
    is_stable = evaluate_fdm_params(CONFIG)
    if not is_stable:
        print("Aborting experiment due to unstable FDM parameters.")
        sys.exit() # 终止脚本

    # 1. 生成数值基准解
    u_exact, x_grid, t_grid = solve_burgers_fdm(CONFIG)
    
    # 2. 训练标准 PINN
    print("\n--- Starting Standard PINN Training ---")
    pinn_standard = PINNSolver(CONFIG, curriculum=False)
    pinn_standard.train()
    print("-" * 50)

    # 3. 训练课程学习 PINN
    print("\n--- Starting Curriculum PINN Training ---")
    pinn_curriculum = PINNSolver(CONFIG, curriculum=True)
    pinn_curriculum.train()
    print("-" * 50)

    # 4. 收集所有结果并进行可视化
    print("\n--- Plotting Comprehensive Results ---")
    results_dict = {
        'config': CONFIG,
        'standard': {
            'solver': pinn_standard,
            'label': 'Standard PINN'
        },
        'curriculum': {
            'solver': pinn_curriculum,
            'label': 'Curriculum PINN'
        },
        'exact': {
            'solution': u_exact,
            'x': x_grid,
            't': t_grid,
            'label': 'Numerical (FDM)'
        }
    }
    plot_comprehensive_results(results_dict)