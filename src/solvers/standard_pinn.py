"""
Standard PINN Solver for Burgers and Navier-Stokes Equations
"""
import torch
import numpy as np
from .base_solver import BasePINNSolver


class BurgersPINN(BasePINNSolver):
    """
    Standard PINN solver for 1D Burgers Equation:
    u_t + u * u_x = nu * u_xx
    """
    
    def __init__(self, model, config, device='cuda'):
        self.nu = config.get('nu', 0.01)
        super().__init__(model, config, device)
        self._generate_training_data()
    
    def _generate_training_data(self):
        """Generate training data for Burgers equation"""
        x_domain = self.config.get('x_domain', [-1.0, 1.0])
        t_domain = self.config.get('t_domain', [0.0, 1.0])
        
        N_ic = self.config.get('N_ic', 100)
        N_bc = self.config.get('N_bc', 100)
        N_pde = self.config.get('N_pde', 10000)
        
        # ========== Initial condition (t=0) ==========
        x_ic = torch.linspace(x_domain[0], x_domain[1], N_ic).view(-1, 1)
        t_ic = torch.zeros_like(x_ic)
        self.X_ic = torch.cat([x_ic, t_ic], dim=1).to(self.device)
        self.u_ic = -torch.sin(np.pi * x_ic).to(self.device)
        
        # ========== Boundary conditions ==========
        t_bc = torch.linspace(t_domain[0], t_domain[1], N_bc).view(-1, 1)
        x_bc_left = torch.full_like(t_bc, x_domain[0])
        x_bc_right = torch.full_like(t_bc, x_domain[1])
        
        X_bc_left = torch.cat([x_bc_left, t_bc], dim=1)
        X_bc_right = torch.cat([x_bc_right, t_bc], dim=1)
        self.X_bc = torch.cat([X_bc_left, X_bc_right], dim=0).to(self.device)
        self.u_bc = torch.zeros(2 * N_bc, 1).to(self.device)
        
        # ========== PDE collocation points ==========
        t_pde = torch.rand(N_pde, 1) * (t_domain[1] - t_domain[0]) + t_domain[0]
        x_pde = torch.rand(N_pde, 1) * (x_domain[1] - x_domain[0]) + x_domain[0]
        self.X_pde = torch.cat([x_pde, t_pde], dim=1).to(self.device)
        self.X_pde.requires_grad = True
        
        # ========== Data points (从FDM采样) ==========
        if self.config.get('use_data_loss', False):
            N_data = self.config.get('N_data', 500)
            fdm_solution = self.config.get('fdm_solution', None)
            
            if fdm_solution is not None:
                print(f"Loading {N_data} data points from FDM solution...")
                
                # 随机采样时空点
                x_data = torch.rand(N_data, 1) * (x_domain[1] - x_domain[0]) + x_domain[0]
                t_data = torch.rand(N_data, 1) * (t_domain[1] - t_domain[0]) + t_domain[0]
                self.X_data = torch.cat([x_data, t_data], dim=1).to(self.device)
                
                # 从FDM插值得到对应的u值
                self.u_data = self._interpolate_fdm_data(
                    x_data.cpu().numpy(), 
                    t_data.cpu().numpy(), 
                    fdm_solution
                )
                
                print(f"✓ Generated {N_data} data points from FDM solution")
            else:
                print("⚠ Warning: use_data_loss=True but no fdm_solution provided")
                self.X_data = None
                self.u_data = None
        else:
            self.X_data = None
            self.u_data = None
    
    def _interpolate_fdm_data(self, x_query, t_query, fdm_solution):
        """
        从FDM数据中插值得到查询点的值
        
        Args:
            x_query: 查询点的x坐标 [N, 1] numpy array
            t_query: 查询点的t坐标 [N, 1] numpy array
            fdm_solution: FDM求解器返回的 (u, x, t) 元组
        
        Returns:
            u_interp: 插值后的u值 [N, 1] torch tensor
        """
        from scipy.interpolate import RegularGridInterpolator
        
        # 解包FDM solution
        u_grid, x_grid, t_grid = fdm_solution
        
        # 创建2D插值器
        # 注意：u_grid的形状是 [nx, nt]，与 (x, t) 对应
        interpolator = RegularGridInterpolator(
            (x_grid, t_grid), 
            u_grid,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        # 准备查询点 [N, 2]: 每行是 [x, t]
        points = np.concatenate([x_query, t_query], axis=1)
        
        # 插值
        u_interp = interpolator(points)
        
        # 转换为torch tensor
        return torch.tensor(u_interp, dtype=torch.float32).reshape(-1, 1).to(self.device)
    
    def pde_residual(self, X):
        """
        Compute PDE residual: f = u_t + u * u_x - nu * u_xx
        
        Args:
            X: Input tensor [x, t]
        
        Returns:
            PDE residual
        """
        # Ensure X requires gradient
        if not X.requires_grad:
            X = X.clone().detach().requires_grad_(True)
        
        # Forward pass
        u = self.model(X)
        
        # Compute gradients with respect to X
        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=X,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        u_x = grad_u[:, 0:1]  # du/dx
        u_t = grad_u[:, 1:2]  # du/dt
        
        # Second derivative with respect to x
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=X,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]  # d²u/dx²
        
        # PDE residual: u_t + u * u_x - nu * u_xx = 0
        residual = u_t + u * u_x - self.nu * u_xx
        
        return residual

class NavierStokesPINN(BasePINNSolver):
    """
    Standard PINN solver for 2D Navier-Stokes Equations (Lid-Driven Cavity):
    u * u_x + v * u_y = -p_x + (1/Re) * (u_xx + u_yy)
    u * v_x + v * v_y = -p_y + (1/Re) * (v_xx + v_yy)
    u_x + v_y = 0
    """
    
    def __init__(self, model, config, device='cuda'):
        self.Re = config.get('Re', 100)
        self.nu = 1.0 / self.Re
        super().__init__(model, config, device)
        self._generate_training_data()
            
    def _generate_training_data(self):
        """Generate training data for Navier-Stokes"""
        x_domain = self.config.get('x_domain', [0.0, 1.0])
        y_domain = self.config.get('y_domain', [0.0, 1.0])
        
        N_bc = self.config.get('N_bc', 400)
        N_pde = self.config.get('N_pde', 10000)
        
        # Boundary conditions
        # Left boundary (x=0)
        x_left = torch.zeros(N_bc, 1)
        y_left = torch.linspace(y_domain[0], y_domain[1], N_bc).reshape(-1, 1)
        
        # Right boundary (x=1)
        x_right = torch.ones(N_bc, 1)
        y_right = torch.linspace(y_domain[0], y_domain[1], N_bc).reshape(-1, 1)
        
        # Bottom boundary (y=0)
        x_bottom = torch.linspace(x_domain[0], x_domain[1], N_bc).reshape(-1, 1)
        y_bottom = torch.zeros(N_bc, 1)
        
        # Top boundary (y=1, lid moving with u=1)
        x_top = torch.linspace(x_domain[0], x_domain[1], N_bc).reshape(-1, 1)
        y_top = torch.ones(N_bc, 1)
        
        # Combine all boundary points
        self.X_bc = torch.cat([
            torch.cat([x_left, y_left], dim=1),
            torch.cat([x_right, y_right], dim=1),
            torch.cat([x_bottom, y_bottom], dim=1),
            torch.cat([x_top, y_top], dim=1),
        ], dim=0).to(self.device)
        
        # Boundary values [u, v, p]
        # All walls have u=0, v=0 except top wall where u=1
        u_bc = torch.cat([
            torch.zeros(N_bc, 1),  # left
            torch.zeros(N_bc, 1),  # right
            torch.zeros(N_bc, 1),  # bottom
            torch.ones(N_bc, 1),   # top (lid)
        ], dim=0).to(self.device)
        
        v_bc = torch.zeros(4 * N_bc, 1).to(self.device)
        
        # We don't enforce pressure BC, only velocity
        self.u_bc = torch.cat([u_bc, v_bc], dim=1)
        
        # PDE collocation points (interior domain)
        x_pde = torch.rand(N_pde, 1) * (x_domain[1] - x_domain[0]) + x_domain[0]
        y_pde = torch.rand(N_pde, 1) * (y_domain[1] - y_domain[0]) + y_domain[0]
        self.X_pde = torch.cat([x_pde, y_pde], dim=1).to(self.device)
        self.X_pde.requires_grad = True
        
        # ========== Data points (从FDM采样) ==========
        if self.config.get('use_data_loss', False):
            N_data = self.config.get('N_data', 500)
            fdm_solver = self.config.get('fdm_solver', None)
            
            if fdm_solver is not None:
                print(f"Loading {N_data} data points from FDM solution...")
                
                # 随机采样空间点
                x_data = torch.rand(N_data, 1) * (x_domain[1] - x_domain[0]) + x_domain[0]
                y_data = torch.rand(N_data, 1) * (y_domain[1] - y_domain[0]) + y_domain[0]
                self.X_data = torch.cat([x_data, y_data], dim=1).to(self.device)
                
                # 从FDM插值得到对应的 [u, v, p] 值
                self.u_data = self._interpolate_fdm_data(
                    x_data.cpu().numpy(), 
                    y_data.cpu().numpy(), 
                    fdm_solver
                )
                
                print(f"✓ Generated {N_data} data points from FDM solution")
            else:
                print("⚠ Warning: use_data_loss=True but no fdm_solver provided")
                self.X_data = None
                self.u_data = None
        else:
            self.X_data = None
            self.u_data = None
    
    def _interpolate_fdm_data(self, x_query, y_query, fdm_solver):
        """
        从FDM数据中插值得到查询点的值
        
        Args:
            x_query: 查询点的x坐标 [N, 1] numpy array
            y_query: 查询点的y坐标 [N, 1] numpy array
            fdm_solver: FDM求解器对象（已经solve过）
        
        Returns:
            uvp_interp: 插值后的 [u, v, p] 值 [N, 3] torch tensor
        """
        from scipy.interpolate import RegularGridInterpolator
        
        # 从FDM solver获取网格和解
        x_grid = fdm_solver.x  # 1D array [nx]
        y_grid = fdm_solver.y  # 1D array [ny]
        u_grid = fdm_solver.u  # 2D array [ny, nx]
        v_grid = fdm_solver.v  # 2D array [ny, nx]
        p_grid = fdm_solver.p  # 2D array [ny, nx]
        
        # 创建2D插值器 (注意：数组是 [ny, nx]，对应 (y, x))
        u_interpolator = RegularGridInterpolator(
            (y_grid, x_grid), 
            u_grid,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        v_interpolator = RegularGridInterpolator(
            (y_grid, x_grid), 
            v_grid,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        p_interpolator = RegularGridInterpolator(
            (y_grid, x_grid), 
            p_grid,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        # 准备查询点 [N, 2]: 每行是 [y, x] (注意顺序！)
        points = np.concatenate([y_query, x_query], axis=1)
        
        # 插值
        u_interp = u_interpolator(points)
        v_interp = v_interpolator(points)
        p_interp = p_interpolator(points)
        
        # 组合成 [u, v, p] 并转换为torch tensor
        uvp_interp = np.stack([u_interp, v_interp, p_interp], axis=1)
        
        return torch.tensor(uvp_interp, dtype=torch.float32).to(self.device)
    
    def loss_bc(self):
        """Override BC loss to handle velocity components only"""
        if self.X_bc is None:
            return torch.tensor(0.0, device=self.device)
        
        UVP_pred = self.model(self.X_bc)
        UV_pred = UVP_pred[:, :2]  # Only u and v
        
        return torch.mean((UV_pred - self.u_bc) ** 2)
    
    def pde_residual(self, X):
        """
        Compute PDE residual for Navier-Stokes equations
        
        Returns:
            Tuple of (f_u, f_v, f_cont) residuals
        """
        # Ensure X requires gradient
        if not X.requires_grad:
            X = X.clone().detach().requires_grad_(True)
        
        # Forward pass
        UVP = self.model(X)
        u = UVP[:, 0:1]
        v = UVP[:, 1:2]
        p = UVP[:, 2:3]
        
        # First derivatives of u
        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=X,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        
        # First derivatives of v
        grad_v = torch.autograd.grad(
            outputs=v,
            inputs=X,
            grad_outputs=torch.ones_like(v),
            create_graph=True,
            retain_graph=True
        )[0]
        v_x = grad_v[:, 0:1]
        v_y = grad_v[:, 1:2]
        
        # First derivatives of p
        grad_p = torch.autograd.grad(
            outputs=p,
            inputs=X,
            grad_outputs=torch.ones_like(p),
            create_graph=True,
            retain_graph=True
        )[0]
        p_x = grad_p[:, 0:1]
        p_y = grad_p[:, 1:2]
        
        # Second derivatives of u
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=X,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]
        
        u_yy = torch.autograd.grad(
            outputs=u_y,
            inputs=X,
            grad_outputs=torch.ones_like(u_y),
            create_graph=True,
            retain_graph=True
        )[0][:, 1:2]
        
        # Second derivatives of v
        v_xx = torch.autograd.grad(
            outputs=v_x,
            inputs=X,
            grad_outputs=torch.ones_like(v_x),
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]
        
        v_yy = torch.autograd.grad(
            outputs=v_y,
            inputs=X,
            grad_outputs=torch.ones_like(v_y),
            create_graph=True,
            retain_graph=True
        )[0][:, 1:2]
        
        # Momentum equations
        f_u = u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        f_v = u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)
        
        # Continuity equation
        f_cont = u_x + v_y
        
        return f_u, f_v, f_cont

    def loss_pde(self):
        """Override PDE loss to handle multiple residuals"""
        if self.X_pde is None:
            return torch.tensor(0.0, device=self.device)
        
        f_u, f_v, f_cont = self.pde_residual(self.X_pde)
        
        loss_momentum = torch.mean(f_u ** 2 + f_v ** 2)
        loss_continuity = torch.mean(f_cont ** 2)
        
        lambda_cont = self.config.get('lambda_cont', 1.0)

        return loss_momentum + lambda_cont * loss_continuity


class CylinderPINN(BasePINNSolver):
    """
    Standard PINN solver for 2D Navier-Stokes around a cylinder
    """
    
    def __init__(self, model, config, device='cuda'):
        self.Re = config.get('Re', 100)
        self.nu = 1.0 / self.Re
        self.U_inf = config.get('U_inf', 1.0)
        
        # Cylinder geometry
        self.cx = config.get('cylinder_center', [0.0, 0.0])[0]
        self.cy = config.get('cylinder_center', [0.0, 0.0])[1]
        self.radius = config.get('cylinder_radius', 0.5)
        
        # Domain
        self.x_domain = config.get('x_domain', [-5.0, 15.0])
        self.y_domain = config.get('y_domain', [-5.0, 5.0])
        
        super().__init__(model, config, device)
        self._generate_training_data()
    
    def _generate_training_data(self):
        """Generate training points"""
        x_min, x_max = self.x_domain
        y_min, y_max = self.y_domain
        
        # Inlet boundary (left)
        N_inlet = self.config.get('N_bc_inlet', 500)
        y_inlet = np.random.uniform(y_min, y_max, N_inlet)
        x_inlet = np.full(N_inlet, x_min)
        self.X_bc_inlet = torch.tensor(np.stack([x_inlet, y_inlet], axis=1), 
                                       dtype=torch.float32, device=self.device)
        self.U_bc_inlet = torch.tensor([[self.U_inf, 0.0, 0.0]] * N_inlet,
                                       dtype=torch.float32, device=self.device)
        
        # Outlet boundary (right)
        N_outlet = self.config.get('N_bc_outlet', 500)
        y_outlet = np.random.uniform(y_min, y_max, N_outlet)
        x_outlet = np.full(N_outlet, x_max)
        self.X_bc_outlet = torch.tensor(np.stack([x_outlet, y_outlet], axis=1),
                                        dtype=torch.float32, device=self.device)
        
        # Top/Bottom walls
        N_wall = self.config.get('N_bc_wall', 200)
        x_top = np.random.uniform(x_min, x_max, N_wall)
        x_bot = np.random.uniform(x_min, x_max, N_wall)
        y_top = np.full(N_wall, y_max)
        y_bot = np.full(N_wall, y_min)
        self.X_bc_wall = torch.tensor(
            np.concatenate([np.stack([x_top, y_top], axis=1),
                           np.stack([x_bot, y_bot], axis=1)]),
            dtype=torch.float32, device=self.device)
        self.U_bc_wall = torch.tensor([[self.U_inf, 0.0, 0.0]] * (2 * N_wall),
                                      dtype=torch.float32, device=self.device)
        
        # Cylinder surface
        N_cyl = self.config.get('N_bc_cylinder', 800)
        theta = np.linspace(0, 2*np.pi, N_cyl)
        x_cyl = self.cx + self.radius * np.cos(theta)
        y_cyl = self.cy + self.radius * np.sin(theta)
        self.X_bc_cylinder = torch.tensor(np.stack([x_cyl, y_cyl], axis=1),
                                         dtype=torch.float32, device=self.device)
        self.U_bc_cylinder = torch.zeros((N_cyl, 3), dtype=torch.float32, device=self.device)
        
        # PDE collocation points (excluding cylinder interior)
        N_pde = self.config.get('N_pde', 20000)
        x_pde = []
        y_pde = []
        n_generated = 0
        
        while n_generated < N_pde:
            x_cand = np.random.uniform(x_min, x_max, N_pde * 2)
            y_cand = np.random.uniform(y_min, y_max, N_pde * 2)
            
            # Filter out points inside cylinder
            dist = np.sqrt((x_cand - self.cx)**2 + (y_cand - self.cy)**2)
            valid = dist > self.radius
            
            x_pde.extend(x_cand[valid])
            y_pde.extend(y_cand[valid])
            n_generated = len(x_pde)
        
        x_pde = np.array(x_pde[:N_pde])
        y_pde = np.array(y_pde[:N_pde])
        self.X_pde = torch.tensor(np.stack([x_pde, y_pde], axis=1),
                                 dtype=torch.float32, device=self.device, requires_grad=True)
        
        # ========== Data points (从FDM采样) ==========
        if self.config.get('use_data_loss', False):
            N_data = self.config.get('N_data', 1000)
            fdm_solver = self.config.get('fdm_solver', None)
            
            if fdm_solver is not None:
                print(f"Loading {N_data} data points from FDM solution...")
                
                # 随机采样空间点（排除圆柱内部）
                x_data_list = []
                y_data_list = []
                n_sampled = 0
                
                while n_sampled < N_data:
                    x_cand = np.random.uniform(x_min, x_max, N_data * 2)
                    y_cand = np.random.uniform(y_min, y_max, N_data * 2)
                    
                    # 排除圆柱内部
                    dist = np.sqrt((x_cand - self.cx)**2 + (y_cand - self.cy)**2)
                    valid = dist > self.radius
                    
                    x_data_list.extend(x_cand[valid])
                    y_data_list.extend(y_cand[valid])
                    n_sampled = len(x_data_list)
                
                x_data = np.array(x_data_list[:N_data]).reshape(-1, 1)
                y_data = np.array(y_data_list[:N_data]).reshape(-1, 1)
                
                self.X_data = torch.cat([
                    torch.tensor(x_data, dtype=torch.float32),
                    torch.tensor(y_data, dtype=torch.float32)
                ], dim=1).to(self.device)
                
                # 从FDM插值得到对应的 [u, v, p] 值
                self.u_data = self._interpolate_fdm_data(x_data, y_data, fdm_solver)
                
                print(f"✓ Generated {N_data} data points from FDM solution")
            else:
                print("⚠ Warning: use_data_loss=True but no fdm_solver provided")
                self.X_data = None
                self.u_data = None
        else:
            self.X_data = None
            self.u_data = None
    
    def _interpolate_fdm_data(self, x_query, y_query, fdm_solver):
        """
        从FDM数据中插值得到查询点的值
        
        Args:
            x_query: 查询点的x坐标 [N, 1] numpy array
            y_query: 查询点的y坐标 [N, 1] numpy array
            fdm_solver: FDM求解器对象（已经solve过）
        
        Returns:
            uvp_interp: 插值后的 [u, v, p] 值 [N, 3] torch tensor
        """
        from scipy.interpolate import RegularGridInterpolator
        
        # 从FDM solver获取网格和解
        x_grid = fdm_solver.x  # 1D array [nx]
        y_grid = fdm_solver.y  # 1D array [ny]
        u_grid = fdm_solver.u  # 2D array [ny, nx]
        v_grid = fdm_solver.v  # 2D array [ny, nx]
        p_grid = fdm_solver.p  # 2D array [ny, nx]
        
        # 创建2D插值器 (注意：数组是 [ny, nx]，对应 (y, x))
        u_interpolator = RegularGridInterpolator(
            (y_grid, x_grid), 
            u_grid,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        v_interpolator = RegularGridInterpolator(
            (y_grid, x_grid), 
            v_grid,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        p_interpolator = RegularGridInterpolator(
            (y_grid, x_grid), 
            p_grid,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        # 准备查询点 [N, 2]: 每行是 [y, x] (注意顺序！)
        points = np.concatenate([y_query, x_query], axis=1)
        
        # 插值
        u_interp = u_interpolator(points)
        v_interp = v_interpolator(points)
        p_interp = p_interpolator(points)
        
        # 组合成 [u, v, p] 并转换为torch tensor
        uvp_interp = np.stack([u_interp, v_interp, p_interp], axis=1)
        
        return torch.tensor(uvp_interp, dtype=torch.float32).to(self.device)
    
    def pde_residual(self, X):
        """Compute Navier-Stokes residual"""
        if not X.requires_grad:
            X = X.clone().detach().requires_grad_(True)
        
        output = self.model(X)
        u = output[:, 0:1]
        v = output[:, 1:2]
        p = output[:, 2:3]
        
        # Compute gradients
        grad_u = torch.autograd.grad(u, X, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        
        grad_v = torch.autograd.grad(v, X, torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        v_x = grad_v[:, 0:1]
        v_y = grad_v[:, 1:2]
        
        grad_p = torch.autograd.grad(p, X, torch.ones_like(p), create_graph=True, retain_graph=True)[0]
        p_x = grad_p[:, 0:1]
        p_y = grad_p[:, 1:2]
        
        u_xx = torch.autograd.grad(u_x, X, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, X, torch.ones_like(u_y), create_graph=True, retain_graph=True)[0][:, 1:2]
        v_xx = torch.autograd.grad(v_x, X, torch.ones_like(v_x), create_graph=True, retain_graph=True)[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y, X, torch.ones_like(v_y), create_graph=True, retain_graph=True)[0][:, 1:2]
        
        # Navier-Stokes equations
        f_u = u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        f_v = u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)
        f_cont = u_x + v_y
        
        return f_u, f_v, f_cont
    
    def loss_pde(self):
        """Override PDE loss"""
        if self.X_pde is None:
            return torch.tensor(0.0, device=self.device)
        
        f_u, f_v, f_cont = self.pde_residual(self.X_pde)
        
        loss_momentum = torch.mean(f_u ** 2 + f_v ** 2)
        loss_continuity = torch.mean(f_cont ** 2)
        
        lambda_cont = self.config.get('lambda_cont', 1.0)
        
        return loss_momentum + lambda_cont * loss_continuity
    
    def compute_loss(self):
        """Compute total loss"""
        # Inlet BC loss
        pred_inlet = self.model(self.X_bc_inlet)
        loss_inlet = torch.mean((pred_inlet - self.U_bc_inlet)**2)
        
        # Outlet BC loss (zero gradient approximation)
        pred_outlet = self.model(self.X_bc_outlet)
        loss_outlet = torch.mean(pred_outlet[:, 0:1]**2)  # 简化为0损失，或者可以不计算
        
        # Wall BC loss
        pred_wall = self.model(self.X_bc_wall)
        loss_wall = torch.mean((pred_wall - self.U_bc_wall)**2)
        
        # Cylinder BC loss
        pred_cyl = self.model(self.X_bc_cylinder)
        loss_cyl = torch.mean((pred_cyl - self.U_bc_cylinder)**2)
        
        # PDE residual loss
        loss_pde = self.loss_pde()
        
        # Data loss
        loss_data_val = self.loss_data()
        
        # Total loss
        lambda_inlet = self.config.get('lambda_bc_inlet', 10.0)
        lambda_outlet = self.config.get('lambda_bc_outlet', 5.0)
        lambda_wall = self.config.get('lambda_bc_wall', 10.0)
        lambda_cyl = self.config.get('lambda_bc_cylinder', 20.0)
        lambda_pde = self.config.get('lambda_pde', 1.0)
        lambda_data = self.config.get('lambda_data', 1.0)
        
        total_loss = (lambda_inlet * loss_inlet +
                    lambda_outlet * loss_outlet +
                    lambda_wall * loss_wall +
                    lambda_cyl * loss_cyl +
                    lambda_pde * loss_pde +
                    lambda_data * loss_data_val)
        
        # 只返回 total 和 pde，不返回其他 boundary loss components
        loss_dict = {
            'total': total_loss.item(),
            'pde': loss_pde.item(),
            'data': loss_data_val.item()
        }
        
        return total_loss, loss_dict