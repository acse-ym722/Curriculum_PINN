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
        """Generate training points using config"""
        x_min, x_max = self.x_domain
        y_min, y_max = self.y_domain
        
        # 从配置读取边界条件
        bc_config = self.config.get('boundary_conditions', {})
        
        # ========== 入口边界 (Inlet) ==========
        inlet_bc = bc_config.get('inlet', {})
        N_inlet = self.config.get('N_bc_inlet', 500)
        
        y_inlet = np.random.uniform(y_min, y_max, N_inlet)
        x_inlet = np.full(N_inlet, x_min)
        self.X_bc_inlet = torch.tensor(np.stack([x_inlet, y_inlet], axis=1), 
                                    dtype=torch.float32, device=self.device)
        
        # 从配置读取边界值
        u_inlet = self.U_inf if inlet_bc.get('u') == 'U_inf' else inlet_bc.get('u', self.U_inf)
        v_inlet = inlet_bc.get('v', 0.0)
        p_inlet = inlet_bc.get('p', 0.0)
        
        self.U_bc_inlet = torch.tensor([[u_inlet, v_inlet, p_inlet]] * N_inlet,
                                    dtype=torch.float32, device=self.device)
        
        # ========== 出口边界 (Outlet) - 零梯度 ==========
        outlet_bc = bc_config.get('outlet', {})
        N_outlet = self.config.get('N_bc_outlet', 500)
        
        y_outlet = np.random.uniform(y_min, y_max, N_outlet)
        x_outlet = np.full(N_outlet, x_max)
        self.X_bc_outlet = torch.tensor(np.stack([x_outlet, y_outlet], axis=1),
                                        dtype=torch.float32, device=self.device)
        # 出口不强制值，只在loss中处理梯度
        
        # ========== 上下壁面 (Walls) - 滑移边界 ==========
        wall_bc = bc_config.get('walls', {})
        N_wall = self.config.get('N_bc_wall', 200)
        
        x_top = np.random.uniform(x_min, x_max, N_wall)
        x_bot = np.random.uniform(x_min, x_max, N_wall)
        y_top = np.full(N_wall, y_max)
        y_bot = np.full(N_wall, y_min)
        
        self.X_bc_wall = torch.tensor(
            np.concatenate([np.stack([x_top, y_top], axis=1),
                        np.stack([x_bot, y_bot], axis=1)]),
            dtype=torch.float32, device=self.device)
        
        u_wall = self.U_inf if wall_bc.get('u') == 'U_inf' else wall_bc.get('u', self.U_inf)
        v_wall = wall_bc.get('v', 0.0)
        
        self.U_bc_wall = torch.tensor([[u_wall, v_wall, 0.0]] * (2 * N_wall),
                                    dtype=torch.float32, device=self.device)
        
        # ========== 圆柱表面 (Cylinder) - 无滑移 ==========
        cyl_bc = bc_config.get('cylinder', {})
        N_cyl = self.config.get('N_bc_cylinder', 800)
        
        theta = np.linspace(0, 2*np.pi, N_cyl)
        x_cyl = self.cx + self.radius * np.cos(theta)
        y_cyl = self.cy + self.radius * np.sin(theta)
        
        self.X_bc_cylinder = torch.tensor(np.stack([x_cyl, y_cyl], axis=1),
                                        dtype=torch.float32, device=self.device)
        
        u_cyl = cyl_bc.get('u', 0.0)
        v_cyl = cyl_bc.get('v', 0.0)
        
        self.U_bc_cylinder = torch.tensor([[u_cyl, v_cyl, 0.0]] * N_cyl,
                                        dtype=torch.float32, device=self.device)
    
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
        
        # Outlet BC loss
        pred_outlet = self.model(self.X_bc_outlet)
        loss_outlet = torch.mean(pred_outlet[:, 0:1]**2)
        
        # Wall BC loss
        pred_wall = self.model(self.X_bc_wall)
        loss_wall = torch.mean((pred_wall - self.U_bc_wall)**2)
        
        # Cylinder BC loss
        pred_cyl = self.model(self.X_bc_cylinder)
        loss_cyl = torch.mean((pred_cyl - self.U_bc_cylinder)**2)
        
        # PDE residual loss
        loss_pde_val = self.loss_pde()
        
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
                    lambda_pde * loss_pde_val +
                    lambda_data * loss_data_val)
        
        # Return detailed loss dict for monitoring
        loss_dict = {
            'total': total_loss.item(),
            'bc_inlet': loss_inlet.item(),
            'bc_outlet': loss_outlet.item(),
            'bc_wall': loss_wall.item(),
            'bc_cylinder': loss_cyl.item(),
            'pde': loss_pde_val.item(),
            'data': loss_data_val.item()
        }
        
        return total_loss, loss_dict


class WavePINN(BasePINNSolver):
    """
    Standard PINN solver for 1D Wave Equation:
    u_tt = c^2 * u_xx
    
    Initial conditions:
    u(x, 0) = sin(k*pi*x)
    u_t(x, 0) = 0
    
    Boundary conditions:
    u(0, t) = 0
    u(1, t) = 0
    """
    
    def __init__(self, model, config, device='cuda'):
        self.c = config.get('c', 1.0)
        self.k = config.get('k', 1)
        super().__init__(model, config, device)
        self._generate_training_data()
    
    def _generate_training_data(self):
        """Generate training data for Wave equation"""
        x_domain = self.config.get('x_domain', [0.0, 1.0])
        t_domain = self.config.get('t_domain', [0.0, 2.0])
        
        N_ic = self.config.get('N_ic', 200)
        N_bc = self.config.get('N_bc', 200)
        N_pde = self.config.get('N_pde', 10000)
        
        # ========== Initial conditions (t=0) ==========
        # u(x, 0) = sin(k*pi*x)
        x_ic = torch.linspace(x_domain[0], x_domain[1], N_ic).view(-1, 1)
        t_ic = torch.zeros_like(x_ic)
        self.X_ic = torch.cat([x_ic, t_ic], dim=1).to(self.device)
        self.u_ic = torch.sin(self.k * np.pi * x_ic).to(self.device)
        
        # u_t(x, 0) = 0 (需要单独处理)
        self.X_ic_t = self.X_ic.clone()
        self.X_ic_t.requires_grad = True
        
        # ========== Boundary conditions ==========
        t_bc = torch.linspace(t_domain[0], t_domain[1], N_bc).view(-1, 1)
        x_bc_left = torch.full_like(t_bc, x_domain[0])
        x_bc_right = torch.full_like(t_bc, x_domain[1])
        
        X_bc_left = torch.cat([x_bc_left, t_bc], dim=1)
        X_bc_right = torch.cat([x_bc_right, t_bc], dim=1)
        self.X_bc = torch.cat([X_bc_left, X_bc_right], dim=0).to(self.device)
        self.u_bc = torch.zeros(2 * N_bc, 1).to(self.device)
        
        # ========== PDE collocation points ==========
        x_pde = torch.rand(N_pde, 1) * (x_domain[1] - x_domain[0]) + x_domain[0]
        t_pde = torch.rand(N_pde, 1) * (t_domain[1] - t_domain[0]) + t_domain[0]
        self.X_pde = torch.cat([x_pde, t_pde], dim=1).to(self.device)
        self.X_pde.requires_grad = True
        
        # ========== Data points ==========
        if self.config.get('use_data_loss', False):
            N_data = self.config.get('N_data', 500)
            fdm_solution = self.config.get('fdm_solution', None)
            
            if fdm_solution is not None:
                print(f"Loading {N_data} data points from FDM solution...")
                
                x_data = torch.rand(N_data, 1) * (x_domain[1] - x_domain[0]) + x_domain[0]
                t_data = torch.rand(N_data, 1) * (t_domain[1] - t_domain[0]) + t_domain[0]
                self.X_data = torch.cat([x_data, t_data], dim=1).to(self.device)
                
                self.u_data = self._interpolate_fdm_data(
                    x_data.cpu().numpy(),
                    t_data.cpu().numpy(),
                    fdm_solution
                )
                
                print(f"✓ Generated {N_data} data points from FDM solution")
            else:
                self.X_data = None
                self.u_data = None
        else:
            self.X_data = None
            self.u_data = None
    
    def _interpolate_fdm_data(self, x_query, t_query, fdm_solution):
        """Interpolate FDM data"""
        from scipy.interpolate import RegularGridInterpolator
        
        u_grid, x_grid, t_grid = fdm_solution
        
        interpolator = RegularGridInterpolator(
            (x_grid, t_grid),
            u_grid,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        points = np.concatenate([x_query, t_query], axis=1)
        u_interp = interpolator(points)
        
        return torch.tensor(u_interp, dtype=torch.float32).reshape(-1, 1).to(self.device)
    
    def loss_ic(self):
        """Override IC loss to include both u and u_t"""
        # u(x, 0) loss
        u_pred = self.model(self.X_ic)
        loss_u = torch.mean((u_pred - self.u_ic) ** 2)
        
        # u_t(x, 0) = 0 loss
        u_pred_t = self.model(self.X_ic_t)
        
        grad_u = torch.autograd.grad(
            outputs=u_pred_t,
            inputs=self.X_ic_t,
            grad_outputs=torch.ones_like(u_pred_t),
            create_graph=True,
            retain_graph=True
        )[0]
        
        u_t = grad_u[:, 1:2]  # du/dt
        loss_u_t = torch.mean(u_t ** 2)
        
        lambda_ic_t = self.config.get('lambda_ic_t', 1.0)
        
        return loss_u + lambda_ic_t * loss_u_t
    
    def pde_residual(self, X):
        """
        Compute PDE residual: f = u_tt - c^2 * u_xx
        """
        if not X.requires_grad:
            X = X.clone().detach().requires_grad_(True)
        
        # Forward pass
        u = self.model(X)
        
        # First derivatives
        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=X,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        u_x = grad_u[:, 0:1]
        u_t = grad_u[:, 1:2]
        
        # Second derivatives
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=X,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]
        
        u_tt = torch.autograd.grad(
            outputs=u_t,
            inputs=X,
            grad_outputs=torch.ones_like(u_t),
            create_graph=True,
            retain_graph=True
        )[0][:, 1:2]
        
        # PDE residual: u_tt - c^2 * u_xx = 0
        residual = u_tt - (self.c ** 2) * u_xx
        
        return residual
    

class WavePINN2D(BasePINNSolver):
    """
    Standard PINN solver for 2D Wave Equation:
    u_tt = c^2 * (u_xx + u_yy)
    
    Initial conditions:
    u(x, y, 0) = sin(kx*pi*x) * sin(ky*pi*y)
    u_t(x, y, 0) = 0
    
    Boundary conditions:
    u = 0 on all boundaries.
    """
    
    def __init__(self, model, config, device='cuda'):
        self.c = config.get('c', 1.0)
        self.kx = config.get('kx', 1)
        self.ky = config.get('ky', 1)
        # 从config中获取总epochs，以供标准PINN使用
        # 对于课程学习，总epochs将在其类中计算
        if 'curriculum_stages' in config:
            self.epochs = sum(stage['epochs'] for stage in config['curriculum_stages'])
        else:
            self.epochs = config.get('epochs', 20000)
            
        super().__init__(model, config, device)
        self._generate_training_data()
    
    def _generate_training_data(self):
        """Generate training data for 2D Wave equation"""
        x_domain = self.config.get('x_domain', [0.0, 1.0])
        y_domain = self.config.get('y_domain', [0.0, 1.0])
        t_domain = self.config.get('t_domain', [0.0, 1.0])
        
        N_ic = self.config.get('N_ic', 500)
        N_bc = self.config.get('N_bc', 1000) # Total for all boundaries
        N_pde = self.config.get('N_pde', 20000)
        
        # ========== Initial conditions (t=0) ==========
        # u(x, y, 0) = sin(kx*pi*x) * sin(ky*pi*y)
        ic_pts = torch.rand(N_ic, 2)
        x_ic = ic_pts[:, 0:1] * (x_domain[1] - x_domain[0]) + x_domain[0]
        y_ic = ic_pts[:, 1:2] * (y_domain[1] - y_domain[0]) + y_domain[0]
        t_ic = torch.zeros_like(x_ic)
        
        self.X_ic = torch.cat([x_ic, y_ic, t_ic], dim=1).to(self.device)
        self.u_ic = (torch.sin(self.kx * np.pi * x_ic) * 
                     torch.sin(self.ky * np.pi * y_ic)).to(self.device)
        
        # For u_t(x, y, 0) = 0
        self.X_ic_t = self.X_ic.clone().detach().requires_grad_(True)
        
        # ========== Boundary conditions (u=0 on all boundaries) ==========
        N_bc_each = N_bc // 4
        
        # Time points for BC
        t_bc = torch.rand(N_bc, 1) * (t_domain[1] - t_domain[0]) + t_domain[0]
        
        # Left/Right boundaries (x=x_min, x=x_max)
        x_lr = torch.cat([torch.full((N_bc_each, 1), x_domain[0]), 
                          torch.full((N_bc_each, 1), x_domain[1])])
        y_lr = torch.rand(2 * N_bc_each, 1) * (y_domain[1] - y_domain[0]) + y_domain[0]
        X_bc_lr = torch.cat([x_lr, y_lr, t_bc[:2*N_bc_each]], dim=1)

        # Bottom/Top boundaries (y=y_min, y=y_max)
        x_bt = torch.rand(2 * N_bc_each, 1) * (x_domain[1] - x_domain[0]) + x_domain[0]
        y_bt = torch.cat([torch.full((N_bc_each, 1), y_domain[0]), 
                          torch.full((N_bc_each, 1), y_domain[1])])
        X_bc_bt = torch.cat([x_bt, y_bt, t_bc[2*N_bc_each:]], dim=1)

        self.X_bc = torch.cat([X_bc_lr, X_bc_bt], dim=0).to(self.device)
        self.u_bc = torch.zeros(self.X_bc.shape[0], 1).to(self.device)
        
        # ========== PDE collocation points ==========
        pde_pts = torch.rand(N_pde, 3)
        x_pde = pde_pts[:, 0:1] * (x_domain[1] - x_domain[0]) + x_domain[0]
        y_pde = pde_pts[:, 1:2] * (y_domain[1] - y_domain[0]) + y_domain[0]
        t_pde = pde_pts[:, 2:3] * (t_domain[1] - t_domain[0]) + t_domain[0]
        self.X_pde = torch.cat([x_pde, y_pde, t_pde], dim=1).to(self.device)
        self.X_pde.requires_grad = True

    def loss_ic(self):
        """Override IC loss to include both u and u_t for 2D."""
        # u(x, y, 0) loss
        u_pred = self.model(self.X_ic)
        loss_u = torch.mean((u_pred - self.u_ic) ** 2)
        
        # u_t(x, y, 0) = 0 loss
        u_pred_t = self.model(self.X_ic_t)
        
        grad_u = torch.autograd.grad(
            outputs=u_pred_t,
            inputs=self.X_ic_t,
            grad_outputs=torch.ones_like(u_pred_t),
            create_graph=True,
            retain_graph=True
        )[0]
        
        u_t = grad_u[:, 2:3]  # du/dt, the third component
        loss_u_t = torch.mean(u_t ** 2)
        
        lambda_ic_t = self.config.get('lambda_ic_t', 1.0)
        
        # Combine losses and return
        self.loss_components_history['ic_u'] = loss_u.item()
        self.loss_components_history['ic_t'] = loss_u_t.item()
        
        return loss_u + lambda_ic_t * loss_u_t

    def pde_residual(self, X):
        """
        Compute PDE residual: f = u_tt - c^2 * (u_xx + u_yy)
        """
        if not X.requires_grad:
            X = X.clone().detach().requires_grad_(True)
        
        u = self.model(X)
        
        # First derivatives
        grad_u = torch.autograd.grad(u, X, torch.ones_like(u), create_graph=True)[0]
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        u_t = grad_u[:, 2:3]
        
        # Second derivatives
        u_xx = torch.autograd.grad(u_x, X, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, X, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
        u_tt = torch.autograd.grad(u_t, X, torch.ones_like(u_t), create_graph=True)[0][:, 2:3]
        
        # PDE residual
        residual = u_tt - (self.c ** 2) * (u_xx + u_yy)
        
        return residual


class DarcyPINN(BasePINNSolver):
    """
    Standard PINN solver for 2D Darcy Flow:
    -∇·(K∇p) = f
    
    Expanded: -K * (∂²p/∂x² + ∂²p/∂y²) = f
    """
    
    def __init__(self, model, config, device='cuda'):
        self.K = config.get('K', 1.0)
        self.f_source = config.get('f_source_torch', None)  # PyTorch函数
        super().__init__(model, config, device)
        self._generate_training_data()
    
    def _generate_training_data(self):
        """Generate training data for Darcy equation"""
        x_domain = self.config.get('x_domain', [0.0, 1.0])
        y_domain = self.config.get('y_domain', [0.0, 0.5])
        
        N_col = self.config.get('N_collocation', 2000)
        N_bc = self.config.get('N_bc', 400)
        
        # ========== Interior collocation points ==========
        x_col = torch.rand(N_col, 1) * (x_domain[1] - x_domain[0]) + x_domain[0]
        y_col = torch.rand(N_col, 1) * (y_domain[1] - y_domain[0]) + y_domain[0]
        self.X_pde = torch.cat([x_col, y_col], dim=1).to(self.device)
        self.X_pde.requires_grad = True
        
        # ========== Boundary conditions (全Dirichlet) ==========
        # Left boundary (x=0)
        x_bc_left = torch.zeros(N_bc, 1)
        y_bc_left = torch.rand(N_bc, 1) * (y_domain[1] - y_domain[0]) + y_domain[0]
        X_bc_left = torch.cat([x_bc_left, y_bc_left], dim=1)
        p_bc_left = torch.full((N_bc, 1), self.config.get('bc_left', 0.0))
        
        # Right boundary (x=1)
        x_bc_right = torch.ones(N_bc, 1) * x_domain[1]
        y_bc_right = torch.rand(N_bc, 1) * (y_domain[1] - y_domain[0]) + y_domain[0]
        X_bc_right = torch.cat([x_bc_right, y_bc_right], dim=1)
        p_bc_right = torch.full((N_bc, 1), self.config.get('bc_right', 0.0))
        
        # Bottom boundary (y=0)
        x_bc_bottom = torch.rand(N_bc, 1) * (x_domain[1] - x_domain[0]) + x_domain[0]
        y_bc_bottom = torch.zeros(N_bc, 1)
        X_bc_bottom = torch.cat([x_bc_bottom, y_bc_bottom], dim=1)
        p_bc_bottom = torch.full((N_bc, 1), self.config.get('bc_bottom', 0.0))
        
        # Top boundary (y=1)
        x_bc_top = torch.rand(N_bc, 1) * (x_domain[1] - x_domain[0]) + x_domain[0]
        y_bc_top = torch.ones(N_bc, 1) * y_domain[1]
        X_bc_top = torch.cat([x_bc_top, y_bc_top], dim=1)
        p_bc_top = torch.full((N_bc, 1), self.config.get('bc_top', 0.0))
        
        # Combine all boundaries
        self.X_bc = torch.cat([X_bc_left, X_bc_right, X_bc_bottom, X_bc_top], dim=0).to(self.device)
        self.u_bc = torch.cat([p_bc_left, p_bc_right, p_bc_bottom, p_bc_top], dim=0).to(self.device)
        
        # ========== Data points (from FDM solution) ==========
        if self.config.get('use_data_loss', False):
            N_data = self.config.get('N_data', 500)
            fdm_solution = self.config.get('fdm_solution', None)
            
            if fdm_solution is not None:
                print(f"  Loading {N_data} data points from FDM solution...")
                
                # Randomly sample spatial points
                x_data = torch.rand(N_data, 1) * (x_domain[1] - x_domain[0]) + x_domain[0]
                y_data = torch.rand(N_data, 1) * (y_domain[1] - y_domain[0]) + y_domain[0]
                self.X_data = torch.cat([x_data, y_data], dim=1).to(self.device)
                
                # Interpolate FDM solution
                self.u_data = self._interpolate_fdm_data(
                    x_data.cpu().numpy(),
                    y_data.cpu().numpy(),
                    fdm_solution
                )
                
                print(f"  ✓ Generated {N_data} data points from FDM solution")
            else:
                print("  ⚠ Warning: use_data_loss=True but no fdm_solution provided")
                self.X_data = None
                self.u_data = None
        else:
            self.X_data = None
            self.u_data = None
    
    def _interpolate_fdm_data(self, x_query, y_query, fdm_solution):
        """Interpolate FDM data at query points"""
        from scipy.interpolate import RegularGridInterpolator
        
        p_grid, x_grid, y_grid = fdm_solution
        
        # Create interpolator (p_grid is [nx, ny], corresponding to (x, y))
        interpolator = RegularGridInterpolator(
            (x_grid, y_grid),
            p_grid,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        points = np.concatenate([x_query, y_query], axis=1)
        p_interp = interpolator(points)
        
        return torch.tensor(p_interp, dtype=torch.float32).reshape(-1, 1).to(self.device)
    
    def pde_residual(self, X):
        """
        Compute PDE residual: f = -K * (∂²p/∂x² + ∂²p/∂y²) - f_source
        """
        if not X.requires_grad:
            X = X.clone().detach().requires_grad_(True)
        
        p = self.model(X)
        
        # First derivatives
        grad_p = torch.autograd.grad(
            p, X, torch.ones_like(p),
            create_graph=True, retain_graph=True
        )[0]
        
        p_x = grad_p[:, 0:1]
        p_y = grad_p[:, 1:2]
        
        # Second derivatives
        p_xx = torch.autograd.grad(
            p_x, X, torch.ones_like(p_x),
            create_graph=True, retain_graph=True
        )[0][:, 0:1]
        
        p_yy = torch.autograd.grad(
            p_y, X, torch.ones_like(p_y),
            create_graph=True, retain_graph=True
        )[0][:, 1:2]
        
        # Laplacian
        laplacian = p_xx + p_yy
        
        # Source term
        if callable(self.f_source):
            f = self.f_source(X[:, 0:1], X[:, 1:2])
        else:
            f = torch.zeros_like(p)
        
        # PDE residual
        residual = -self.K * laplacian - f
        
        return residual
    
    def loss_bc(self):
        """Compute boundary condition loss (all Dirichlet)"""
        if self.X_bc is None:
            return torch.tensor(0.0, device=self.device)
        
        p_pred = self.model(self.X_bc)
        return torch.mean((p_pred - self.u_bc) ** 2)
    
    def compute_loss(self):
        """Compute total loss"""
        lambda_pde = self.config.get('lambda_pde', 1.0)
        lambda_bc = self.config.get('lambda_bc_dirichlet', 10.0)
        lambda_data = self.config.get('lambda_data', 100.0)
        
        # PDE loss
        loss_pde_val = self.loss_pde()
        
        # BC loss
        loss_bc_val = self.loss_bc()
        
        # Data loss
        loss_data_val = self.loss_data()
        
        # Total loss
        total_loss = (lambda_pde * loss_pde_val +
                     lambda_bc * loss_bc_val +
                     lambda_data * loss_data_val)
        
        loss_dict = {
            'total': total_loss.item(),
            'pde': loss_pde_val.item(),
            'bc': loss_bc_val.item(),
            'data': loss_data_val.item()
        }
        
        return total_loss, loss_dict


