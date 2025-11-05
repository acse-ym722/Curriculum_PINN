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
        
        # Initial condition (t=0)
        x_ic = torch.linspace(x_domain[0], x_domain[1], N_ic).view(-1, 1)
        t_ic = torch.zeros_like(x_ic)
        self.X_ic = torch.cat([x_ic, t_ic], dim=1).to(self.device)
        self.u_ic = -torch.sin(np.pi * x_ic).to(self.device)
        
        # Boundary conditions (x=-1 and x=1)
        t_bc = torch.linspace(t_domain[0], t_domain[1], N_bc).view(-1, 1)
        x_bc_left = torch.full_like(t_bc, x_domain[0])
        x_bc_right = torch.full_like(t_bc, x_domain[1])
        
        X_bc_left = torch.cat([x_bc_left, t_bc], dim=1)
        X_bc_right = torch.cat([x_bc_right, t_bc], dim=1)
        self.X_bc = torch.cat([X_bc_left, X_bc_right], dim=0).to(self.device)
        self.u_bc = torch.zeros(2 * N_bc, 1).to(self.device)
        
        # PDE collocation points
        t_pde = torch.rand(N_pde, 1) * (t_domain[1] - t_domain[0]) + t_domain[0]
        x_pde = torch.rand(N_pde, 1) * (x_domain[1] - x_domain[0]) + x_domain[0]
        self.X_pde = torch.cat([x_pde, t_pde], dim=1).to(self.device)
        self.X_pde.requires_grad = True
    
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
        
        N_bc = self.config.get('N_bc', 200)
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