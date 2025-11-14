# 文件名: src/numerics/fdm_wave_2d.py

"""
Finite Difference Method for 2D Wave Equation
"""
import numpy as np
import time

class WaveFDM2D:
    """
    Finite Difference solver for 2D Wave equation:
    u_tt = c^2 * (u_xx + u_yy)
    
    Using explicit central difference scheme.
    """
    
    def __init__(self, c, x_domain=(0.0, 1.0), y_domain=(0.0, 1.0), t_domain=(0.0, 2.0),
                 nx=128, ny=128, nt=2000, kx=1, ky=1):
        """
        Initialize FDM solver for 2D wave equation.
        
        Args:
            c: Wave speed
            x_domain: Spatial domain in x [x_min, x_max]
            y_domain: Spatial domain in y [y_min, y_max]
            t_domain: Time domain [t_min, t_max]
            nx: Number of spatial grid points in x
            ny: Number of spatial grid points in y
            nt: Number of time steps
            kx: Wave number for initial condition in x
            ky: Wave number for initial condition in y
        """
        self.c = c
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.t_domain = t_domain
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.kx = kx
        self.ky = ky
        
        # Create grid
        self.x = np.linspace(x_domain[0], x_domain[1], nx)
        self.y = np.linspace(y_domain[0], y_domain[1], ny)
        self.t = np.linspace(t_domain[0], t_domain[1], nt)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dt = self.t[1] - self.t[0]
        
        # Solution array
        self.u = np.zeros((nx, ny, nt))
        
        # Check stability
        self._check_stability()
    
    def _check_stability(self):
        """Check CFL stability condition for 2D."""
        print("\n" + "-"*60)
        print("Evaluating FDM Stability for 2D Wave Equation")
        print("-"*60)
        print(f"  Space steps (dx, dy): {self.dx:.6f}, {self.dy:.6f}")
        print(f"  Time step (dt): {self.dt:.6f}")
        print(f"  Wave speed (c): {self.c:.2f}")
        print("-"*20)
        
        # 2D CFL condition: c * dt * sqrt(1/dx^2 + 1/dy^2) <= 1
        cfl = self.c * self.dt * np.sqrt(1/self.dx**2 + 1/self.dy**2)
        
        print(f"  CFL number = c*dt*sqrt(1/dx^2 + 1/dy^2) = {cfl:.4f} (Required: <= 1.0)")
        print("-"*20)
        
        is_stable = cfl <= 1.0
        
        if is_stable:
            print("  [SUCCESS] FDM parameters are STABLE")
        else:
            print("  [FAILURE] FDM parameters are UNSTABLE")
            max_dt = 1.0 / (self.c * np.sqrt(1/self.dx**2 + 1/self.dy**2))
            print(f"  Suggested max dt: {max_dt:.6f}")
        
        print("-"*60 + "\n")
        
        return is_stable
    
    def set_initial_conditions(self):
        """
        Set initial conditions:
        u(x, y, 0) = sin(kx*pi*x) * sin(ky*pi*y)
        u_t(x, y, 0) = 0
        """
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # u(x, y, 0)
        self.u[:, :, 0] = np.sin(self.kx * np.pi * X) * np.sin(self.ky * np.pi * Y)
        
        # For u_t(x, y, 0) = 0, use backward approximation
        # u(..., 1) ≈ u(..., 0) since velocity is zero
        self.u[:, :, 1] = self.u[:, :, 0]
    
    def set_boundary_conditions(self, n):
        """
        Set boundary conditions: u=0 on all boundaries (Dirichlet).
        """
        self.u[0, :, n] = 0.0
        self.u[-1, :, n] = 0.0
        self.u[:, 0, n] = 0.0
        self.u[:, -1, n] = 0.0
    
    def solve(self):
        """Solve 2D wave equation using explicit FDM"""
        print("Starting FDM solver for 2D Wave equation...")
        print(f"Grid: {self.nx}x{self.ny} spatial points, {self.nt} time steps")
        print(f"Wave speed: c = {self.c}")
        print(f"Wave numbers: kx = {self.kx}, ky = {self.ky}")
        print("-" * 60)
        
        start_time = time.time()
        
        self.set_initial_conditions()
        
        # Pre-compute constants
        cx2 = (self.c * self.dt / self.dx) ** 2
        cy2 = (self.c * self.dt / self.dy) ** 2
        
        # Time stepping (from n=1 to nt-2)
        for n in range(1, self.nt - 1):
            # Interior points
            for i in range(1, self.nx - 1):
                for j in range(1, self.ny - 1):
                    u_xx = self.u[i+1, j, n] - 2*self.u[i, j, n] + self.u[i-1, j, n]
                    u_yy = self.u[i, j+1, n] - 2*self.u[i, j, n] + self.u[i, j-1, n]
                    
                    self.u[i, j, n+1] = (2*self.u[i, j, n] - self.u[i, j, n-1] + 
                                         cx2 * u_xx + cy2 * u_yy)
            
            # Boundary conditions
            self.set_boundary_conditions(n+1)
        
        elapsed = time.time() - start_time
        print(f"FDM solver completed in {elapsed:.2f} seconds")
        print("-" * 60 + "\n")
        
        return self.u, self.x, self.y, self.t

    def analytical_solution(self):
        """
        Analytical solution for validation.
        u(x,y,t) = sin(kx*pi*x)sin(ky*pi*y)cos(c*pi*sqrt(kx^2+ky^2)*t)
        """
        X, Y, T = np.meshgrid(self.x, self.y, self.t, indexing='ij')
        omega = self.c * np.pi * np.sqrt(self.kx**2 + self.ky**2)
        u_analytical = (np.sin(self.kx * np.pi * X) * 
                        np.sin(self.ky * np.pi * Y) * 
                        np.cos(omega * T))
        return u_analytical