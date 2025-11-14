"""
Finite Difference Method for 1D Wave Equation
"""
import numpy as np
import time


class WaveFDM:
    """
    Finite Difference solver for 1D Wave equation:
    u_tt = c^2 * u_xx
    
    Using explicit central difference scheme
    """
    
    def __init__(self, c, x_domain=(0.0, 1.0), t_domain=(0.0, 2.0),
                 nx=512, nt=2000, k=1):
        """
        Initialize FDM solver
        
        Args:
            c: Wave speed
            x_domain: Spatial domain [x_min, x_max]
            t_domain: Time domain [t_min, t_max]
            nx: Number of spatial grid points
            nt: Number of time steps
            k: Wave number for initial condition
        """
        self.c = c
        self.x_domain = x_domain
        self.t_domain = t_domain
        self.nx = nx
        self.nt = nt
        self.k = k
        
        # Create grid
        self.x = np.linspace(x_domain[0], x_domain[1], nx)
        self.t = np.linspace(t_domain[0], t_domain[1], nt)
        self.dx = self.x[1] - self.x[0]
        self.dt = self.t[1] - self.t[0]
        
        # Solution array
        self.u = np.zeros((nx, nt))
        
        # Check stability
        self._check_stability()
    
    def _check_stability(self):
        """Check CFL stability condition"""
        print("\n" + "-"*60)
        print("Evaluating FDM Stability for Wave Equation")
        print("-"*60)
        print(f"  Space step (dx): {self.dx:.6f}")
        print(f"  Time step (dt): {self.dt:.6f}")
        print(f"  Wave speed (c): {self.c:.2f}")
        print("-"*20)
        
        # CFL condition: c * dt / dx <= 1
        cfl = self.c * self.dt / self.dx
        
        print(f"  CFL number = c*dt/dx = {cfl:.4f} (Required: <= 1.0)")
        print("-"*20)
        
        is_stable = cfl <= 1.0
        
        if is_stable:
            print("  [SUCCESS] FDM parameters are STABLE")
        else:
            print("  [FAILURE] FDM parameters are UNSTABLE")
            print(f"  Suggested max dt: {self.dx / self.c:.6f}")
        
        print("-"*60 + "\n")
        
        return is_stable
    
    def set_initial_conditions(self):
        """
        Set initial conditions:
        u(x, 0) = sin(k*pi*x)
        u_t(x, 0) = 0
        """
        # u(x, 0)
        self.u[:, 0] = np.sin(self.k * np.pi * self.x)
        
        # For u_t(x, 0) = 0, use backward approximation
        # u(:, 1) ≈ u(:, 0) since velocity is zero
        self.u[:, 1] = self.u[:, 0]
    
    def set_boundary_conditions(self, n):
        """
        Set boundary conditions: u(0, t) = 0, u(1, t) = 0
        """
        self.u[0, n] = 0.0
        self.u[-1, n] = 0.0
    
    def solve(self):
        """Solve wave equation using explicit FDM"""
        print("Starting FDM solver for 1D Wave equation...")
        print(f"Grid: {self.nx} spatial points, {self.nt} time steps")
        print(f"Wave speed: c = {self.c}")
        print(f"Wave number: k = {self.k}")
        print("-" * 60)
        
        start_time = time.time()
        
        # Set initial conditions
        self.set_initial_conditions()
        
        # Pre-compute constant
        r = (self.c * self.dt / self.dx) ** 2
        
        # Time stepping (starting from n=1 since we have u[:, 0] and u[:, 1])
        for n in range(1, self.nt - 1):
            # Interior points using central difference
            for i in range(1, self.nx - 1):
                self.u[i, n+1] = (2 * (1 - r) * self.u[i, n] + 
                                 r * (self.u[i+1, n] + self.u[i-1, n]) - 
                                 self.u[i, n-1])
            
            # Boundary conditions
            self.set_boundary_conditions(n+1)
        
        elapsed = time.time() - start_time
        print(f"FDM solver completed in {elapsed:.2f} seconds")
        print("-" * 60 + "\n")
        
        return self.u, self.x, self.t
    
    def analytical_solution(self):
        """
        Analytical solution for validation:
        u(x, t) = sin(k*pi*x) * cos(k*pi*c*t)
        """
        X, T = np.meshgrid(self.x, self.t, indexing='ij')
        u_analytical = np.sin(self.k * np.pi * X) * np.cos(self.k * np.pi * self.c * T)
        return u_analytical
    
    def save_solution(self, filepath):
        """Save FDM solution"""
        np.savez(filepath,
                 x=self.x,
                 t=self.t,
                 u=self.u,
                 c=self.c,
                 k=self.k,
                 nx=self.nx,
                 nt=self.nt)
        print(f"✓ FDM solution saved to {filepath}")
    
    @staticmethod
    def load_solution(filepath):
        """Load saved FDM solution"""
        data = np.load(filepath)
        print(f"✓ FDM solution loaded from {filepath}")
        return {
            'x': data['x'],
            't': data['t'],
            'u': data['u'],
            'c': float(data['c']),
            'k': int(data['k']),
            'nx': int(data['nx']),
            'nt': int(data['nt'])
        }