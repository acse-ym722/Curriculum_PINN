"""
Finite Difference Method for 1D Burgers Equation
"""
import numpy as np
import time


class BurgersFDM:
    """
    Finite Difference solver for 1D Burgers equation:
    u_t + u * u_x = nu * u_xx
    
    Using explicit time integration with upwind scheme
    """
    
    def __init__(self, nu, x_domain=(-1.0, 1.0), t_domain=(0.0, 1.0), 
                 nx=512, nt=10000):
        """
        Initialize FDM solver
        
        Args:
            nu: Viscosity coefficient
            x_domain: Spatial domain [x_min, x_max]
            t_domain: Time domain [t_min, t_max]
            nx: Number of spatial grid points
            nt: Number of time steps
        """
        self.nu = nu
        self.x_domain = x_domain
        self.t_domain = t_domain
        self.nx = nx
        self.nt = nt
        
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
        """Check CFL stability conditions"""
        print("\n" + "-"*60)
        print("Evaluating FDM Stability Conditions (CFL)")
        print("-"*60)
        print(f"  Space step (dx): {self.dx:.6f}")
        print(f"  Time step (dt): {self.dt:.6f}")
        print("-"*20)
        
        u_max_est = 1.0
        cfl_adv = u_max_est * self.dt / self.dx
        cfl_diff = 2 * self.nu * self.dt / (self.dx**2)
        
        print(f"  Advective CFL = |u|*dt/dx = {cfl_adv:.4f} (Req: <= 1.0)")
        print(f"  Diffusive CFL = 2*nu*dt/dx^2 = {cfl_diff:.4f} (Req: <= 0.5)")
        print("-"*20)
        
        is_stable = True
        if cfl_adv > 1.0:
            print("  [WARNING] Advective CFL condition not met!")
            is_stable = False
        if cfl_diff > 0.5:
            print("  [WARNING] Diffusive CFL condition not met!")
            is_stable = False
        
        if is_stable:
            print("  [SUCCESS] FDM parameters are STABLE")
        else:
            print("  [FAILURE] FDM parameters are UNSTABLE")
        
        print("-"*60 + "\n")
        
        return is_stable
    
    def set_initial_condition(self):
        """Set initial condition: u(x, 0) = -sin(pi*x)"""
        self.u[:, 0] = -np.sin(np.pi * self.x)
    
    def set_boundary_conditions(self, n):
        """Set boundary conditions: u(-1, t) = 0, u(1, t) = 0"""
        self.u[0, n] = 0.0
        self.u[-1, n] = 0.0
    
    def solve(self):
        """Solve Burgers equation using explicit FDM"""
        print("Starting FDM solver for Burgers equation...")
        print(f"Grid: {self.nx} spatial points, {self.nt} time steps")
        print(f"Viscosity: nu = {self.nu}")
        print("-" * 60)
        
        start_time = time.time()
        
        # Set initial condition
        self.set_initial_condition()
        
        # Time stepping
        for n in range(self.nt - 1):
            un = self.u[:, n].copy()
            
            # Interior points
            for i in range(1, self.nx - 1):
                # Central difference for diffusion
                u_x = (un[i+1] - un[i-1]) / (2 * self.dx)
                u_xx = (un[i+1] - 2*un[i] + un[i-1]) / (self.dx**2)
                
                # Time step
                self.u[i, n+1] = un[i] - self.dt * (un[i] * u_x - self.nu * u_xx)
            
            # Boundary conditions
            self.set_boundary_conditions(n+1)
        
        elapsed = time.time() - start_time
        print(f"FDM solver completed in {elapsed:.2f} seconds")
        print("-" * 60 + "\n")
        
        return self.u, self.x, self.t
    
    def save_solution(self, filepath):
        """
        Save FDM solution for use as training data
        
        Args:
            filepath: Path to save the data (e.g., 'outputs/burgers/fdm_data.npz')
        """
        np.savez(filepath,
                 x=self.x,
                 t=self.t,
                 u=self.u,
                 nu=self.nu,
                 nx=self.nx,
                 nt=self.nt)
        print(f"✓ FDM solution saved to {filepath}")
    
    @staticmethod
    def load_solution(filepath):
        """
        Load saved FDM solution
        
        Args:
            filepath: Path to the saved data
        
        Returns:
            Dictionary containing x, t, u, and metadata
        """
        data = np.load(filepath)
        print(f"✓ FDM solution loaded from {filepath}")
        return {
            'x': data['x'],
            't': data['t'],
            'u': data['u'],
            'nu': float(data['nu']),
            'nx': int(data['nx']),
            'nt': int(data['nt'])
        }