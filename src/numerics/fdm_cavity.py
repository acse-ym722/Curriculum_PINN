"""
Finite Difference Method for 2D Lid-Driven Cavity Flow
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import time


class CavityFDM:
    """
    Finite Difference solver for 2D lid-driven cavity flow
    using vorticity-stream function formulation
    """
    
    def __init__(self, Re, nx=64, ny=64, dt=0.0001, 
                 max_iter=10000, tol=1e-6):
        """
        Initialize FDM solver
        
        Args:
            Re: Reynolds number
            nx: Number of grid points in x
            ny: Number of grid points in y
            dt: Time step
            max_iter: Maximum iterations
            tol: Convergence tolerance
        """
        self.Re = Re
        self.nu = 1.0 / Re
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.max_iter = max_iter
        self.tol = tol
        
        # Domain
        self.Lx = 1.0
        self.Ly = 1.0
        self.dx = self.Lx / (nx - 1)
        self.dy = self.Ly / (ny - 1)
        
        # Grid
        self.x = np.linspace(0, self.Lx, nx)
        self.y = np.linspace(0, self.Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Solution arrays
        self.psi = np.zeros((ny, nx))  # Stream function
        self.omega = np.zeros((ny, nx))  # Vorticity
        self.u = np.zeros((ny, nx))  # x-velocity
        self.v = np.zeros((ny, nx))  # y-velocity
        self.p = np.zeros((ny, nx))  # Pressure
        
        self._check_stability()
    
    def _check_stability(self):
        """Check CFL stability conditions"""
        print("\n" + "-"*60)
        print("Evaluating FDM Stability Conditions (CFL)")
        print("-"*60)
        print(f"  Reynolds number: {self.Re}")
        print(f"  Space step (dx): {self.dx:.6f}")
        print(f"  Time step (dt): {self.dt:.6f}")
        print("-"*20)
        
        u_max = 1.0
        cfl_adv = u_max * self.dt / self.dx
        cfl_diff = 2 * self.nu * self.dt / (self.dx**2)
        
        print(f"  Advective CFL = {cfl_adv:.4f} (Req: <= 1.0)")
        print(f"  Diffusive CFL = {cfl_diff:.4f} (Req: <= 0.5)")
        print("-"*20)
        
        is_stable = cfl_adv <= 1.0 and cfl_diff <= 0.5
        
        if is_stable:
            print("  [SUCCESS] FDM parameters are STABLE")
        else:
            print("  [WARNING] FDM parameters may be UNSTABLE")
        
        print("-"*60 + "\n")
    
    def solve_poisson_psi(self):
        """Solve Poisson equation: ∇²ψ = -ω"""
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
    
    def compute_velocity(self):
        """Compute velocity from stream function"""
        # Interior points
        self.u[1:-1, 1:-1] = (self.psi[2:, 1:-1] - self.psi[:-2, 1:-1]) / (2 * self.dy)
        self.v[1:-1, 1:-1] = -(self.psi[1:-1, 2:] - self.psi[1:-1, :-2]) / (2 * self.dx)
        
        # Boundary conditions
        self.u[:, 0] = 0.0
        self.u[:, -1] = 0.0
        self.u[0, :] = 0.0
        self.u[-1, :] = 1.0  # Lid velocity
        
        self.v[:, 0] = 0.0
        self.v[:, -1] = 0.0
        self.v[0, :] = 0.0
        self.v[-1, :] = 0.0
    
    def update_omega_boundary(self):
        """Update boundary vorticity"""
        dx2, dy2 = self.dx**2, self.dy**2
        
        self.omega[:, 0] = -2.0 * self.psi[:, 1] / dx2
        self.omega[:, -1] = -2.0 * self.psi[:, -2] / dx2
        self.omega[0, :] = -2.0 * self.psi[1, :] / dy2
        self.omega[-1, :] = -2.0 * self.psi[-2, :] / dy2 - 2.0 / self.dy
    
    def update_omega(self):
        """Update vorticity using explicit scheme"""
        dx, dy, dt = self.dx, self.dy, self.dt
        
        du_omega = np.zeros_like(self.omega)
        dv_omega = np.zeros_like(self.omega)
        
        # Upwind scheme for advection
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
        
        # Diffusion
        d2omega = (
            (self.omega[1:-1, 2:] - 2*self.omega[1:-1, 1:-1] + self.omega[1:-1, :-2]) / dx**2 +
            (self.omega[2:, 1:-1] - 2*self.omega[1:-1, 1:-1] + self.omega[:-2, 1:-1]) / dy**2
        )
        
        # Update
        self.omega[1:-1, 1:-1] += dt * (
            -du_omega[1:-1, 1:-1] - dv_omega[1:-1, 1:-1] + self.nu * d2omega
        )
    
    def compute_pressure(self):
        """Compute pressure (simplified)"""
        self.p = -self.psi
    
    def solve(self):
        """Main solver loop"""
        print(f"\nStarting FDM solver for lid-driven cavity flow")
        print(f"Reynolds number: {self.Re}")
        print(f"Grid: {self.nx} x {self.ny}")
        print("-" * 60)
        
        start_time = time.time()
        
        for iteration in range(self.max_iter):
            omega_old = self.omega.copy()
            
            self.solve_poisson_psi()
            self.compute_velocity()
            self.update_omega()
            self.update_omega_boundary()
            
            residual = np.max(np.abs(self.omega - omega_old))
            
            if (iteration + 1) % 500 == 0:
                elapsed = time.time() - start_time
                print(f"Iter {iteration+1:5d}: Residual = {residual:.2e}, "
                      f"Time = {elapsed:.1f}s")
            
            if residual < self.tol:
                elapsed = time.time() - start_time
                print("-" * 60)
                print(f"✓ FDM converged at iteration {iteration+1}")
                print(f"  Final residual: {residual:.2e}")
                print(f"  Total time: {elapsed:.1f}s")
                print("-" * 60 + "\n")
                break
        else:
            print(f"\n⚠ Warning: Max iterations ({self.max_iter}) reached\n")
        
        self.compute_pressure()
        
        return self.u, self.v, self.p