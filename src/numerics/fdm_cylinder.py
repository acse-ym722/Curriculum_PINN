"""
Finite Difference Method for 2D Flow Past a Cylinder
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import time
import os
import pickle

class CylinderFDM:
    """
    Finite Difference solver for 2D flow past a cylinder
    using vorticity-stream function formulation with immersed boundary method
    """
    
    def __init__(self, Re, U_inf=1.0, cylinder_center=(0.0, 0.0), 
                 cylinder_radius=0.5, nx=200, ny=100, 
                 x_domain=(-5.0, 15.0), y_domain=(-5.0, 5.0),
                 dt=0.001, max_iter=20000, tol=1e-6, config=None):
        """Initialize FDM solver for cylinder flow"""
        self.Re = Re
        self.nu = U_inf * 2 * cylinder_radius / Re  # ν = U∞ * D / Re
        self.U_inf = U_inf
        self.cx, self.cy = cylinder_center
        self.radius = cylinder_radius
        self.diameter = 2 * cylinder_radius
        
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.max_iter = max_iter
        self.tol = tol
        
        # Domain
        self.x_min, self.x_max = x_domain
        self.y_min, self.y_max = y_domain
        self.Lx = self.x_max - self.x_min
        self.Ly = self.y_max - self.y_min
        self.dx = self.Lx / (nx - 1)
        self.dy = self.Ly / (ny - 1)
        
        # Calculate total simulation time
        self.total_time = dt * max_iter
        
        # Grid
        self.x = np.linspace(self.x_min, self.x_max, nx)
        self.y = np.linspace(self.y_min, self.y_max, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Solution arrays
        self.psi = np.zeros((ny, nx))
        self.omega = np.zeros((ny, nx))
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))
        
        # Convergence tracking
        self.convergence_history = []
        self.actual_iterations = 0
        self.final_time = 0.0

        self.config = config

        # Identify cylinder points
        self._identify_cylinder_region()
        self._initialize_flow()
        self._check_stability()
        self._print_initialization()
    
    def _print_initialization(self):
        """Print initialization information"""
        print("\n" + "="*70)
        print("FDM SOLVER INITIALIZATION - Flow Past Cylinder")
        print("="*70)
        print(f"Flow Parameters:")
        print(f"  Reynolds number (Re):        {self.Re}")
        print(f"  Free stream velocity (U∞):   {self.U_inf} m/s")
        print(f"  Cylinder diameter (D):        {self.diameter} m")
        print(f"  Kinematic viscosity (ν):      {self.nu:.6e} m²/s")
        print(f"\nDomain:")
        print(f"  x: [{self.x_min}, {self.x_max}] ({self.Lx:.1f} m)")
        print(f"  y: [{self.y_min}, {self.y_max}] ({self.Ly:.1f} m)")
        print(f"  Cylinder center: ({self.cx}, {self.cy})")
        print(f"  Cylinder radius: {self.radius} m")
        print(f"\nDiscretization:")
        print(f"  Grid size: {self.nx} × {self.ny} = {self.nx * self.ny:,} points")
        print(f"  dx = {self.dx:.6f} m")
        print(f"  dy = {self.dy:.6f} m")
        print(f"\nTime Integration:")
        print(f"  Time step (dt):              {self.dt:.6f} s")
        print(f"  Max iterations:              {self.max_iter:,}")
        print(f"  Maximum simulation time:     {self.total_time:.4f} s")
        print(f"  Convergence tolerance:       {self.tol:.2e}")
        print(f"\nCharacteristic Time Scales:")
        T_conv = self.diameter / self.U_inf
        T_diff = self.diameter**2 / self.nu
        print(f"  Convective time (D/U∞):      {T_conv:.4f} s")
        print(f"  Diffusive time (D²/ν):       {T_diff:.4f} s")
        if self.Re > 47:  # Approximate critical Re for vortex shedding
            St = 0.198 * (1 - 19.7/self.Re) if self.Re < 200 else 0.2
            T_shed = self.diameter / (St * self.U_inf)
            print(f"  Vortex shedding period:      ~{T_shed:.4f} s (St≈{St:.3f})")
        print("="*70 + "\n")
    
    def _identify_cylinder_region(self):
        """Identify grid points inside cylinder"""
        dist = np.sqrt((self.X - self.cx)**2 + (self.Y - self.cy)**2)
        self.is_cylinder = dist <= self.radius
        self.is_near_cylinder = (dist > self.radius) & (dist <= self.radius + 2*max(self.dx, self.dy))
        
        n_cylinder = np.sum(self.is_cylinder)
        n_fluid = self.nx * self.ny - n_cylinder
        print(f"Grid Statistics:")
        print(f"  Cylinder points: {n_cylinder} ({100*n_cylinder/(self.nx*self.ny):.1f}%)")
        print(f"  Fluid points:    {n_fluid} ({100*n_fluid/(self.nx*self.ny):.1f}%)")
    
    def _initialize_flow(self):
        """Initialize flow field using config"""
        if self.config is None:
            # 默认行为：均匀流
            self.u[:, :] = self.U_inf
            self.v[:, :] = 0.0
            for i in range(self.ny):
                self.psi[i, :] = self.U_inf * (self.y[i] - self.y_min)
        else:
            # 从配置读取初始条件类型
            ic_type = self.config.get('initial_condition_type', 'uniform')
            ic_config = self.config['initial_conditions'][ic_type]
            
            if ic_type == 'uniform':
                self.u[:, :] = self.U_inf
                self.v[:, :] = 0.0
                for i in range(self.ny):
                    self.psi[i, :] = self.U_inf * (self.y[i] - self.y_min)
            
            elif ic_type == 'potential_flow':
                # 使用圆柱势流解
                print("Initializing with potential flow solution...")
                for i in range(self.ny):
                    for j in range(self.nx):
                        x = self.x[j] - self.cx
                        y = self.y[i] - self.cy
                        r = np.sqrt(x**2 + y**2)
                        
                        if r > self.radius:
                            # 势流解
                            cos_theta = x / r if r > 0 else 0
                            sin_theta = y / r if r > 0 else 0
                            
                            # u = U∞(1 - R²/r² * (1 - 2sin²θ))
                            self.u[i, j] = self.U_inf * (1 - self.radius**2 / r**2 * (1 - 2 * sin_theta**2))
                            # v = -U∞ * R²/r² * 2cosθ sinθ
                            self.v[i, j] = -self.U_inf * self.radius**2 / r**2 * 2 * cos_theta * sin_theta
                            # ψ = U∞(r - R²/r) sinθ
                            self.psi[i, j] = self.U_inf * (r - self.radius**2 / r) * sin_theta
                        else:
                            # 圆柱内部
                            self.u[i, j] = 0.0
                            self.v[i, j] = 0.0
                            self.psi[i, j] = 0.0
            
            elif ic_type == 'zero':
                self.u[:, :] = 0.0
                self.v[:, :] = 0.0
                self.psi[:, :] = 0.0
        
        # 强制圆柱内部为零
        self.u[self.is_cylinder] = 0.0
        self.v[self.is_cylinder] = 0.0
        self.omega[self.is_cylinder] = 0.0

    def _check_stability(self):
        """Check CFL stability conditions"""
        print("\n" + "-"*70)
        print("STABILITY ANALYSIS")
        print("-"*70)
        
        cfl_adv_x = self.U_inf * self.dt / self.dx
        cfl_adv_y = self.U_inf * self.dt / self.dy
        cfl_adv = max(cfl_adv_x, cfl_adv_y)
        
        cfl_diff_x = self.nu * self.dt / (self.dx**2)
        cfl_diff_y = self.nu * self.dt / (self.dy**2)
        cfl_diff = cfl_diff_x + cfl_diff_y
        
        print(f"CFL Numbers:")
        print(f"  Advective CFL (x-direction):  {cfl_adv_x:.6f}")
        print(f"  Advective CFL (y-direction):  {cfl_adv_y:.6f}")
        print(f"  Advective CFL (maximum):      {cfl_adv:.6f}  [Requirement: ≤ 1.0]")
        print(f"  Diffusive CFL (combined):     {cfl_diff:.6f}  [Requirement: ≤ 0.5]")
        print("-"*70)
        
        is_stable = True
        warnings = []
        
        if cfl_adv > 1.0:
            is_stable = False
            warnings.append(f"⚠ Advective CFL = {cfl_adv:.4f} > 1.0 (UNSTABLE)")
            warnings.append(f"  → Reduce dt to ≤ {0.9 * self.dt / cfl_adv:.6f} s")
        
        if cfl_diff > 0.5:
            is_stable = False
            warnings.append(f"⚠ Diffusive CFL = {cfl_diff:.4f} > 0.5 (UNSTABLE)")
            warnings.append(f"  → Reduce dt to ≤ {0.4 * self.dt / cfl_diff:.6f} s")
        
        if is_stable:
            print("✓ STABILITY: All CFL conditions satisfied - STABLE")
        else:
            print("✗ STABILITY: CFL conditions violated - MAY BE UNSTABLE")
            for warning in warnings:
                print(f"  {warning}")
        
        print("-"*70 + "\n")
        
        return is_stable
    
    def solve_poisson_psi(self):
        """Solve Poisson equation: ∇²ψ = -ω with cylinder BC"""
        nx, ny = self.nx, self.ny
        dx2, dy2 = self.dx**2, self.dy**2
        
        # Count fluid points (excluding cylinder)
        fluid_mask = ~self.is_cylinder
        N = np.sum(fluid_mask)
        
        # Create mapping
        point_to_idx = -np.ones((ny, nx), dtype=int)
        idx_to_point = []
        k = 0
        for i in range(ny):
            for j in range(nx):
                if fluid_mask[i, j]:
                    point_to_idx[i, j] = k
                    idx_to_point.append((i, j))
                    k += 1
        
        A = sparse.lil_matrix((N, N))
        b = np.zeros(N)
        
        for k, (i, j) in enumerate(idx_to_point):
            A[k, k] = -2.0 / dx2 - 2.0 / dy2
            b[k] = -self.omega[i, j]
            
            # Left neighbor
            if j > 0 and fluid_mask[i, j-1]:
                A[k, point_to_idx[i, j-1]] = 1.0 / dx2
            else:
                b[k] -= self.psi[i, max(0, j-1)] / dx2
            
            # Right neighbor
            if j < nx-1 and fluid_mask[i, j+1]:
                A[k, point_to_idx[i, j+1]] = 1.0 / dx2
            else:
                b[k] -= self.psi[i, min(nx-1, j+1)] / dx2
            
            # Bottom neighbor
            if i > 0 and fluid_mask[i-1, j]:
                A[k, point_to_idx[i-1, j]] = 1.0 / dy2
            else:
                b[k] -= self.psi[max(0, i-1), j] / dy2
            
            # Top neighbor
            if i < ny-1 and fluid_mask[i+1, j]:
                A[k, point_to_idx[i+1, j]] = 1.0 / dy2
            else:
                b[k] -= self.psi[min(ny-1, i+1), j] / dy2
        
        A = A.tocsr()
        psi_fluid = spsolve(A, b)
        
        # Update solution
        for k, (i, j) in enumerate(idx_to_point):
            self.psi[i, j] = psi_fluid[k]
    
    def compute_velocity(self):
        """Compute velocity from stream function"""
        # Central difference for interior fluid points
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                if not self.is_cylinder[i, j]:
                    self.u[i, j] = (self.psi[i+1, j] - self.psi[i-1, j]) / (2 * self.dy)
                    self.v[i, j] = -(self.psi[i, j+1] - self.psi[i, j-1]) / (2 * self.dx)
        
        # Boundary conditions
        # Inlet (left)
        self.u[:, 0] = self.U_inf
        self.v[:, 0] = 0.0
        
        # Outlet (right) - zero gradient
        self.u[:, -1] = self.u[:, -2]
        self.v[:, -1] = self.v[:, -2]
        
        # Top and bottom walls - slip condition
        self.u[0, :] = self.U_inf
        self.u[-1, :] = self.U_inf
        self.v[0, :] = 0.0
        self.v[-1, :] = 0.0
        
        # Cylinder surface - no-slip
        self.u[self.is_cylinder] = 0.0
        self.v[self.is_cylinder] = 0.0
    
    def update_omega_boundary(self):
        """Update vorticity at boundaries"""
        dx2, dy2 = self.dx**2, self.dy**2
        
        # Cylinder surface - use Thom's formula for no-slip BC
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                if self.is_cylinder[i, j]:
                    # Count neighboring fluid points
                    neighbors = []
                    if not self.is_cylinder[i-1, j]:
                        neighbors.append(self.psi[i-1, j])
                    if not self.is_cylinder[i+1, j]:
                        neighbors.append(self.psi[i+1, j])
                    if not self.is_cylinder[i, j-1]:
                        neighbors.append(self.psi[i, j-1])
                    if not self.is_cylinder[i, j+1]:
                        neighbors.append(self.psi[i, j+1])
                    
                    if neighbors:
                        avg_psi = np.mean(neighbors)
                        self.omega[i, j] = -2.0 * (self.psi[i, j] - avg_psi) / (self.radius**2 / len(neighbors))
        
        # Inlet
        self.omega[:, 0] = 0.0
        
        # Outlet - zero gradient
        self.omega[:, -1] = self.omega[:, -2]
        
        # Top/bottom walls
        self.omega[0, :] = -2.0 * self.psi[1, :] / dy2
        self.omega[-1, :] = -2.0 * self.psi[-2, :] / dy2
    
    def update_omega(self):
        """Update vorticity using explicit scheme"""
        dx, dy, dt = self.dx, self.dy, self.dt
        
        omega_new = self.omega.copy()
        
        # Update only fluid points
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                if self.is_cylinder[i, j]:
                    continue
                
                # Upwind scheme for advection
                if self.u[i, j] > 0:
                    du_omega = self.u[i, j] * (self.omega[i, j] - self.omega[i, j-1]) / dx
                else:
                    du_omega = self.u[i, j] * (self.omega[i, j+1] - self.omega[i, j]) / dx
                
                if self.v[i, j] > 0:
                    dv_omega = self.v[i, j] * (self.omega[i, j] - self.omega[i-1, j]) / dy
                else:
                    dv_omega = self.v[i, j] * (self.omega[i+1, j] - self.omega[i, j]) / dy
                
                # Diffusion (central difference)
                d2omega_dx2 = (self.omega[i, j+1] - 2*self.omega[i, j] + self.omega[i, j-1]) / dx**2
                d2omega_dy2 = (self.omega[i+1, j] - 2*self.omega[i, j] + self.omega[i-1, j]) / dy**2
                
                # Update
                omega_new[i, j] = self.omega[i, j] + dt * (
                    -du_omega - dv_omega + self.nu * (d2omega_dx2 + d2omega_dy2)
                )
        
        self.omega = omega_new
    
    def compute_pressure(self):
        """Compute pressure using Bernoulli equation (simplified)"""
        self.p = 0.5 * (self.U_inf**2 - (self.u**2 + self.v**2))
    
    def solve(self):
        """Main solver loop"""
        print("="*70)
        print("STARTING FDM TIME INTEGRATION")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        for iteration in range(self.max_iter):
            omega_old = self.omega.copy()
            
            self.solve_poisson_psi()
            self.compute_velocity()
            self.update_omega()
            self.update_omega_boundary()
            
            # Compute residual only in fluid region
            residual = np.max(np.abs((self.omega - omega_old)[~self.is_cylinder]))
            self.convergence_history.append(residual)
            
            # Current physical time
            current_time = (iteration + 1) * self.dt
            
            if (iteration + 1) % 500 == 0:
                elapsed = time.time() - start_time
                iter_per_sec = (iteration + 1) / elapsed
                print(f"Iter {iteration+1:6d}/{self.max_iter}: "
                      f"t = {current_time:8.4f}s | "
                      f"Residual = {residual:.2e} | "
                      f"Speed: {iter_per_sec:.1f} iter/s | "
                      f"Elapsed: {elapsed:.1f}s")
            
            if residual < self.tol:
                elapsed = time.time() - start_time
                self.actual_iterations = iteration + 1
                self.final_time = current_time
                
                print("\n" + "="*70)
                print("✓ FDM SOLVER CONVERGED")
                print("="*70)
                print(f"Convergence achieved at iteration: {iteration+1:,}")
                print(f"Physical time simulated:           {current_time:.6f} s")
                print(f"Final residual:                    {residual:.2e}")
                print(f"Wall-clock time:                   {elapsed:.2f} s")
                print(f"Average speed:                     {(iteration+1)/elapsed:.1f} iter/s")
                print("="*70 + "\n")
                break
        else:
            elapsed = time.time() - start_time
            self.actual_iterations = self.max_iter
            self.final_time = self.total_time
            
            print("\n" + "="*70)
            print("⚠ FDM SOLVER: MAXIMUM ITERATIONS REACHED")
            print("="*70)
            print(f"Maximum iterations:                {self.max_iter:,}")
            print(f"Total physical time simulated:     {self.total_time:.6f} s")
            print(f"Final residual:                    {residual:.2e}")
            print(f"Wall-clock time:                   {elapsed:.2f} s")
            print(f"⚠ Solution may not be fully converged")
            print("="*70 + "\n")
        
        self.compute_pressure()
        self._compute_drag_lift()
        
        return self.u, self.v, self.p
    
    def _compute_drag_lift(self):
        """Compute drag and lift coefficients using surface integration"""
        # Initialize forces
        Fx_pressure = 0.0
        Fy_pressure = 0.0
        Fx_viscous = 0.0
        Fy_viscous = 0.0
        
        # Integrate around cylinder surface
        theta = np.linspace(0, 2*np.pi, 100)
        for k in range(len(theta)-1):
            # Surface element
            th = theta[k]
            dtheta = theta[k+1] - theta[k]
            
            # Point on cylinder surface
            xs = self.cx + self.radius * np.cos(th)
            ys = self.cy + self.radius * np.sin(th)
            
            # Find nearest grid point
            i = np.argmin(np.abs(self.y - ys))
            j = np.argmin(np.abs(self.x - xs))
            
            # Normal vector (pointing outward)
            nx = np.cos(th)
            ny = np.sin(th)
            
            # Pressure force
            p_surf = self.p[i, j]
            Fx_pressure += -p_surf * nx * self.radius * dtheta
            Fy_pressure += -p_surf * ny * self.radius * dtheta
            
            # Viscous force (shear stress)
            # Approximate velocity gradient at surface
            if j < self.nx - 1 and i < self.ny - 1:
                dudy = (self.u[i+1, j] - self.u[i, j]) / self.dy
                dvdx = (self.v[i, j+1] - self.v[i, j]) / self.dx
                
                tau_s = self.nu * (dudy + dvdx)
                Fx_viscous += tau_s * (-ny) * self.radius * dtheta
                Fy_viscous += tau_s * nx * self.radius * dtheta
        
        # Total forces
        Fx_total = Fx_pressure + Fx_viscous
        Fy_total = Fy_pressure + Fy_viscous
        
        # Non-dimensional coefficients
        q_inf = 0.5 * self.U_inf**2
        ref_area = self.diameter  # Per unit length in 2D
        
        self.Cd = Fx_total / (q_inf * ref_area) if q_inf * ref_area > 0 else 0.0
        self.Cl = Fy_total / (q_inf * ref_area) if q_inf * ref_area > 0 else 0.0
        
        self.Cd_pressure = Fx_pressure / (q_inf * ref_area) if q_inf * ref_area > 0 else 0.0
        self.Cd_viscous = Fx_viscous / (q_inf * ref_area) if q_inf * ref_area > 0 else 0.0
        
        print("-"*70)
        print("FORCE COEFFICIENTS")
        print("-"*70)
        print(f"Drag coefficient (Cd):             {self.Cd:.6f}")
        print(f"  - Pressure drag:                 {self.Cd_pressure:.6f}")
        print(f"  - Viscous drag:                  {self.Cd_viscous:.6f}")
        print(f"Lift coefficient (Cl):             {self.Cl:.6f}")
        print("-"*70 + "\n")
    
    def get_solution_at_points(self, x_points, y_points):
        """Interpolate solution at given points"""
        from scipy.interpolate import RegularGridInterpolator
        
        # Create interpolators
        u_interp = RegularGridInterpolator((self.y, self.x), self.u, 
                                           bounds_error=False, fill_value=0.0)
        v_interp = RegularGridInterpolator((self.y, self.x), self.v,
                                           bounds_error=False, fill_value=0.0)
        p_interp = RegularGridInterpolator((self.y, self.x), self.p,
                                           bounds_error=False, fill_value=0.0)
        
        # Interpolate
        points = np.column_stack([y_points, x_points])
        u_vals = u_interp(points)
        v_vals = v_interp(points)
        p_vals = p_interp(points)
        
        return u_vals, v_vals, p_vals

    def save_solution(self, filepath):
        """Save FDM solution to file"""
        solution_data = {
            'u': self.u,
            'v': self.v,
            'p': self.p,
            'psi': self.psi,
            'omega': self.omega,
            'x': self.x,
            'y': self.y,
            'X': self.X,
            'Y': self.Y,
            'Re': self.Re,
            'U_inf': self.U_inf,
            'cx': self.cx,
            'cy': self.cy,
            'radius': self.radius,
            'convergence_history': self.convergence_history,
            'actual_iterations': self.actual_iterations,
            'final_time': self.final_time,
        }
        
        # 如果有 drag/lift 系数
        if hasattr(self, 'Cd'):
            solution_data['Cd'] = self.Cd
            solution_data['Cl'] = self.Cl
            solution_data['Cd_pressure'] = self.Cd_pressure
            solution_data['Cd_viscous'] = self.Cd_viscous
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(solution_data, f)
        
        print(f"✓ FDM solution saved to {filepath}")
    
    @classmethod
    def load_solution(cls, filepath, config):
        """
        Load FDM solution from file
        
        Args:
            filepath: Path to saved solution
            config: Configuration dict (for initialization)
        
        Returns:
            CylinderFDM instance with loaded solution
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"FDM solution file not found: {filepath}")
        
        print(f"Loading FDM solution from {filepath}...")
        
        # Create instance (without solving)
        solver = cls(
            Re=config['Re_target'],
            U_inf=config['U_inf'],
            cylinder_center=config['cylinder_center'],
            cylinder_radius=config['cylinder_radius'],
            nx=config['fdm_nx'],
            ny=config['fdm_ny'],
            x_domain=config['x_domain'],
            y_domain=config['y_domain'],
            dt=config['fdm_dt'],
            max_iter=config['fdm_max_iter'],
            tol=config['fdm_tolerance'],
            config=config
        )
        
        # Load solution data
        with open(filepath, 'rb') as f:
            solution_data = pickle.load(f)
        
        # Restore solution
        solver.u = solution_data['u']
        solver.v = solution_data['v']
        solver.p = solution_data['p']
        solver.psi = solution_data['psi']
        solver.omega = solution_data['omega']
        solver.convergence_history = solution_data['convergence_history']
        solver.actual_iterations = solution_data['actual_iterations']
        solver.final_time = solution_data['final_time']
        
        # Restore drag/lift if available
        if 'Cd' in solution_data:
            solver.Cd = solution_data['Cd']
            solver.Cl = solution_data['Cl']
            solver.Cd_pressure = solution_data['Cd_pressure']
            solver.Cd_viscous = solution_data['Cd_viscous']
        
        print(f"✓ FDM solution loaded successfully")
        print(f"  Iterations: {solver.actual_iterations}")
        print(f"  Final time: {solver.final_time:.6f} s")
        if hasattr(solver, 'Cd'):
            print(f"  Cd = {solver.Cd:.6f}, Cl = {solver.Cl:.6f}")
        
        return solver