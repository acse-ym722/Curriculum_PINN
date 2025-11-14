"""
Finite Difference Method solver for 2D Darcy Flow with Source/Sink
"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import time


class DarcyFDM:
    """
    FDM solver for: -K * ∇²p = f(x, y)
    """
    
    def __init__(self, K=1.0, f_source=None, x_domain=(0.0, 1.0), 
                 y_domain=(0.0, 0.5), nx=128, ny=64, 
                 bc_left=0.0, bc_right=0.0, bc_top=0.0, bc_bottom=0.0):
        self.K = K
        self.f_source = f_source  # 函数 f(x, y)
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.nx = nx
        self.ny = ny
        
        # BCs
        self.bc_left = bc_left
        self.bc_right = bc_right
        self.bc_top = bc_top
        self.bc_bottom = bc_bottom
        
        # Grid
        self.x = np.linspace(x_domain[0], x_domain[1], nx)
        self.y = np.linspace(y_domain[0], y_domain[1], ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        self.p = np.zeros((nx, ny))
    
    def solve(self):
        """求解"""
        print("\n" + "-"*60)
        print("Solving 2D Darcy Flow with Source/Sink (FDM)")
        print("-"*60)
        print(f"  Grid: {self.nx} × {self.ny}")
        print(f"  dx = {self.dx:.6f}, dy = {self.dy:.6f}")
        print(f"  K = {self.K}")
        print("-"*60)
        
        start_time = time.time()
        
        # 构建线性系统
        A, b = self._build_system()
        
        # 求解
        print("  Solving linear system...")
        p_flat = spsolve(A.tocsr(), b)
        self.p = p_flat.reshape((self.nx, self.ny))
        
        elapsed = time.time() - start_time
        print(f"  ✓ FDM completed in {elapsed:.2f}s")
        print("-"*60 + "\n")
        
        return self.p, self.x, self.y
    
    def _build_system(self):
        """构建稀疏矩阵"""
        N = self.nx * self.ny
        A = lil_matrix((N, N))
        b = np.zeros(N)
        
        cx = self.K / (self.dx ** 2)
        cy = self.K / (self.dy ** 2)
        c_center = -2 * (cx + cy)
        
        def idx(i, j):
            return i * self.ny + j
        
        # 计算源项
        if callable(self.f_source):
            F = self.f_source(self.X, self.Y)
        else:
            F = np.zeros_like(self.X)
        
        # 内部点
        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                k = idx(i, j)
                A[k, idx(i-1, j)] = cx
                A[k, idx(i+1, j)] = cx
                A[k, idx(i, j-1)] = cy
                A[k, idx(i, j+1)] = cy
                A[k, k] = c_center
                b[k] = -F[i, j]
        
        # 边界条件（Dirichlet）
        # Left
        for j in range(self.ny):
            k = idx(0, j)
            A[k, :] = 0
            A[k, k] = 1.0
            b[k] = self.bc_left
        
        # Right
        for j in range(self.ny):
            k = idx(self.nx - 1, j)
            A[k, :] = 0
            A[k, k] = 1.0
            b[k] = self.bc_right
        
        # Bottom
        for i in range(1, self.nx - 1):
            k = idx(i, 0)
            A[k, :] = 0
            A[k, k] = 1.0
            b[k] = self.bc_bottom
        
        # Top
        for i in range(1, self.nx - 1):
            k = idx(i, self.ny - 1)
            A[k, :] = 0
            A[k, k] = 1.0
            b[k] = self.bc_top
        
        return A, b