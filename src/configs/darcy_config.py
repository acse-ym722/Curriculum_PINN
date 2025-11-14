"""
Configuration for 2D Darcy Flow with Source/Sink
Curriculum: Single Source → Source + Sink
"""
import numpy as np
import torch


def gaussian_source(x, y, x0, y0, strength, radius=0.08):
    """高斯源/汇项"""
    r_sq = (x - x0)**2 + (y - y0)**2
    return strength * np.exp(-r_sq / (2 * radius**2))


def source_single(x, y):
    """单个源"""
    return gaussian_source(x, y, x0=0.3, y0=0.25, strength=5.0, radius=0.08)


def source_sink_pair(x, y):
    """一源一汇"""
    source = gaussian_source(x, y, x0=0.3, y0=0.25, strength=5.0, radius=0.08)
    sink = gaussian_source(x, y, x0=0.7, y0=0.25, strength=-5.0, radius=0.08)
    return source + sink


# PyTorch版本（用于PINN）
def torch_gaussian_source(x, y, x0, y0, strength, radius=0.08):
    """PyTorch版本的高斯源"""
    r_sq = (x - x0)**2 + (y - y0)**2
    return strength * torch.exp(-r_sq / (2 * radius**2))


def torch_source_single(x, y):
    """PyTorch: 单个源"""
    return torch_gaussian_source(x, y, x0=0.3, y0=0.25, strength=5.0, radius=0.08)


def torch_source_sink_pair(x, y):
    """PyTorch: 一源一汇"""
    source = torch_gaussian_source(x, y, x0=0.3, y0=0.25, strength=5.0, radius=0.08)
    sink = torch_gaussian_source(x, y, x0=0.7, y0=0.25, strength=-5.0, radius=0.08)
    return source + sink


# ========== 配置字典 ==========
DARCY_CONFIG = {
    # Domain (2:1 aspect ratio)
    "x_domain": [0.0, 1.0],
    "y_domain": [0.0, 0.5],
    
    # Physical parameters
    "K": 1.0,  # 均匀渗透率
    
    # Boundary conditions (全零压力边界)
    "bc_left": 0.0,
    "bc_right": 0.0,
    "bc_top": 0.0,
    "bc_bottom": 0.0,
    
    # FDM grid
    "fdm_nx": 128,
    "fdm_ny": 64,
    
    # Neural network
    "model_type": "mlp",
    "layers": [2, 128, 128, 128, 1],
    "activation": "tanh",
    
    # Training
    "epochs": 15000,
    "learning_rate": 1e-3,
    "optimizer": "adam",
    "use_scheduler": True,
    "lr_decay_steps": 2000,
    "lr_decay_rate": 0.9,
    
    # Sampling
    "N_collocation": 2000,
    "N_bc": 400,
    "N_data": 500,
    
    # Loss weights
    "lambda_pde": 1.0,
    "lambda_bc_dirichlet": 10.0,
    "lambda_bc_neumann": 5.0,
    "lambda_data": 100.0,
    
    # Data loss
    "use_data_loss": True,
    
    # Curriculum stages
    "curriculum_stages": [
        {
            "name": "Stage 1: Single Source",
            "epochs": 7000,
            "source_type": "single",
            "f_source_np": source_single,        # NumPy版本（FDM用）
            "f_source_torch": torch_source_single,  # PyTorch版本（PINN用）
            "n_collocation": 2000,
            "lambda_pde": 1.0,
            "lambda_bc_dirichlet": 10.0,
            "lambda_data": 100.0,
        },
        {
            "name": "Stage 2: Source + Sink",
            "epochs": 8000,
            "source_type": "pair",
            "f_source_np": source_sink_pair,
            "f_source_torch": torch_source_sink_pair,
            "n_collocation": 2000,
            "lambda_pde": 1.0,
            "lambda_bc_dirichlet": 15.0,
            "lambda_data": 150.0,
        },
    ]
}


def get_config(name='default'):
    """获取配置"""
    if name == 'default':
        return DARCY_CONFIG.copy()
    else:
        raise ValueError(f"Unknown config: {name}")