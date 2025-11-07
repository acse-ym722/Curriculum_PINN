"""
Configuration for lid-driven cavity flow experiments
"""

# Default configuration for cavity flow
CAVITY_CONFIG = {
    # Physical parameters
    "Re_target": 100,  # Target Reynolds number
    "x_domain": [0.0, 1.0],
    "y_domain": [0.0, 1.0],
    
    # Neural network architecture
    "model_type": "mlp",
    "layers": [2, 64, 128, 128, 128, 64, 3],  # Output: [u, v, p]
    "activation": "tanh",
    
    # Standard PINN training parameters
    "pinn_epochs": 80000,
    "pinn_learning_rate": 5e-4,
    "pinn_optimizer": "adam",
    "pinn_use_scheduler": True,
    "pinn_lr_decay_steps": 5000,
    "pinn_lr_decay_rate": 0.9,
    
    # Curriculum PINN parameters
    "curriculum_total_epochs": 80000,
    "curriculum_learning_rate": 5e-4,
    "curriculum_optimizer": "adam",
    "curriculum_use_scheduler": True,
    "curriculum_lr_decay_steps": 5000,
    "curriculum_lr_decay_rate": 0.9,
    
    # Curriculum stages
    "curriculum_stages": [
        {"Re": 10, "epochs": 5000},
        {"Re": 30, "epochs": 5000},
        {"Re": 60, "epochs": 5000},
        {"Re": 100, "epochs": 65000},
    ],
    
    # Data points
    "N_bc": 1000,    # Boundary condition points
    "N_pde": 10000,  # PDE collocation points
    
    # Data loss parameters
    "use_data_loss": False,      # 默认不使用，实验时设置为True
    "N_data": 500,               # 数据采样点数量
    "fdm_solver": None,          # FDM solver对象，运行时指定
    
    # Loss weights
    "lambda_bc": 15.0,
    "lambda_pde": 1.0,
    "lambda_cont": 1.0,  # Continuity equation weight
    "lambda_data": 1.0,  # Data loss weight
    
    # Numerical solution (ground truth)
    "fdm_grid_size": 64,
    "fdm_max_iter": 10000,
    "fdm_tolerance": 1e-6,
    "fdm_dt": 0.0001,
    
    # Visualization
    "plot_resolution": 100,
}


# Configuration with data loss enabled
CAVITY_WITH_DATA = {
    **CAVITY_CONFIG,
    "use_data_loss": True,
    "N_data": 1000,
    "lambda_data": 10.0,  # 增加data loss权重
}


# Low Reynolds number (easier problem)
CAVITY_LOW_RE = {
    **CAVITY_CONFIG,
    "Re_target": 100,
}


# High Reynolds number (harder problem)
CAVITY_HIGH_RE = {
    **CAVITY_CONFIG,
    "Re_target": 400,
    "pinn_epochs": 80000,
    "curriculum_total_epochs": 80000,
    "curriculum_stages": [
        {"Re": 10, "epochs": 4000},
        {"Re": 50, "epochs": 4000},
        {"Re": 100, "epochs": 6000},
        {"Re": 200, "epochs": 6000},
        {"Re": 400, "epochs": 60000},
    ],
    "N_pde": 20000,
    "fdm_grid_size": 128,
}


# Configuration with deeper network
CAVITY_DEEP = {
    **CAVITY_CONFIG,
    "layers": [2, 128, 128, 128, 128, 128, 128, 3],
    "pinn_epochs": 30000,
}


# Configuration with ResNet architecture
CAVITY_RESNET = {
    **CAVITY_CONFIG,
    "model_type": "resnet",
    "input_dim": 2,
    "hidden_dim": 128,
    "output_dim": 3,
    "num_blocks": 6,
}


def get_config(config_name='default'):
    """
    Get configuration by name
    
    Args:
        config_name: Name of configuration
    
    Returns:
        Configuration dictionary
    """
    configs = {
        'default': CAVITY_CONFIG,
        'with_data': CAVITY_WITH_DATA,
        'low_re': CAVITY_LOW_RE,
        'high_re': CAVITY_HIGH_RE,
        'deep': CAVITY_DEEP,
        'resnet': CAVITY_RESNET,
    }
    
    if config_name not in configs:
        print(f"Warning: Unknown config '{config_name}', using default")
        return CAVITY_CONFIG
    
    return configs[config_name].copy()