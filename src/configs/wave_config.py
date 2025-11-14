"""
Configuration for 1D Wave Equation experiments
"""

# Default configuration for 1D Wave equation
WAVE_CONFIG = {
    # Physical parameters
    # PDE: u_tt = c^2 * u_xx
    "c": 1.0,  # Wave speed (课程学习参数)
    "x_domain": [0.0, 1.0],
    "t_domain": [0.0, 2.0],
    
    # Initial conditions
    # u(x, 0) = sin(k*pi*x)
    # u_t(x, 0) = 0
    "k": 1,  # Wave number (影响初始条件复杂度)
    
    # Neural network architecture
    "model_type": "mlp",
    "layers": [2, 64, 64, 64, 64, 1],
    "activation": "tanh",
    
    # Training parameters
    "epochs": 50000,
    "learning_rate": 1e-3,
    "optimizer": "adam",
    "use_scheduler": True,
    "lr_decay_steps": 5000,
    "lr_decay_rate": 0.9,
    
    # Curriculum learning parameters
    "curriculum_ramp_ratio": 0.4,  # 40% of training for c ramp
    
    # Data points
    "N_ic": 200,    # Initial condition points (u and u_t)
    "N_bc": 200,    # Boundary condition points
    "N_pde": 10000, # PDE collocation points
    
    # Data loss parameters
    "use_data_loss": False,
    "N_data": 500,
    "fdm_solution": None,
    
    # Loss weights
    "lambda_ic": 10.0,   # 初始位移权重
    "lambda_ic_t": 10.0, # 初始速度权重
    "lambda_bc": 10.0,
    "lambda_pde": 1.0,
    "lambda_data": 1.0,
    
    # FDM parameters
    "fdm_nx": 1024,
    "fdm_nt": 10000,
    
    # Visualization
    "plot_t_snapshots": [0.0, 0.5, 1.0, 1.5, 2.0],
    "plot_x_points": 512,
}


# Configuration with data loss
WAVE_WITH_DATA = {
    **WAVE_CONFIG,
    "use_data_loss": True,
    "N_data": 1000,
    "lambda_data": 5.0,
}


# Low wave speed (easier)
WAVE_LOW_SPEED = {
    **WAVE_CONFIG,
    "c": 0.5,
    "epochs": 10000,
}


# High wave speed (harder)
WAVE_HIGH_SPEED = {
    **WAVE_CONFIG,
    "c": 5.0,
    "epochs": 30000,
    "N_pde": 20000,
}


# Multi-frequency (complex initial condition)
WAVE_MULTI_FREQ = {
    **WAVE_CONFIG,
    "k": 3,  # Higher frequency
    "epochs": 25000,
}


# Configuration with ResNet
WAVE_RESNET = {
    **WAVE_CONFIG,
    "model_type": "resnet",
    "input_dim": 2,
    "hidden_dim": 64,
    "output_dim": 1,
    "num_blocks": 4,
}


# Configuration with Fourier features
WAVE_FOURIER = {
    **WAVE_CONFIG,
    "model_type": "fourier",
    "layers": [256, 64, 64, 64, 1],
    "input_dim": 2,
    "fourier_dim": 256,
    "sigma": 2.0,
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
        'default': WAVE_CONFIG,
        'with_data': WAVE_WITH_DATA,
        'low_speed': WAVE_LOW_SPEED,
        'high_speed': WAVE_HIGH_SPEED,
        'multi_freq': WAVE_MULTI_FREQ,
        'resnet': WAVE_RESNET,
        'fourier': WAVE_FOURIER,
    }
    
    if config_name not in configs:
        print(f"Warning: Unknown config '{config_name}', using default")
        return WAVE_CONFIG
    
    return configs[config_name].copy()