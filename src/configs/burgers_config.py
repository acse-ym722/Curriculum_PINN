"""
Configuration for Burgers equation experiments
"""

# Default configuration for Burgers equation
BURGERS_CONFIG = {
    # Physical parameters
    "nu": 0.002,  # Viscosity coefficient
    "x_domain": [-1.0, 1.0],
    "t_domain": [0.0, 1.0],
    
    # Neural network architecture
    "model_type": "mlp",  # Options: 'mlp', 'resnet', 'fourier'
    "layers": [2, 32, 32, 32, 32, 32, 1],
    "activation": "tanh",
    
    # Training parameters
    "epochs": 20000,
    "learning_rate": 5e-4,
    "optimizer": "adam",
    "use_scheduler": True,
    "lr_decay_steps": 5000,
    "lr_decay_rate": 0.9,
    
    # Curriculum learning parameters
    "curriculum_ramp_ratio": 0.4,  # Portion of training for beta ramp-up
    
    # Data points
    "N_ic": 100,   # Initial condition points
    "N_bc": 100,   # Boundary condition points
    "N_pde": 10000,  # PDE collocation points
    
    # Loss weights
    "lambda_ic": 1.0,
    "lambda_bc": 1.0,
    "lambda_pde": 1.0,
    
    # Numerical solution (ground truth)
    "NX": 512,     # Spatial grid points
    "NT": 10000,   # Time steps
    
    # Visualization
    "plot_t_final": 0.01,  # Time for final comparison
    "plot_x_points": 512,  # Spatial resolution for plotting
}


# High viscosity configuration (easier problem)
BURGERS_HIGH_NU = {
    **BURGERS_CONFIG,
    "nu": 0.01,
    "epochs": 10000,
}


# Low viscosity configuration (harder problem)
BURGERS_LOW_NU = {
    **BURGERS_CONFIG,
    "nu": 0.001,
    "epochs": 30000,
    "N_pde": 20000,
}


# Configuration with ResNet architecture
BURGERS_RESNET = {
    **BURGERS_CONFIG,
    "model_type": "resnet",
    "input_dim": 2,
    "hidden_dim": 64,
    "output_dim": 1,
    "num_blocks": 4,
}


# Configuration with Fourier features
BURGERS_FOURIER = {
    **BURGERS_CONFIG,
    "model_type": "fourier",
    "layers": [256, 64, 64, 64, 1],  # After Fourier embedding
    "input_dim": 2,
    "fourier_dim": 256,
    "sigma": 1.0,
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
        'default': BURGERS_CONFIG,
        'high_nu': BURGERS_HIGH_NU,
        'low_nu': BURGERS_LOW_NU,
        'resnet': BURGERS_RESNET,
        'fourier': BURGERS_FOURIER,
    }
    
    if config_name not in configs:
        print(f"Warning: Unknown config '{config_name}', using default")
        return BURGERS_CONFIG
    
    return configs[config_name].copy()