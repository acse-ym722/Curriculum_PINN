"""
Configuration for Burgers equation experiments
"""

# Default configuration for Burgers equation
BURGERS_CONFIG = {
    # Physical parameters
    "nu": 0.002,
    "x_domain": [-1.0, 1.0],
    "t_domain": [0.0, 1.0],
    
    # Neural network architecture
    "model_type": "mlp",
    "layers": [2, 32, 32, 32, 32, 32, 1],
    "activation": "tanh",
    
    # Training parameters
    "epochs": 20000,
    "learning_rate": 5e-4,
    "optimizer": "adam",
    "use_scheduler": True,
    "lr_decay_steps": 5000,
    "lr_decay_rate": 0.9,
    
    # ================= Stage-based Curriculum Learning =================
    "nu_initial": 0.02, 
    "nu_target": 0.002, 
    
    # ✅ 每个阶段都必须包含 "nu" 键
    "curriculum_stages": [
        {
            "name": "Stage 1: High Viscosity", 
            "epochs": 3000,
            "nu": 0.02  # 高粘性（简单）
        },
        {
            "name": "Stage 2: Annealing", 
            "epochs": 5000,
            "nu": 0.01  # 中等粘性（过渡）
        },
        {
            "name": "Stage 3: Target Viscosity", 
            "epochs": 12000,
            "nu": 0.002  # 目标粘性（困难）
        }
    ],
    # ==================================================================
    
    # Data points
    "N_ic": 100,
    "N_bc": 100,
    "N_pde": 10000,

    # Data loss parameters
    "use_data_loss": True,
    "N_data": 1000,
    "fdm_solution": None,

    # Loss weights
    "lambda_ic": 1.0,
    "lambda_bc": 1.0,
    "lambda_pde": 1.0,
    "lambda_data": 10.0,

    # Numerical solution (ground truth)
    "fdm_nx": 512,
    "fdm_nt": 10000,
    
    # Visualization
    "plot_t_final": 0.99,
    "plot_x_points": 512,
}


# High viscosity configuration
BURGERS_HIGH_NU = {
    **BURGERS_CONFIG,
    "nu": 0.01,
    "nu_target": 0.01,  # ✅ 同步修改
    "epochs": 10000,
    "curriculum_stages": [
        {"name": "Stage 1", "epochs": 3000, "nu": 0.05},
        {"name": "Stage 2", "epochs": 4000, "nu": 0.02},
        {"name": "Stage 3", "epochs": 3000, "nu": 0.01}
    ],
}


# Low viscosity configuration
BURGERS_LOW_NU = {
    **BURGERS_CONFIG,
    "nu": 0.001,
    "nu_target": 0.001,  # ✅ 同步修改
    "epochs": 30000,
    "N_pde": 20000,
    "curriculum_stages": [
        {"name": "Stage 1", "epochs": 10000, "nu": 0.01},
        {"name": "Stage 2", "epochs": 10000, "nu": 0.005},
        {"name": "Stage 3", "epochs": 10000, "nu": 0.001}
    ],
}


# ResNet configuration
BURGERS_RESNET = {
    **BURGERS_CONFIG,
    "model_type": "resnet",
    "input_dim": 2,
    "hidden_dim": 64,
    "output_dim": 1,
    "num_blocks": 5,
}


# Fourier configuration
BURGERS_FOURIER = {
    **BURGERS_CONFIG,
    "model_type": "fourier",
    "layers": [256, 64, 64, 64, 1],
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