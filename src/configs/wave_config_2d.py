# 文件名: src/configs/wave_config_2d.py

"""
Configuration for 2D Wave Equation experiments
"""

# Default configuration for 2D Wave equation
WAVE_CONFIG_2D = {
    # Physical parameters
    # PDE: u_tt = c^2 * (u_xx + u_yy)
    "c": 2.0,  # Target Wave speed
    "x_domain": [0.0, 1.0],
    "y_domain": [0.0, 1.0],
    "t_domain": [0.0, 1.0],
    
    # Initial conditions
    # u(x, y, 0) = sin(kx*pi*x) * sin(ky*pi*y)
    # u_t(x, y, 0) = 0
    "kx": 1,  # Wave number in x
    "ky": 1,  # Wave number in y
    
    # Neural network architecture
    "model_type": "mlp",
    "layers": [3, 64, 64, 64, 64, 1], # Input is (x, y, t)
    "activation": "tanh",
    
    # Training parameters
    "epochs": 50000,
    "learning_rate": 1e-3,
    "optimizer": "adam",
    "use_scheduler": True,
    "lr_decay_steps": 5000,
    "lr_decay_rate": 0.9,
    
    # Curriculum learning stages
    # 定义课程学习的三个阶段
    "curriculum_stages": [
        # 阶段1: 使用较低的波速 c_initial 进行训练，更容易学习基本模式
        {"name": "Stage 1: Easy Start", "c": 0.5, "epochs": 10000},
        # 阶段2: 将波速 c 从 c_initial 线性增加到 c_target
        # {"name": "Stage 2: Annealing", "c_start": 1, "c_end": 1.0, "epochs": 20000},
        {"name": "Stage 2: Annealing", "c": 1.0, "epochs": 20000},
        # 阶段3: 使用目标波速 c_target 进行精细训练
        {"name": "Stage 3: Fine-tuning", "c": 2.0, "epochs": 20000}
    ],
    
    # Data points
    "N_ic": 1000,    # Initial condition points
    "N_bc": 1000,   # Boundary condition points (total for 4 boundaries)
    "N_pde": 20000, # PDE collocation points
    
    # Loss weights
    "lambda_ic": 1.0,   # 初始位移权重
    "lambda_ic_t": 1.0, # 初始速度权重
    "lambda_bc": 1.0,
    "lambda_pde": 1.0,
    
    # FDM parameters for ground truth
    "fdm_nx": 101,
    "fdm_ny": 101,
    "fdm_nt": 2001, # 增加nt以保证CFL稳定性
    
    # Visualization
    "plot_t_snapshots": [0.0, 0.25, 0.5, 0.75, 1.0],
    "plot_xy_points": 101,
}

def get_config(config_name='default'):
    """
    Get configuration by name
    """
    configs = {
        'default': WAVE_CONFIG_2D,
    }
    
    if config_name not in configs:
        print(f"Warning: Unknown config '{config_name}', using default")
        return WAVE_CONFIG_2D.copy()
    
    return configs[config_name].copy()