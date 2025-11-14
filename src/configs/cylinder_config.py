"""
Configuration for 2D flow past a cylinder
Only 3 predefined configs: Low Re, Medium Re, High Re
"""

# ==================== 边界条件定义 ====================
BOUNDARY_CONDITIONS = {
    # 入口边界 (x = x_min)
    "inlet": {
        "type": "dirichlet",
        "u": "U_inf",      # 速度 u = U_inf
        "v": 0.0,          # 速度 v = 0
        "p": 0.0,          # 压力参考值设为0（或者不施加约束）
        "omega": 0.0,      # 涡量 = 0 (势流)
    },
    
    # 出口边界 (x = x_max)
    "outlet": {
        "type": "neumann",  # 零梯度
        "du_dx": 0.0,
        "dv_dx": 0.0,
        "dp_dx": 0.0,
        "domega_dx": 0.0,
    },
    
    # 上下壁面 (y = y_min, y_max)
    "walls": {
        "type": "slip",     # 滑移边界（远场）
        "u": "U_inf",       # 切向速度 = U_inf
        "v": 0.0,           # 法向速度 = 0 (无穿透)
        "omega_bc": "thom", # 涡量用 Thom's formula
    },
    
    # 圆柱表面
    "cylinder": {
        "type": "no_slip",  # 无滑移
        "u": 0.0,
        "v": 0.0,
        "omega_bc": "thom", # 涡量用 Thom's formula
    }
}

# ==================== 初始条件定义 ====================
INITIAL_CONDITIONS = {
    "type": "potential_flow",  # 默认使用势流初始化
    
    # 均匀流初始条件
    "uniform": {
        "u": "U_inf",
        "v": 0.0,
        "p": 0.0,
        "omega": 0.0,
        "psi": "U_inf * y",
    },
    
    # 圆柱势流解初始条件（推荐）
    "potential_flow": {
        "description": "Analytical potential flow solution around cylinder",
        "formula": {
            "u": "U_inf * (1 - R²/r² * (1 - 2*sin²θ))",
            "v": "-U_inf * R²/r² * 2*cosθ*sinθ",
            "psi": "U_inf * (r - R²/r) * sinθ",
            "omega": 0.0,
        }
    },
}

# ==================== 基础配置 ====================
BASE_CONFIG = {
    # Physical parameters (will be overridden by specific configs)
    "Re_target": 100,
    "U_inf": 1.0,
    
    # Domain
    "x_domain": [-5.0, 15.0],
    "y_domain": [-5.0, 5.0],
    
    # Cylinder geometry
    "cylinder_center": [0.0, 0.0],
    "cylinder_radius": 0.5,
    
    # Boundary and initial conditions
    "boundary_conditions": BOUNDARY_CONDITIONS,
    "initial_conditions": INITIAL_CONDITIONS,
    "initial_condition_type": "potential_flow",  # 默认使用势流初始化
    
    # Neural network architecture
    "model_type": "mlp",
    "layers": [2, 128, 256, 256, 256, 128, 3],
    "activation": "tanh",
    
    # Standard PINN training
    "pinn_epochs": 60000,
    "pinn_learning_rate": 1e-3,
    "pinn_optimizer": "adam",
    "pinn_use_scheduler": True,
    "pinn_lr_decay_steps": 5000,
    "pinn_lr_decay_rate": 0.95,
    
    # Curriculum PINN (will be overridden)
    "curriculum_total_epochs": 60000,
    "curriculum_learning_rate": 1e-3,
    "curriculum_optimizer": "adam",
    "curriculum_use_scheduler": True,
    "curriculum_lr_decay_steps": 5000,
    "curriculum_lr_decay_rate": 0.95,
    
    # Curriculum stages (will be overridden)
    "curriculum_stages": [
        {"Re": 10, "epochs": 5000},
        {"Re": 40, "epochs": 5000},
        {"Re": 100, "epochs": 50000},
    ],
    
    # Collocation points
    "N_bc_inlet": 500,
    "N_bc_outlet": 500,
    "N_bc_wall": 200,
    "N_bc_cylinder": 800,
    "N_pde": 20000,
    
    # ========== Data loss parameters (默认启用) ==========
    "use_data_loss": True,
    "N_data": 1000,
    "lambda_data": 10.0,
    
    # Loss weights
    "lambda_bc_inlet": 10.0,
    "lambda_bc_outlet": 5.0,
    "lambda_bc_wall": 10.0,
    "lambda_bc_cylinder": 20.0,
    "lambda_pde": 1.0,
    "lambda_cont": 1.0,
    
    # Numerical solution (FDM) - will be overridden
    "fdm_nx": 200,
    "fdm_ny": 100,
    "fdm_max_iter": 20000,
    "fdm_tolerance": 1e-6,
    "fdm_dt": 0.001,
    
    # Visualization
    "plot_resolution": 150,
}


# ==================== 1. 低雷诺数 (Re=40) ====================
# 稳定流动，有对称的尾流回流区
LOW_RE_CONFIG = {
    **BASE_CONFIG,
    
    # Physical parameters
    "Re_target": 40,
    
    # PINN training - 较少迭代即可收敛
    "pinn_epochs": 30000,
    
    # Curriculum stages
    "curriculum_total_epochs": 30000,
    "curriculum_stages": [
        {"Re": 10, "epochs": 3000},
        {"Re": 20, "epochs": 3000},
        {"Re": 40, "epochs": 24000},
    ],
    
    # Collocation points - 适度减少
    "N_bc_inlet": 300,
    "N_bc_outlet": 300,
    "N_bc_wall": 150,
    "N_bc_cylinder": 500,
    "N_pde": 12000,
    
    # Data loss
    "use_data_loss": True,
    "N_data": 800,
    "lambda_data": 10.0,
    
    # FDM solver - 较粗网格和较少迭代
    "fdm_nx": 120,
    "fdm_ny": 60,
    "fdm_max_iter": 10000,    # 10秒物理时间
    "fdm_tolerance": 1e-6,
    "fdm_dt": 0.001,
    
    # Visualization
    "plot_resolution": 100,
}


# ==================== 2. 中等雷诺数 (Re=100) ====================
# 开始出现Kármán涡街
MEDIUM_RE_CONFIG = {
    **BASE_CONFIG,
    
    # Physical parameters
    "Re_target": 100,
    
    # PINN training - 标准配置
    "pinn_epochs": 60000,
    
    # Curriculum stages
    "curriculum_total_epochs": 60000,
    "curriculum_stages": [
        {"Re": 10, "epochs": 5000},
        {"Re": 40, "epochs": 5000},
        {"Re": 100, "epochs": 50000},
    ],
    
    # Collocation points - 标准配置
    "N_bc_inlet": 500,
    "N_bc_outlet": 500,
    "N_bc_wall": 200,
    "N_bc_cylinder": 800,
    "N_pde": 20000,
    
    # Data loss
    "use_data_loss": True,
    "N_data": 1000,
    "lambda_data": 10.0,
    
    # FDM solver - 标准配置
    "fdm_nx": 200,
    "fdm_ny": 100,
    "fdm_max_iter": 20000,    # 20秒物理时间
    "fdm_tolerance": 1e-6,
    "fdm_dt": 0.001,
    
    # Visualization
    "plot_resolution": 150,
}


# ==================== 3. 高雷诺数 (Re=200) ====================
# 强涡街脱落，尾流复杂
HIGH_RE_CONFIG = {
    **BASE_CONFIG,
    
    # Physical parameters
    "Re_target": 200,
    
    # PINN training - 需要更多迭代
    "pinn_epochs": 100000,
    
    # Curriculum stages - 更细致的课程学习
    "curriculum_total_epochs": 100000,
    "curriculum_stages": [
        {"Re": 20, "epochs": 10000},
        {"Re": 60, "epochs": 10000},
        {"Re": 100, "epochs": 20000},
        {"Re": 200, "epochs": 60000},
    ],
    
    # Collocation points - 增加采样密度
    "N_bc_inlet": 600,
    "N_bc_outlet": 600,
    "N_bc_wall": 250,
    "N_bc_cylinder": 1000,
    "N_pde": 30000,
    
    # Data loss
    "use_data_loss": True,
    "N_data": 1500,
    "lambda_data": 12.0,
    
    # FDM solver - 更精细的网格
    "fdm_nx": 250,
    "fdm_ny": 120,
    "fdm_max_iter": 40000,    # 20秒物理时间（更小时间步）
    "fdm_tolerance": 1e-6,
    "fdm_dt": 0.0005,         # 更小的时间步以捕捉涡街
    
    # Visualization
    "plot_resolution": 180,
}


# ==================== 配置字典 ====================
CONFIGS = {
    'low_re': LOW_RE_CONFIG,      # Re=40
    'medium_re': MEDIUM_RE_CONFIG, # Re=100
    'high_re': HIGH_RE_CONFIG,     # Re=200
}


def get_config(config_name='medium_re'):
    """
    Get configuration by name
    
    Args:
        config_name: 'low_re', 'medium_re', or 'high_re'
    
    Returns:
        Configuration dictionary
    """
    if config_name not in CONFIGS:
        print(f"⚠ Warning: Unknown config '{config_name}', using 'medium_re'")
        return MEDIUM_RE_CONFIG.copy()
    
    return CONFIGS[config_name].copy()


def print_config_info(config_name='medium_re'):
    """Print detailed information about a configuration"""
    config = get_config(config_name)
    
    Re = config['Re_target']
    
    print("\n" + "="*80)
    print(f"CONFIGURATION: {config_name.upper()} (Re={Re})")
    print("="*80)
    
    print("\n📊 Physical Parameters:")
    print(f"  Reynolds Number (Re):         {Re}")
    print(f"  Free Stream Velocity (U∞):    {config['U_inf']} m/s")
    print(f"  Cylinder Diameter (D):        {2*config['cylinder_radius']} m")
    nu = config['U_inf'] * 2 * config['cylinder_radius'] / Re
    print(f"  Kinematic Viscosity (ν):      {nu:.6e} m²/s")
    
    # Flow regime description
    print(f"\n🌊 Flow Regime:")
    if Re <= 5:
        regime = "Creeping flow (Stokes flow)"
    elif Re <= 40:
        regime = "Steady symmetric flow with recirculation"
    elif Re <= 47:
        regime = "Steady asymmetric flow"
    elif Re <= 150:
        regime = "Laminar vortex shedding (Kármán vortex street)"
    elif Re <= 300:
        regime = "Transitional flow"
    else:
        regime = "Turbulent wake (2D simulation limit)"
    print(f"  {regime}")
    
    print(f"\n📐 Domain:")
    print(f"  x: [{config['x_domain'][0]}, {config['x_domain'][1]}] "
          f"({config['x_domain'][1] - config['x_domain'][0]:.1f} m)")
    print(f"  y: [{config['y_domain'][0]}, {config['y_domain'][1]}] "
          f"({config['y_domain'][1] - config['y_domain'][0]:.1f} m)")
    print(f"  Cylinder center: ({config['cylinder_center'][0]}, {config['cylinder_center'][1]})")
    print(f"  Cylinder radius: {config['cylinder_radius']} m")
    
    print(f"\n🧠 Neural Network:")
    print(f"  Architecture: {config['model_type'].upper()}")
    print(f"  Layers: {config['layers']}")
    print(f"  Activation: {config['activation']}")
    
    print(f"\n🎓 Standard PINN Training:")
    print(f"  Epochs:          {config['pinn_epochs']:,}")
    print(f"  Learning Rate:   {config['pinn_learning_rate']}")
    print(f"  Optimizer:       {config['pinn_optimizer']}")
    print(f"  LR Scheduler:    {'Yes' if config['pinn_use_scheduler'] else 'No'}")
    
    print(f"\n📚 Curriculum Learning:")
    print(f"  Total Epochs:    {config['curriculum_total_epochs']:,}")
    print(f"  Stages:")
    for i, stage in enumerate(config['curriculum_stages'], 1):
        print(f"    Stage {i}: Re={stage['Re']:3d}, Epochs={stage['epochs']:,}")
    
    print(f"\n📍 Collocation Points:")
    print(f"  Inlet BC:        {config['N_bc_inlet']:,}")
    print(f"  Outlet BC:       {config['N_bc_outlet']:,}")
    print(f"  Wall BC:         {config['N_bc_wall']:,}")
    print(f"  Cylinder BC:     {config['N_bc_cylinder']:,}")
    print(f"  PDE Interior:    {config['N_pde']:,}")
    total_pts = (config['N_pde'] + config['N_bc_inlet'] + 
                 config['N_bc_outlet'] + config['N_bc_wall'] + 
                 config['N_bc_cylinder'])
    print(f"  Total:           {total_pts:,}")
    
    print(f"\n📊 Data Loss:")
    print(f"  Enabled:         {'Yes' if config['use_data_loss'] else 'No'}")
    if config['use_data_loss']:
        print(f"  Data Points:     {config['N_data']:,}")
        print(f"  Loss Weight:     {config['lambda_data']}")
    
    print(f"\n🔢 FDM Solver (Ground Truth):")
    print(f"  Grid Size:       {config['fdm_nx']} × {config['fdm_ny']} "
          f"= {config['fdm_nx']*config['fdm_ny']:,} points")
    
    Lx = config['x_domain'][1] - config['x_domain'][0]
    Ly = config['y_domain'][1] - config['y_domain'][0]
    dx = Lx / (config['fdm_nx'] - 1)
    dy = Ly / (config['fdm_ny'] - 1)
    print(f"  Grid Spacing:    dx={dx:.6f} m, dy={dy:.6f} m")
    print(f"  Time Step (dt):  {config['fdm_dt']:.6f} s")
    print(f"  Max Iterations:  {config['fdm_max_iter']:,}")
    
    total_time = config['fdm_dt'] * config['fdm_max_iter']
    print(f"  Max Sim Time:    {total_time:.2f} s")
    print(f"  Tolerance:       {config['fdm_tolerance']:.2e}")
    
    # CFL analysis
    print(f"\n⚡ Stability Analysis (CFL):")
    cfl_adv = config['U_inf'] * config['fdm_dt'] / min(dx, dy)
    cfl_diff = nu * config['fdm_dt'] * (1/dx**2 + 1/dy**2)
    
    status_adv = "✓" if cfl_adv <= 1.0 else "✗"
    status_diff = "✓" if cfl_diff <= 0.5 else "✗"
    
    print(f"  Advective CFL:   {cfl_adv:.6f} {status_adv} [Limit: ≤ 1.0]")
    print(f"  Diffusive CFL:   {cfl_diff:.6f} {status_diff} [Limit: ≤ 0.5]")
    
    # Time scales
    print(f"\n⏱️  Characteristic Time Scales:")
    D = 2 * config['cylinder_radius']
    T_conv = D / config['U_inf']
    T_diff = D**2 / nu
    print(f"  Convective (D/U∞):  {T_conv:.4f} s")
    print(f"  Diffusive (D²/ν):   {T_diff:.4f} s")
    
    if Re > 47:  # Vortex shedding regime
        St = 0.198 * (1 - 19.7/Re) if Re < 200 else 0.2
        T_shed = D / (St * config['U_inf'])
        f_shed = 1 / T_shed
        print(f"  Vortex Shedding:")
        print(f"    Strouhal (St):    ~{St:.3f}")
        print(f"    Period (T):       ~{T_shed:.4f} s")
        print(f"    Frequency (f):    ~{f_shed:.2f} Hz")
        print(f"    Cycles in sim:    ~{total_time/T_shed:.1f} cycles")
    
    print("\n" + "="*80 + "\n")


def print_all_configs():
    """Print summary of all available configurations"""
    print("\n" + "="*80)
    print("AVAILABLE CONFIGURATIONS FOR CYLINDER FLOW")
    print("="*80)
    
    print("\n{:<15} {:<10} {:<50}".format("Config", "Re", "Description"))
    print("-" * 80)
    
    configs_info = [
        ("low_re", 40, "Steady symmetric flow with recirculation"),
        ("medium_re", 100, "Onset of Kármán vortex street (laminar)"),
        ("high_re", 200, "Strong vortex shedding (transitional)"),
    ]
    
    for name, re, desc in configs_info:
        print(f"{name:<15} {re:<10} {desc:<50}")
    
    print("\n" + "="*80)
    print("💡 Quick Start:")
    print("="*80)
    print("  # Get configuration")
    print("  config = get_config('medium_re')  # Default: Re=100")
    print()
    print("  # Show detailed info")
    print("  print_config_info('low_re')       # Re=40")
    print("  print_config_info('medium_re')    # Re=100")
    print("  print_config_info('high_re')      # Re=200")
    print()
    print("  # Run experiment")
    print("  python experiments/experiment_cylinder.py --config low_re")
    print("  python experiments/experiment_cylinder.py --config medium_re")
    print("  python experiments/experiment_cylinder.py --config high_re")
    print("="*80)
    
    print("\n📊 Default Settings (All Configs):")
    print("  ✓ Data loss:        ENABLED")
    print("  ✓ Initial condition: Potential flow solution")
    print("  ✓ Boundary conditions: Unified across FDM and PINN")
    print("  ✓ Curriculum learning: Gradual Re ramping")
    print("="*80 + "\n")


if __name__ == "__main__":
    print_all_configs()
    
    print("\n" + "="*80)
    print("DETAILED CONFIGURATION INFORMATION")
    print("="*80)
    
    for config_name in ['low_re', 'medium_re', 'high_re']:
        print_config_info(config_name)
        print()