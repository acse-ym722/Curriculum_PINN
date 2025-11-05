"""
Configuration for 2D flow past a cylinder
"""

# Default configuration for cylinder flow
CYLINDER_CONFIG = {
    # Physical parameters
    "Re_target": 100,
    "x_domain": [-5.0, 15.0],
    "y_domain": [-5.0, 5.0],
    
    # Cylinder geometry
    "cylinder_center": [0.0, 0.0],
    "cylinder_radius": 0.5,
    
    # Inlet conditions
    "U_inf": 1.0,
    
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
    
    # Curriculum PINN
    "curriculum_total_epochs": 60000,
    "curriculum_learning_rate": 1e-3,
    "curriculum_optimizer": "adam",
    "curriculum_use_scheduler": True,
    "curriculum_lr_decay_steps": 5000,
    "curriculum_lr_decay_rate": 0.95,
    
    # Curriculum stages
    "curriculum_stages": [
        {"Re": 10, "epochs": 5000},
        {"Re": 40, "epochs": 5000},
        {"Re": 100, "epochs": 50000},
    ],
    
    # Data points
    "N_bc_inlet": 500,
    "N_bc_outlet": 500,
    "N_bc_wall": 200,
    "N_bc_cylinder": 800,
    "N_pde": 20000,
    
    # Data loss parameters
    "use_data_loss": False,
    "N_data": 1000,
    "fdm_solver": None,
    
    # Loss weights
    "lambda_bc_inlet": 10.0,
    "lambda_bc_outlet": 5.0,
    "lambda_bc_wall": 10.0,
    "lambda_bc_cylinder": 20.0,
    "lambda_pde": 1.0,
    "lambda_cont": 1.0,
    "lambda_data": 10.0,
    
    # Numerical solution (FDM)
    "fdm_nx": 200,
    "fdm_ny": 100,
    "fdm_max_iter": 20000,
    "fdm_tolerance": 1e-6,
    "fdm_dt": 0.001,
    
    # Visualization
    "plot_resolution": 150,
}


# With data loss
CYLINDER_WITH_DATA = {
    **CYLINDER_CONFIG,
    "use_data_loss": True,
    "N_data": 2000,
    "lambda_data": 15.0,
    "fdm_max_iter": 25000,
}


# Very low Reynolds number (Re=20) - Steady symmetric flow
CYLINDER_VERY_LOW_RE = {
    **CYLINDER_CONFIG,
    "Re_target": 20,
    "pinn_epochs": 40000,
    "curriculum_total_epochs": 40000,
    "curriculum_stages": [
        {"Re": 5, "epochs": 5000},
        {"Re": 10, "epochs": 5000},
        {"Re": 20, "epochs": 30000},
    ],
    "fdm_nx": 200,
    "fdm_ny": 100,
    "fdm_max_iter": 10000,
    "fdm_tolerance": 1e-6,
    "fdm_dt": 0.002,  # Larger time step for stable flow
    # Data loss parameters
    "use_data_loss": False,
    "N_data": 1000,
    "lambda_data": 10.0,
}


# Low Reynolds number (Re=40) - Steady with recirculation
CYLINDER_LOW_RE = {
    **CYLINDER_CONFIG,
    "Re_target": 40,
    "pinn_epochs": 50000,
    "curriculum_total_epochs": 50000,
    "curriculum_stages": [
        {"Re": 10, "epochs": 5000},
        {"Re": 20, "epochs": 5000},
        {"Re": 40, "epochs": 40000},
    ],
    "fdm_nx": 200,
    "fdm_ny": 100,
    "fdm_max_iter": 15000,
    "fdm_tolerance": 1e-6,
    "fdm_dt": 0.001,
    # Data loss parameters
    "use_data_loss": False,
    "N_data": 1000,
    "lambda_data": 10.0,
}


# Lightweight Re=40 - Quick convergence (RECOMMENDED for Re=40)
CYLINDER_LOW_RE_LIGHT = {
    **CYLINDER_CONFIG,
    "Re_target": 40,
    
    # Reduced PINN training
    "pinn_epochs": 30000,
    "curriculum_total_epochs": 30000,
    "curriculum_stages": [
        {"Re": 10, "epochs": 3000},
        {"Re": 20, "epochs": 3000},
        {"Re": 40, "epochs": 24000},
    ],
    
    # Reduced collocation points
    "N_bc_inlet": 300,
    "N_bc_outlet": 300,
    "N_bc_wall": 150,
    "N_bc_cylinder": 500,
    "N_pde": 12000,
    
    # Lightweight FDM - coarser grid and fewer iterations
    "fdm_nx": 120,
    "fdm_ny": 60,
    "fdm_max_iter": 5000,    # Only 5 seconds simulation time
    "fdm_tolerance": 1e-5,    # Relaxed tolerance
    "fdm_dt": 0.001,
    
    # Data loss parameters
    "use_data_loss": False,
    "N_data": 800,
    "lambda_data": 10.0,
    
    # Visualization
    "plot_resolution": 100,
}


# Ultra-lightweight Re=40 - Very quick testing
CYLINDER_LOW_RE_ULTRA_LIGHT = {
    **CYLINDER_CONFIG,
    "Re_target": 40,
    
    # Minimal PINN training
    "pinn_epochs": 15000,
    "curriculum_total_epochs": 15000,
    "curriculum_stages": [
        {"Re": 10, "epochs": 2000},
        {"Re": 40, "epochs": 13000},
    ],
    
    # Minimal collocation points
    "N_bc_inlet": 200,
    "N_bc_outlet": 200,
    "N_bc_wall": 100,
    "N_bc_cylinder": 400,
    "N_pde": 8000,
    
    # Ultra-lightweight FDM
    "fdm_nx": 80,
    "fdm_ny": 40,
    "fdm_max_iter": 3000,     # Only 6 seconds simulation time
    "fdm_tolerance": 5e-5,     # More relaxed tolerance
    "fdm_dt": 0.002,           # Larger time step (still stable for Re=40)
    
    # Data loss parameters
    "use_data_loss": False,
    "N_data": 500,
    "lambda_data": 10.0,
    
    # Visualization
    "plot_resolution": 80,
}


# Adaptive Re=40 - Balanced performance
CYLINDER_LOW_RE_ADAPTIVE = {
    **CYLINDER_CONFIG,
    "Re_target": 40,
    
    # Moderate PINN training
    "pinn_epochs": 25000,
    "curriculum_total_epochs": 25000,
    "curriculum_stages": [
        {"Re": 15, "epochs": 3000},
        {"Re": 40, "epochs": 22000},
    ],
    
    # Moderate collocation points
    "N_bc_inlet": 250,
    "N_bc_outlet": 250,
    "N_bc_wall": 120,
    "N_bc_cylinder": 450,
    "N_pde": 10000,
    
    # Adaptive FDM - balanced
    "fdm_nx": 100,
    "fdm_ny": 50,
    "fdm_max_iter": 4000,
    "fdm_tolerance": 2e-5,
    "fdm_dt": 0.0015,
    
    # Data loss parameters
    "use_data_loss": False,
    "N_data": 600,
    "lambda_data": 10.0,
    
    # Visualization
    "plot_resolution": 90,
}


# Medium Reynolds number (Re=100) - Onset of vortex shedding
CYLINDER_MEDIUM_RE = {
    **CYLINDER_CONFIG,
    "Re_target": 100,
    "pinn_epochs": 60000,
    "curriculum_total_epochs": 60000,
    "curriculum_stages": [
        {"Re": 10, "epochs": 5000},
        {"Re": 40, "epochs": 5000},
        {"Re": 100, "epochs": 50000},
    ],
    "fdm_nx": 250,
    "fdm_ny": 120,
    "fdm_max_iter": 30000,
    "fdm_tolerance": 1e-6,
    "fdm_dt": 0.0005,  # Smaller time step for unsteady flow
    # Data loss parameters
    "use_data_loss": False,
    "N_data": 1200,
    "lambda_data": 10.0,
}


# High Reynolds number (Re=200) - Strong vortex shedding
CYLINDER_HIGH_RE = {
    **CYLINDER_CONFIG,
    "Re_target": 200,
    "pinn_epochs": 80000,
    "curriculum_total_epochs": 80000,
    "curriculum_stages": [
        {"Re": 20, "epochs": 6000},
        {"Re": 60, "epochs": 6000},
        {"Re": 100, "epochs": 8000},
        {"Re": 200, "epochs": 60000},
    ],
    "N_pde": 30000,
    "N_bc_cylinder": 1000,
    "fdm_nx": 300,
    "fdm_ny": 150,
    "fdm_max_iter": 50000,
    "fdm_tolerance": 5e-6,  # Relaxed tolerance for unsteady flow
    "fdm_dt": 0.0003,
    # Data loss parameters
    "use_data_loss": False,
    "N_data": 1500,
    "lambda_data": 10.0,
}


# Very high Reynolds number (Re=500) - Turbulent wake
CYLINDER_VERY_HIGH_RE = {
    **CYLINDER_CONFIG,
    "Re_target": 500,
    "pinn_epochs": 100000,
    "curriculum_total_epochs": 100000,
    "curriculum_stages": [
        {"Re": 40, "epochs": 8000},
        {"Re": 100, "epochs": 10000},
        {"Re": 200, "epochs": 12000},
        {"Re": 500, "epochs": 70000},
    ],
    "layers": [2, 256, 512, 512, 512, 256, 3],  # Deeper network
    "N_pde": 40000,
    "N_bc_cylinder": 1200,
    "fdm_nx": 400,
    "fdm_ny": 200,
    "fdm_max_iter": 100000,
    "fdm_tolerance": 1e-5,
    "fdm_dt": 0.0001,  # Very small time step
    # Data loss parameters
    "use_data_loss": False,
    "N_data": 2000,
    "lambda_data": 10.0,
}


# Fine grid configuration for high accuracy
CYLINDER_FINE_GRID = {
    **CYLINDER_CONFIG,
    "Re_target": 100,
    "fdm_nx": 400,
    "fdm_ny": 200,
    "fdm_max_iter": 50000,
    "fdm_tolerance": 1e-7,
    "fdm_dt": 0.0002,
    "N_pde": 30000,
    "plot_resolution": 200,
    # Data loss parameters
    "use_data_loss": False,
    "N_data": 1500,
    "lambda_data": 10.0,
}


# Coarse grid for quick testing
CYLINDER_COARSE_GRID = {
    **CYLINDER_CONFIG,
    "Re_target": 100,
    "pinn_epochs": 30000,
    "curriculum_total_epochs": 30000,
    "curriculum_stages": [
        {"Re": 20, "epochs": 3000},
        {"Re": 100, "epochs": 27000},
    ],
    "fdm_nx": 100,
    "fdm_ny": 50,
    "fdm_max_iter": 5000,
    "fdm_tolerance": 1e-5,
    "fdm_dt": 0.002,
    "N_pde": 10000,
    "plot_resolution": 100,
    # Data loss parameters
    "use_data_loss": False,
    "N_data": 500,
    "lambda_data": 10.0,
}


# ResNet architecture for cylinder
CYLINDER_RESNET = {
    **CYLINDER_CONFIG,
    "model_type": "resnet",
    "input_dim": 2,
    "hidden_dim": 256,
    "output_dim": 3,
    "num_blocks": 8,
    "fdm_max_iter": 25000,
    # Data loss parameters
    "use_data_loss": False,
    "N_data": 1000,
    "lambda_data": 10.0,
}


# With data loss and fine grid
CYLINDER_WITH_DATA_FINE = {
    **CYLINDER_CONFIG,
    "use_data_loss": True,
    "N_data": 3000,
    "lambda_data": 15.0,
    "fdm_nx": 300,
    "fdm_ny": 150,
    "fdm_max_iter": 40000,
    "fdm_tolerance": 1e-6,
    "fdm_dt": 0.0005,
}


# Low Re with data loss
CYLINDER_LOW_RE_DATA = {
    **CYLINDER_LOW_RE,
    "use_data_loss": True,
    "N_data": 1500,
    "lambda_data": 15.0,
    "fdm_max_iter": 20000,
}


# High Re with data loss
CYLINDER_HIGH_RE_DATA = {
    **CYLINDER_HIGH_RE,
    "use_data_loss": True,
    "N_data": 2000,
    "lambda_data": 15.0,
    "fdm_max_iter": 60000,
}


def get_config(config_name='default'):
    """Get configuration by name"""
    configs = {
        'default': CYLINDER_CONFIG,
        'with_data': CYLINDER_WITH_DATA,
        'very_low_re': CYLINDER_VERY_LOW_RE,
        'low_re': CYLINDER_LOW_RE,
        'low_re_light': CYLINDER_LOW_RE_LIGHT,           # ⚡ Recommended for Re=40
        'low_re_ultra_light': CYLINDER_LOW_RE_ULTRA_LIGHT, # ⚡⚡ Fastest for Re=40
        'low_re_adaptive': CYLINDER_LOW_RE_ADAPTIVE,     # ⚡ Balanced for Re=40
        'medium_re': CYLINDER_MEDIUM_RE,
        'high_re': CYLINDER_HIGH_RE,
        'very_high_re': CYLINDER_VERY_HIGH_RE,
        'fine_grid': CYLINDER_FINE_GRID,
        'coarse_grid': CYLINDER_COARSE_GRID,
        'resnet': CYLINDER_RESNET,
        'with_data_fine': CYLINDER_WITH_DATA_FINE,
        'low_re_data': CYLINDER_LOW_RE_DATA,
        'high_re_data': CYLINDER_HIGH_RE_DATA,
    }
    
    if config_name not in configs:
        print(f"Warning: Unknown config '{config_name}', using default")
        return CYLINDER_CONFIG.copy()
    
    return configs[config_name].copy()


def print_config_info(config_name='default'):
    """Print detailed information about a configuration"""
    config = get_config(config_name)
    
    print("\n" + "="*80)
    print(f"CONFIGURATION: {config_name.upper()}")
    print("="*80)
    
    print("\nPhysical Parameters:")
    print(f"  Reynolds Number:              {config['Re_target']}")
    print(f"  Free Stream Velocity:         {config['U_inf']} m/s")
    print(f"  Cylinder Diameter:            {2*config['cylinder_radius']} m")
    print(f"  Kinematic Viscosity:          {config['U_inf']*2*config['cylinder_radius']/config['Re_target']:.6e} m²/s")
    
    print("\nDomain:")
    print(f"  x: [{config['x_domain'][0]}, {config['x_domain'][1]}]")
    print(f"  y: [{config['y_domain'][0]}, {config['y_domain'][1]}]")
    
    print("\nNeural Network:")
    print(f"  Architecture Type:            {config['model_type'].upper()}")
    if config['model_type'] == 'mlp':
        print(f"  Layers:                       {config['layers']}")
        print(f"  Activation:                   {config['activation']}")
    else:
        print(f"  Hidden Dimension:             {config.get('hidden_dim', 'N/A')}")
        print(f"  Number of Blocks:             {config.get('num_blocks', 'N/A')}")
    
    print("\nPINN Training:")
    print(f"  Epochs:                       {config['pinn_epochs']:,}")
    print(f"  Learning Rate:                {config['pinn_learning_rate']}")
    print(f"  Optimizer:                    {config['pinn_optimizer']}")
    
    print("\nCurriculum Learning:")
    print(f"  Total Epochs:                 {config['curriculum_total_epochs']:,}")
    print(f"  Stages:")
    for i, stage in enumerate(config['curriculum_stages'], 1):
        print(f"    Stage {i}: Re={stage['Re']:3d}, Epochs={stage['epochs']:,}")
    
    print("\nCollocation Points:")
    print(f"  PDE Interior:                 {config['N_pde']:,}")
    print(f"  Inlet BC:                     {config['N_bc_inlet']:,}")
    print(f"  Outlet BC:                    {config['N_bc_outlet']:,}")
    print(f"  Wall BC:                      {config['N_bc_wall']:,}")
    print(f"  Cylinder BC:                  {config['N_bc_cylinder']:,}")
    total_pts = config['N_pde'] + config['N_bc_inlet'] + config['N_bc_outlet'] + config['N_bc_wall'] + config['N_bc_cylinder']
    print(f"  Total:                        {total_pts:,}")
    
    print("\nFDM Solver:")
    print(f"  Grid Size:                    {config['fdm_nx']} × {config['fdm_ny']} = {config['fdm_nx']*config['fdm_ny']:,} points")
    Lx = config['x_domain'][1] - config['x_domain'][0]
    Ly = config['y_domain'][1] - config['y_domain'][0]
    dx = Lx / (config['fdm_nx'] - 1)
    dy = Ly / (config['fdm_ny'] - 1)
    print(f"  Grid Spacing:                 dx={dx:.6f}, dy={dy:.6f}")
    print(f"  Time Step:                    {config['fdm_dt']:.6f} s")
    print(f"  Max Iterations:               {config['fdm_max_iter']:,}")
    total_time = config['fdm_dt'] * config['fdm_max_iter']
    print(f"  Max Simulation Time:          {total_time:.4f} s")
    print(f"  Convergence Tolerance:        {config['fdm_tolerance']:.2e}")
    
    # CFL numbers
    nu = config['U_inf'] * 2 * config['cylinder_radius'] / config['Re_target']
    cfl_adv = config['U_inf'] * config['fdm_dt'] / min(dx, dy)
    cfl_diff = nu * config['fdm_dt'] * (1/dx**2 + 1/dy**2)
    print(f"\n  CFL (Advective):              {cfl_adv:.6f} {'✓' if cfl_adv <= 1.0 else '⚠'}")
    print(f"  CFL (Diffusive):              {cfl_diff:.6f} {'✓' if cfl_diff <= 0.5 else '⚠'}")
    
    # Performance estimate
    print("\nPerformance Estimate:")
    print(f"  FDM Iterations:               ~{config['fdm_max_iter']:,} iterations")
    print(f"  Estimated FDM Time:           ~{total_time:.1f} seconds physical time")
    if config_name in ['low_re_ultra_light', 'low_re_light', 'low_re_adaptive']:
        print(f"  ⚡ Lightweight configuration - faster convergence expected")
    
    print("\nData Loss:")
    print(f"  Use Data Loss:                {config['use_data_loss']}")
    if config['use_data_loss']:
        print(f"  Number of Data Points:        {config['N_data']:,}")
        print(f"  Data Loss Weight:             {config['lambda_data']}")
    
    print("\n" + "="*80 + "\n")


def print_all_configs():
    """Print summary of all available configurations"""
    print("\n" + "="*80)
    print("AVAILABLE CONFIGURATIONS")
    print("="*80)
    
    configs_info = [
        ("default", "Re=100", "Default configuration", ""),
        ("very_low_re", "Re=20", "Steady symmetric flow", ""),
        ("low_re", "Re=40", "Steady with recirculation", ""),
        ("low_re_light", "Re=40", "⚡ Lightweight (RECOMMENDED)", "5s sim"),
        ("low_re_ultra_light", "Re=40", "⚡⚡ Ultra-fast testing", "6s sim"),
        ("low_re_adaptive", "Re=40", "⚡ Balanced performance", "6s sim"),
        ("medium_re", "Re=100", "Onset of vortex shedding", ""),
        ("high_re", "Re=200", "Strong vortex shedding", ""),
        ("very_high_re", "Re=500", "Turbulent wake (2D limit)", ""),
        ("fine_grid", "Re=100", "High accuracy (fine mesh)", ""),
        ("coarse_grid", "Re=100", "Quick testing (coarse)", ""),
        ("with_data", "Re=100", "With data loss", ""),
        ("with_data_fine", "Re=100", "Data loss + fine grid", ""),
        ("low_re_data", "Re=40", "Low Re + data loss", ""),
        ("high_re_data", "Re=200", "High Re + data loss", ""),
        ("resnet", "Re=100", "ResNet architecture", ""),
    ]
    
    print("\n{:<22} {:<10} {:<35} {:<15}".format("Config Name", "Re", "Description", "FDM Grid"))
    print("-" * 80)
    
    for name, re_info, description, note in configs_info:
        config = get_config(name)
        total_time = config['fdm_dt'] * config['fdm_max_iter']
        grid_info = f"{config['fdm_nx']}×{config['fdm_ny']}"
        
        display_desc = description
        if note:
            display_desc += f" ({note})"
        
        print(f"{name:<22} {re_info:<10} {display_desc:<35} {grid_info:<15}")
    
    print("\n" + "="*80)
    print("Usage Examples:")
    print("  config = get_config('low_re_light')      # For Re=40 (recommended)")
    print("  config = get_config('low_re_ultra_light') # For Re=40 (fastest)")
    print("  print_config_info('low_re_light')        # Show detailed info")
    print("="*80 + "\n")


if __name__ == "__main__":
    print_all_configs()
    print("\nDetailed info for recommended Re=40 configuration:")
    print_config_info('low_re_light')