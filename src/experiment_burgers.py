"""
Main experiment script for Burgers equation
"""
import sys
import os
import torch
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.neural_networks import create_model
from src.solvers.standard_pinn import BurgersPINN
from src.solvers.curriculum_pinn import CurriculumBurgersPINN
from src.numerics.fdm_burgers import BurgersFDM
from src.visualization.burgers_plots import create_all_burgers_plots
from src.configs.burgers_config import get_config


def check_fdm_stability(config):
    """
    Check FDM stability before running expensive computations
    
    Args:
        config: Configuration dictionary
    
    Returns:
        bool: Whether parameters are stable
    """
    print("\n" + "="*80)
    print("PRE-FLIGHT CHECK: FDM Stability Analysis")
    print("="*80)
    
    nx = config["fdm_nx"]
    nt = config["fdm_nt"]
    x_domain = config["x_domain"]
    t_domain = config["t_domain"]
    nu = config["nu"]
    
    dx = (x_domain[1] - x_domain[0]) / (nx - 1)
    dt = (t_domain[1] - t_domain[0]) / (nt - 1)
    
    u_max_est = 1.0
    cfl_adv = u_max_est * dt / dx
    cfl_diff = 2 * nu * dt / (dx**2)
    
    print(f"  dx = {dx:.6f}, dt = {dt:.6f}")
    print(f"  Advective CFL = {cfl_adv:.4f} (must be <= 1.0)")
    print(f"  Diffusive CFL = {cfl_diff:.4f} (must be <= 0.5)")
    
    is_stable = cfl_adv <= 1.0 and cfl_diff <= 0.5
    
    if is_stable:
        print("  ✓ FDM parameters are STABLE")
    else:
        print("  ✗ FDM parameters are UNSTABLE - consider adjusting fdm_nt or fdm_nx")
    
    print("="*80 + "\n")
    
    return is_stable


def run_burgers_experiment(config_name='default', output_dir='outputs/burgers', 
                          use_data_loss=False):
    """
    Run complete Burgers equation experiment
    
    Args:
        config_name: Name of configuration to use
        output_dir: Directory to save outputs
        use_data_loss: Whether to use data loss in training
    """
    # Load configuration
    config = get_config(config_name)
    
    # Override data loss setting if specified
    if use_data_loss:
        config['use_data_loss'] = True
        print(f"\n⚠ Data loss ENABLED with {config['N_data']} sampling points")
    
    print("\n" + "="*80)
    print(f"BURGERS EQUATION EXPERIMENT: {config_name}")
    print("="*80)
    print("\nConfiguration:")
    print(json.dumps({k: v for k, v in config.items() if k != 'fdm_solution'}, indent=2))
    print("="*80 + "\n")
    
    # Check stability
    if not check_fdm_stability(config):
        response = input("FDM may be unstable. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Experiment aborted.")
            return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # ==================== STEP 1: Generate Ground Truth ====================
    print("="*80)
    print("STEP 1: Generating Ground Truth with FDM")
    print("="*80)
    
    fdm_solver = BurgersFDM(
        nu=config['nu'],
        x_domain=config['x_domain'],
        t_domain=config['t_domain'],
        nx=config['fdm_nx'],
        nt=config['fdm_nt']
    )
    u_exact, x_grid, t_grid = fdm_solver.solve()
    
    # Store FDM solution in config for data loss
    fdm_solution = (u_exact, x_grid, t_grid)
    
    # ==================== STEP 2: Train Standard PINN ====================
    print("="*80)
    print("STEP 2: Training Standard PINN")
    print("="*80)
    
    # Create model
    if config['model_type'] == 'mlp':
        model_std = create_model(model_type='mlp', layers=config['layers'], 
                                activation=config['activation'])
    elif config['model_type'] == 'resnet':
        model_std = create_model(model_type='resnet', 
                                input_dim=config['input_dim'],
                                hidden_dim=config['hidden_dim'],
                                output_dim=config['output_dim'],
                                num_blocks=config.get('num_blocks', 4),
                                activation=config['activation'])
    elif config['model_type'] == 'fourier':
        model_std = create_model(model_type='fourier',
                                layers=config['layers'],
                                input_dim=config['input_dim'],
                                fourier_dim=config['fourier_dim'],
                                sigma=config.get('sigma', 1.0),
                                activation=config['activation'])
    
    # Create solver config (注入FDM solution)
    standard_config = {
        'nu': config['nu'],
        'x_domain': config['x_domain'],
        't_domain': config['t_domain'],
        'N_ic': config['N_ic'],
        'N_bc': config['N_bc'],
        'N_pde': config['N_pde'],
        'lambda_ic': config['lambda_ic'],
        'lambda_bc': config['lambda_bc'],
        'lambda_pde': config['lambda_pde'],
        'lambda_data': config['lambda_data'],
        'epochs': config['epochs'],
        'learning_rate': config['learning_rate'],
        'optimizer': config.get('optimizer', 'adam'),
        'use_scheduler': config.get('use_scheduler', True),
        'lr_decay_steps': config.get('lr_decay_steps', 5000),
        'lr_decay_rate': config.get('lr_decay_rate', 0.9),
        'use_data_loss': config['use_data_loss'],
        'N_data': config['N_data'],
        'fdm_solution': fdm_solution if config['use_data_loss'] else None,
    }
    
    pinn_standard = BurgersPINN(model_std, standard_config, device=device)
    pinn_standard.train(verbose=True, save_interval=2000)
    
    # ==================== STEP 3: Train Curriculum PINN ====================
    print("="*80)
    print("STEP 3: Training Curriculum PINN")
    print("="*80)
    
    # Create new model
    if config['model_type'] == 'mlp':
        model_curr = create_model(model_type='mlp', layers=config['layers'], 
                                 activation=config['activation'])
    elif config['model_type'] == 'resnet':
        model_curr = create_model(model_type='resnet', 
                                 input_dim=config['input_dim'],
                                 hidden_dim=config['hidden_dim'],
                                 output_dim=config['output_dim'],
                                 num_blocks=config.get('num_blocks', 4),
                                 activation=config['activation'])
    elif config['model_type'] == 'fourier':
        model_curr = create_model(model_type='fourier',
                                 layers=config['layers'],
                                 input_dim=config['input_dim'],
                                 fourier_dim=config['fourier_dim'],
                                 sigma=config.get('sigma', 1.0),
                                 activation=config['activation'])
    
    curriculum_config = {**standard_config, 
                        'curriculum_ramp_ratio': config['curriculum_ramp_ratio']}
    
    pinn_curriculum = CurriculumBurgersPINN(model_curr, curriculum_config, device=device)
    pinn_curriculum.train(verbose=True, save_interval=2000)
    
    # ==================== STEP 4: Visualization ====================
    print("="*80)
    print("STEP 4: Generating Visualizations")
    print("="*80)
    
    results = {
        'config': config,
        'standard': {
            'solver': pinn_standard,
            'label': 'Standard PINN' + (' (with data)' if config['use_data_loss'] else '')
        },
        'curriculum': {
            'solver': pinn_curriculum,
            'label': 'Curriculum PINN' + (' (with data)' if config['use_data_loss'] else '')
        },
        'exact': {
            'solution': u_exact,
            'x': x_grid,
            't': t_grid,
            'label': 'Numerical (FDM)'
        }
    }
    
    create_all_burgers_plots(results, output_dir=output_dir)
    
    # ==================== STEP 5: Save Models ====================
    os.makedirs(output_dir, exist_ok=True)
    
    pinn_standard.save_model(os.path.join(output_dir, 'standard_pinn.pt'))
    pinn_curriculum.save_model(os.path.join(output_dir, 'curriculum_pinn.pt'))
    
    # Save configuration (exclude fdm_solution)
    config_to_save = {k: v for k, v in config.items() if k != 'fdm_solution'}
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    if config['use_data_loss']:
        print(f"✓ Data loss was ENABLED with {config['N_data']} sampling points")
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Burgers equation experiment')
    parser.add_argument('--config', type=str, default='default',
                       help='Configuration name (default, with_data, high_nu, low_nu, resnet, fourier)')
    parser.add_argument('--output', type=str, default='outputs/burgers',
                       help='Output directory')
    parser.add_argument('--use-data-loss', action='store_true',
                       help='Enable data loss during training')
    
    args = parser.parse_args()
    
    run_burgers_experiment(
        config_name=args.config, 
        output_dir=args.output,
        use_data_loss=args.use_data_loss
    )