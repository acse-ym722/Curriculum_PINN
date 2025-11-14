"""
Main experiment script for 2D Darcy Flow equation
"""
import sys
import os
import torch
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.neural_networks import create_model
from src.solvers.standard_pinn import DarcyPINN
from src.solvers.curriculum_pinn import CurriculumDarcyPINN
from src.numerics.fdm_darcy import DarcyFDM
from src.visualization.darcy_plots import create_all_darcy_plots
from src.configs.darcy_config import get_config


def run_darcy_experiment(config_name='default', output_dir='outputs/darcy',
                        use_data_loss=False):
    """
    Run complete Darcy equation experiment
    
    Args:
        config_name: Name of configuration to use
        output_dir: Directory to save outputs
        use_data_loss: Whether to use data loss in training
    """
    # Load configuration
    config = get_config(config_name)
    
    # Override data loss setting
    if use_data_loss:
        config['use_data_loss'] = True
        print(f"\n⚠ Data loss ENABLED with {config['N_data']} sampling points")
    
    print("\n" + "="*80)
    print(f"2D DARCY FLOW EXPERIMENT: {config_name}")
    print("="*80)
    print("\nConfiguration:")
    config_display = {k: v for k, v in config.items() 
                     if k not in ['fdm_solution', 'curriculum_stages']}
    print(json.dumps(config_display, indent=2))
    print("="*80 + "\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # ==================== STEP 1: Generate Ground Truth ====================
    print("="*80)
    print("STEP 1: Generating Ground Truth with FDM")
    print("="*80)
    
    fdm_solver = DarcyFDM(
        K=config['K'],
        f_source=config['f_source'],
        x_domain=config['x_domain'],
        y_domain=config['y_domain'],
        nx=config['fdm_nx'],
        ny=config['fdm_ny'],
        bc_left=config['bc_left'],
        bc_right=config['bc_right']
    )
    p_exact, x_grid, y_grid = fdm_solver.solve()
    
    # Store FDM solution for data loss
    fdm_solution = (p_exact, x_grid, y_grid)
    
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
    
    # Create solver config
    standard_config = {
        'K': config['K'],
        'f_source': config['f_source'],
        'x_domain': config['x_domain'],
        'y_domain': config['y_domain'],
        'bc_left': config['bc_left'],
        'bc_right': config['bc_right'],
        'N_collocation': config['N_collocation'],
        'N_bc': config['N_bc'],
        'lambda_pde': config['lambda_pde'],
        'lambda_bc_dirichlet': config['lambda_bc_dirichlet'],
        'lambda_bc_neumann': config['lambda_bc_neumann'],
        'lambda_data': config['lambda_data'],
        'epochs': config['epochs'],
        'learning_rate': config['learning_rate'],
        'optimizer': config.get('optimizer', 'adam'),
        'use_scheduler': config.get('use_scheduler', True),
        'lr_decay_steps': config.get('lr_decay_steps', 2000),
        'lr_decay_rate': config.get('lr_decay_rate', 0.9),
        'use_data_loss': config['use_data_loss'],
        'N_data': config['N_data'],
        'fdm_solution': fdm_solution if config['use_data_loss'] else None,
    }
    
    pinn_standard = DarcyPINN(model_std, standard_config, device=device)
    pinn_standard.train(verbose=True, save_interval=500)
    
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
    
    curriculum_config = {
        **standard_config,
        'curriculum_stages': config['curriculum_stages']
    }
    
    pinn_curriculum = CurriculumDarcyPINN(model_curr, curriculum_config, device=device)
    pinn_curriculum.train(verbose=True, save_interval=200)
    
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
            'p': p_exact,
            'x': x_grid,
            'y': y_grid,
            'label': 'Numerical (FDM)'
        }
    }
    
    create_all_darcy_plots(results, output_dir=output_dir)
    
    # ==================== STEP 5: Save Models ====================
    os.makedirs(output_dir, exist_ok=True)
    
    pinn_standard.save_model(os.path.join(output_dir, 'standard_pinn.pt'))
    pinn_curriculum.save_model(os.path.join(output_dir, 'curriculum_pinn.pt'))
    
    # Save configuration
    config_to_save = {k: v for k, v in config.items() 
                     if k not in ['fdm_solution', 'curriculum_stages']}
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
    
    parser = argparse.ArgumentParser(description='Run Darcy equation experiment')
    parser.add_argument('--config', type=str, default='default',
                       help='Configuration name (default, with_data, resnet, fourier)')
    parser.add_argument('--output', type=str, default='outputs/darcy',
                       help='Output directory')
    parser.add_argument('--use-data-loss', action='store_true',
                       help='Enable data loss during training')
    
    args = parser.parse_args()
    
    run_darcy_experiment(
        config_name=args.config,
        output_dir=args.output,
        use_data_loss=args.use_data_loss
    )