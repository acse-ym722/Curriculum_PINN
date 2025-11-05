"""
Main experiment script for lid-driven cavity flow
"""
import sys
import os
import torch
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.neural_networks import create_model
from src.solvers.standard_pinn import NavierStokesPINN
from src.solvers.curriculum_pinn import CurriculumNavierStokesPINN
from src.numerics.fdm_cavity import CavityFDM
from src.visualization.cavity_plots import create_all_cavity_plots
from src.configs.cavity_config import get_config


def run_cavity_experiment(config_name='default', output_dir='outputs/cavity'):
    """
    Run complete cavity flow experiment
    
    Args:
        config_name: Name of configuration to use
        output_dir: Directory to save outputs
    """
    # Load configuration
    config = get_config(config_name)
    
    print("\n" + "="*80)
    print(f"LID-DRIVEN CAVITY FLOW EXPERIMENT: {config_name}")
    print("="*80)
    print("\nConfiguration:")
    print(json.dumps(config, indent=2))
    print("="*80 + "\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # ==================== STEP 1: Generate Ground Truth ====================
    print("="*80)
    print("STEP 1: Generating Ground Truth with FDM")
    print("="*80)
    
    fdm_solver = CavityFDM(
        Re=config['Re_target'],
        nx=config['fdm_grid_size'],
        ny=config['fdm_grid_size'],
        dt=config['fdm_dt'],
        max_iter=config['fdm_max_iter'],
        tol=config['fdm_tolerance']
    )
    fdm_solver.solve()
    
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
    
    # Create solver
    standard_config = {
        'Re': config['Re_target'],
        'x_domain': config['x_domain'],
        'y_domain': config['y_domain'],
        'N_bc': config['N_bc'],
        'N_pde': config['N_pde'],
        'lambda_bc': config['lambda_bc'],
        'lambda_pde': config['lambda_pde'],
        'lambda_cont': config['lambda_cont'],
        'epochs': config['pinn_epochs'],
        'learning_rate': config['pinn_learning_rate'],
        'optimizer': config.get('pinn_optimizer', 'adam'),
        'use_scheduler': config.get('pinn_use_scheduler', True),
        'lr_decay_steps': config.get('pinn_lr_decay_steps', 5000),
        'lr_decay_rate': config.get('pinn_lr_decay_rate', 0.9),
    }
    
    pinn_standard = NavierStokesPINN(model_std, standard_config, device=device)
    pinn_standard.train(verbose=True, save_interval=1000)
    
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
    
    curriculum_config = {
        'x_domain': config['x_domain'],
        'y_domain': config['y_domain'],
        'N_bc': config['N_bc'],
        'N_pde': config['N_pde'],
        'lambda_bc': config['lambda_bc'],
        'lambda_pde': config['lambda_pde'],
        'lambda_cont': config['lambda_cont'],
        'curriculum_stages': config['curriculum_stages'],
        'learning_rate': config['curriculum_learning_rate'],
        'optimizer': config.get('curriculum_optimizer', 'adam'),
        'use_scheduler': config.get('curriculum_use_scheduler', True),
        'lr_decay_steps': config.get('curriculum_lr_decay_steps', 5000),
        'lr_decay_rate': config.get('curriculum_lr_decay_rate', 0.9),
    }
    
    pinn_curriculum = CurriculumNavierStokesPINN(model_curr, curriculum_config, device=device)
    pinn_curriculum.train(verbose=True, save_interval=500)
    
    # ==================== STEP 4: Visualization ====================
    print("="*80)
    print("STEP 4: Generating Visualizations")
    print("="*80)
    
    create_all_cavity_plots(fdm_solver, pinn_standard, pinn_curriculum, 
                           config, output_dir=output_dir)
    
    # ==================== STEP 5: Save Models ====================
    os.makedirs(output_dir, exist_ok=True)
    
    pinn_standard.save_model(os.path.join(output_dir, 'standard_pinn.pt'))
    pinn_curriculum.save_model(os.path.join(output_dir, 'curriculum_pinn.pt'))
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run cavity flow experiment')
    parser.add_argument('--config', type=str, default='default',
                       help='Configuration name (default, low_re, high_re, deep, resnet)')
    parser.add_argument('--output', type=str, default='outputs/cavity',
                       help='Output directory')
    
    args = parser.parse_args()
    
    run_cavity_experiment(config_name=args.config, output_dir=args.output)