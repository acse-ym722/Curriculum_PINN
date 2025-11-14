# 文件名: experiment_wave.py

"""
1D Wave Equation Experiment with Loss Landscape Visualization

This experiment compares:
1. Standard PINN training
2. Curriculum learning PINN
3. Ground truth from FDM solver

Features:
- Automatic FDM stability checking
- Loss landscape visualization
- Training trajectory tracking
- Comprehensive plots and metrics
"""

import sys
import os
import torch
import json
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.neural_networks import create_model
from src.solvers.standard_pinn import WavePINN
from src.solvers.curriculum_pinn import CurriculumWavePINN
from src.numerics.fdm_wave import WaveFDM
from src.visualization.wave_plots import create_all_wave_plots
from src.visualization.loss_landscape_generic import LossLandscapeVisualizer
from src.configs.wave_config import get_config


def check_fdm_stability(config):
    """Check FDM stability before running"""
    print("\n" + "="*80)
    print("PRE-FLIGHT CHECK: FDM Stability Analysis")
    print("="*80)
    
    nx = config["fdm_nx"]
    nt = config["fdm_nt"]
    x_domain = config["x_domain"]
    t_domain = config["t_domain"]
    c = config["c"]
    
    dx = (x_domain[1] - x_domain[0]) / (nx - 1)
    dt = (t_domain[1] - t_domain[0]) / (nt - 1)
    
    cfl = c * dt / dx
    
    print(f"  dx = {dx:.6f}, dt = {dt:.6f}")
    print(f"  Wave speed c = {c:.2f}")
    print(f"  CFL = c*dt/dx = {cfl:.4f} (must be <= 1.0)")
    
    is_stable = cfl <= 1.0
    
    if is_stable:
        print("  ✓ FDM parameters are STABLE")
    else:
        print("  ✗ FDM parameters are UNSTABLE")
        print(f"  Suggested: increase nt to at least {int(c * (t_domain[1] - t_domain[0]) / dx) + 1}")
    
    print("="*80 + "\n")
    
    return is_stable


def prepare_evaluation_data(config, device, n_samples=2000):
    """
    Prepare evaluation data for loss landscape computation.
    
    Args:
        config: Experiment configuration
        device: torch device
        n_samples: Number of evaluation samples
    
    Returns:
        X_eval: Input tensor (x, t)
        u_target: Target tensor (dummy, will use PDE loss)
    """
    print(f"  Preparing {n_samples} evaluation points...")
    
    x_domain = config['x_domain']
    t_domain = config['t_domain']
    
    # Random sampling in domain
    x_eval = np.random.uniform(x_domain[0], x_domain[1], n_samples)
    t_eval = np.random.uniform(t_domain[0], t_domain[1], n_samples)
    
    X_eval = torch.tensor(np.column_stack([x_eval, t_eval]), 
                         dtype=torch.float32, device=device)
    
    # For wave equation, we'll use PDE residual as the metric
    # Create dummy target (zeros)
    u_target = torch.zeros(n_samples, 1, dtype=torch.float32, device=device)
    
    return X_eval, u_target


def create_loss_landscape_plots(results_dict, config, output_dir, 
                                landscape_steps=40, landscape_distance=1.0):
    """
    Create loss landscape visualizations comparing standard and curriculum learning.
    
    Args:
        results_dict: Dictionary with training results
        config: Experiment configuration
        output_dir: Directory to save plots
        landscape_steps: Resolution of landscape grid
        landscape_distance: Distance to explore in parameter space
    """
    print("\n" + "="*80)
    print("STEP 5: Creating Loss Landscape Visualizations")
    print("="*80 + "\n")
    
    try:
        visualizer = LossLandscapeVisualizer(output_dir=output_dir)
    except ImportError as e:
        print(f"⚠ Skipping loss landscape: {e}")
        return
    
    # Prepare evaluation data
    X_eval, u_target = prepare_evaluation_data(config, device='cpu')  # Use CPU for landscape
    
    # Extract solvers
    pinn_standard = results_dict['standard']['solver']
    pinn_curriculum = results_dict['curriculum']['solver']
    
    # Get model class and kwargs for trajectory projection
    model_class = type(pinn_standard.model)
    
    # Build model_kwargs based on model type
    if config['model_type'] == 'mlp':
        model_kwargs = {
            'layers': config['layers'],
            'activation': config['activation']
        }
    elif config['model_type'] == 'resnet':
        model_kwargs = {
            'input_dim': config['input_dim'],
            'hidden_dim': config['hidden_dim'],
            'output_dim': config['output_dim'],
            'num_blocks': config.get('num_blocks', 4),
            'activation': config['activation']
        }
    elif config['model_type'] == 'fourier':
        model_kwargs = {
            'layers': config['layers'],
            'input_dim': config['input_dim'],
            'fourier_dim': config['fourier_dim'],
            'sigma': config.get('sigma', 1.0),
            'activation': config['activation']
        }
    else:
        model_kwargs = {
            'layers': config['layers'],
            'activation': config['activation']
        }
    
    # Prepare data for visualization
    landscape_data = {
        'Standard PINN': {
            'model': pinn_standard.model.cpu(),
            'trajectory_states': getattr(pinn_standard, 'trajectory_states', []),
            'trajectory_epochs': getattr(pinn_standard, 'trajectory_epochs', []),
            'color': 'cyan'
        },
        'Curriculum PINN': {
            'model': pinn_curriculum.model.cpu(),
            'trajectory_states': getattr(pinn_curriculum, 'trajectory_states', []),
            'trajectory_epochs': getattr(pinn_curriculum, 'trajectory_epochs', []),
            'color': 'yellow',
            'stage_boundaries': getattr(pinn_curriculum, 'stage_boundaries', []),
            'stage_names': ['Curriculum Stage'] if hasattr(pinn_curriculum, 'stage_boundaries') else []
        }
    }
    
    # Create 2D landscape comparison
    print("  Creating 2D loss landscape comparison...")
    visualizer.create_2d_landscape_comparison(
        results_dict=landscape_data,
        X_eval=X_eval,
        u_target=u_target,
        model_class=model_class,
        model_kwargs=model_kwargs,
        filename=os.path.join(output_dir, 'loss_landscape_2d_wave.png'),
        steps=landscape_steps,
        distance=landscape_distance
    )
    
    # Create 3D landscape comparison
    print("  Creating 3D loss landscape comparison...")
    visualizer.create_3d_landscape_comparison(
        results_dict=landscape_data,
        X_eval=X_eval,
        u_target=u_target,
        model_class=model_class,
        model_kwargs=model_kwargs,
        filename=os.path.join(output_dir, 'loss_landscape_3d_wave.png'),
        steps=landscape_steps,
        distance=landscape_distance
    )
    
    print("  ✓ Loss landscape visualizations complete!")


def run_wave_experiment(config_name='default', output_dir='outputs/wave',
                       use_data_loss=False, enable_landscape=True,
                       landscape_steps=40, landscape_distance=1.0):
    """
    Run complete 1D Wave equation experiment.
    
    Args:
        config_name: Configuration name from wave_config.py
        output_dir: Directory to save results
        use_data_loss: Whether to use data loss during training
        enable_landscape: Whether to generate loss landscape plots
        landscape_steps: Resolution for loss landscape
        landscape_distance: Exploration distance for loss landscape
    """
    # Load configuration
    config = get_config(config_name)
    
    if use_data_loss:
        config['use_data_loss'] = True
        print(f"\n⚠ Data loss ENABLED with {config['N_data']} sampling points")
    
    print("\n" + "="*80)
    print(f"1D WAVE EQUATION EXPERIMENT: {config_name}")
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
    
    # =========================================================================
    # STEP 1: Generate Ground Truth with FDM
    # =========================================================================
    print("="*80)
    print("STEP 1: Generating Ground Truth with FDM")
    print("="*80 + "\n")
    
    fdm_solver = WaveFDM(
        c=config['c'],
        x_domain=config['x_domain'],
        t_domain=config['t_domain'],
        nx=config['fdm_nx'],
        nt=config['fdm_nt'],
        k=config['k']
    )
    u_exact, x_grid, t_grid = fdm_solver.solve()
    print("  ✓ Ground truth generated\n")
    
    # Store FDM solution
    fdm_solution = (u_exact, x_grid, t_grid)
    
    # =========================================================================
    # STEP 2: Train Standard PINN
    # =========================================================================
    print("="*80)
    print("STEP 2: Training Standard PINN")
    print("="*80 + "\n")
    
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
        'c': config['c'],
        'k': config['k'],
        'x_domain': config['x_domain'],
        't_domain': config['t_domain'],
        'N_ic': config['N_ic'],
        'N_bc': config['N_bc'],
        'N_pde': config['N_pde'],
        'lambda_ic': config['lambda_ic'],
        'lambda_ic_t': config['lambda_ic_t'],
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
    
    pinn_standard = WavePINN(model_std, standard_config, device=device)
    pinn_standard.train(verbose=True, save_interval=2000)
    print("  ✓ Standard PINN training complete\n")
    
    # =========================================================================
    # STEP 3: Train Curriculum PINN
    # =========================================================================
    print("="*80)
    print("STEP 3: Training Curriculum PINN")
    print("="*80 + "\n")
    
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
    
    pinn_curriculum = CurriculumWavePINN(model_curr, curriculum_config, device=device)
    pinn_curriculum.train(verbose=True, save_interval=2000)
    print("  ✓ Curriculum PINN training complete\n")
    
    # =========================================================================
    # STEP 4: Generate Standard Visualizations
    # =========================================================================
    print("="*80)
    print("STEP 4: Generating Standard Visualizations")
    print("="*80 + "\n")
    
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
            'label': 'Exact (FDM)'
        }
    }
    
    os.makedirs(output_dir, exist_ok=True)
    create_all_wave_plots(results, output_dir=output_dir)
    print("  ✓ Standard visualizations complete\n")
    
    # =========================================================================
    # STEP 5: Generate Loss Landscape Visualizations (Optional)
    # =========================================================================
    if enable_landscape:
        create_loss_landscape_plots(
            results_dict=results,
            config=config,
            output_dir=output_dir,
            landscape_steps=landscape_steps,
            landscape_distance=landscape_distance
        )
    
    # =========================================================================
    # STEP 6: Save Models and Configuration
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 6: Saving Models and Configuration")
    print("="*80 + "\n")
    
    pinn_standard.save_model(os.path.join(output_dir, 'standard_pinn.pt'))
    pinn_curriculum.save_model(os.path.join(output_dir, 'curriculum_pinn.pt'))
    
    # Save configuration
    config_to_save = {k: v for k, v in config.items() if k != 'fdm_solution'}
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    print("  ✓ Models and configuration saved\n")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("="*80)
    print("1D WAVE EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Standard PINN model: standard_pinn.pt")
    print(f"  - Curriculum PINN model: curriculum_pinn.pt")
    print(f"  - Configuration: config.json")
    print(f"  - Visualizations: *.png")
    if enable_landscape:
        print(f"  - Loss landscapes: loss_landscape_*.png")
    if config['use_data_loss']:
        print(f"\n✓ Data loss was ENABLED with {config['N_data']} sampling points")
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run 1D Wave equation experiment with curriculum learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='default',
        help='Configuration name from wave_config.py'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/wave',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--use-data-loss',
        action='store_true',
        help='Enable data loss during training'
    )
    
    parser.add_argument(
        '--no-landscape',
        action='store_true',
        help='Disable loss landscape visualization'
    )
    
    parser.add_argument(
        '--landscape-steps',
        type=int,
        default=40,
        help='Resolution for loss landscape grid'
    )
    
    parser.add_argument(
        '--landscape-distance',
        type=float,
        default=1.0,
        help='Exploration distance for loss landscape'
    )
    
    args = parser.parse_args()
    
    run_wave_experiment(
        config_name=args.config,
        output_dir=args.output,
        use_data_loss=args.use_data_loss,
        enable_landscape=not args.no_landscape,
        landscape_steps=args.landscape_steps,
        landscape_distance=args.landscape_distance
    )