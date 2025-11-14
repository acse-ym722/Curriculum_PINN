# 文件名: experiment_wave_2d.py

"""
2D Wave Equation Experiment with Loss Landscape Visualization

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
from src.solvers.standard_pinn import WavePINN2D
from src.solvers.curriculum_pinn import CurriculumWavePINN2D
from src.numerics.fdm_wave_2d import WaveFDM2D
from src.visualization.wave_plots_2d import create_all_wave_plots_2d
from src.visualization.loss_landscape_generic import LossLandscapeVisualizer
from src.configs.wave_config_2d import get_config


def check_fdm_stability_2d(config):
    """Check 2D FDM stability before running"""
    print("\n" + "="*80)
    print("PRE-FLIGHT CHECK: 2D FDM Stability Analysis")
    print("="*80)
    
    c = config["c"]
    dx = (config["x_domain"][1] - config["x_domain"][0]) / (config["fdm_nx"] - 1)
    dy = (config["y_domain"][1] - config["y_domain"][0]) / (config["fdm_ny"] - 1)
    dt = (config["t_domain"][1] - config["t_domain"][0]) / (config["fdm_nt"] - 1)
    
    cfl = c * dt * np.sqrt(1/dx**2 + 1/dy**2)
    
    print(f"  dx={dx:.6f}, dy={dy:.6f}, dt={dt:.6f}")
    print(f"  Target wave speed c = {c:.2f}")
    print(f"  CFL = c*dt*sqrt(1/dx^2+1/dy^2) = {cfl:.4f} (must be <= 1.0)")
    
    is_stable = cfl <= 1.0
    if is_stable:
        print("  ✓ FDM parameters are STABLE")
    else:
        print("  ✗ FDM parameters are UNSTABLE")
        min_nt = int(c * (config["t_domain"][1] - config["t_domain"][0]) * 
                    np.sqrt(1/dx**2 + 1/dy**2)) + 1
        print(f"  Suggested: increase fdm_nt to at least {min_nt}")
    
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
        X_eval: Input tensor (x, y, t)
        u_target: Target tensor (dummy, will use PDE loss)
    """
    print(f"  Preparing {n_samples} evaluation points...")
    
    x_domain = config['x_domain']
    y_domain = config['y_domain']
    t_domain = config['t_domain']
    
    # Random sampling in domain
    x_eval = np.random.uniform(x_domain[0], x_domain[1], n_samples)
    y_eval = np.random.uniform(y_domain[0], y_domain[1], n_samples)
    t_eval = np.random.uniform(t_domain[0], t_domain[1], n_samples)
    
    X_eval = torch.tensor(np.column_stack([x_eval, y_eval, t_eval]), 
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_eval, u_target = prepare_evaluation_data(config, device='cpu')  # Use CPU for landscape
    
    # Extract solvers
    pinn_standard = results_dict['standard']['solver']
    pinn_curriculum = results_dict['curriculum']['solver']
    
    # Get model class and kwargs for trajectory projection
    model_class = type(pinn_standard.model)
    model_kwargs = {
        'layers': config['layers'],
        'activation': config['activation']
    }
    
    # Prepare data for visualization
    landscape_data = {
        'Standard PINN': {
            'model': pinn_standard.model.cpu(),
            'trajectory_states': pinn_standard.trajectory_states if hasattr(pinn_standard, 'trajectory_states') else [],
            'trajectory_epochs': pinn_standard.trajectory_epochs if hasattr(pinn_standard, 'trajectory_epochs') else [],
            'color': 'cyan'
        },
        'Curriculum PINN': {
            'model': pinn_curriculum.model.cpu(),
            'trajectory_states': pinn_curriculum.trajectory_states if hasattr(pinn_curriculum, 'trajectory_states') else [],
            'trajectory_epochs': pinn_curriculum.trajectory_epochs if hasattr(pinn_curriculum, 'trajectory_epochs') else [],
            'color': 'yellow',
            'stage_boundaries': getattr(pinn_curriculum, 'stage_boundaries', []),
            'stage_names': [f"c={stage['c']}" for stage in config['curriculum_stages'][1:]]  # Skip first stage
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

def run_wave_experiment_2d(config_name='default', output_dir='outputs/wave_2d',
                          enable_landscape=True, landscape_steps=40, landscape_distance=1.0):
    """
    Run complete 2D Wave equation experiment.
    
    Args:
        config_name: Configuration name from wave_config_2d.py
        output_dir: Directory to save results
        enable_landscape: Whether to generate loss landscape plots
        landscape_steps: Resolution for loss landscape
        landscape_distance: Exploration distance for loss landscape
    """
    config = get_config(config_name)
    
    print("\n" + "="*80)
    print(f"2D WAVE EQUATION EXPERIMENT: {config_name}")
    print("="*80)
    print("\nConfiguration:")
    print(json.dumps(config, indent=2))
    print("="*80 + "\n")
    
    # Pre-flight check
    if not check_fdm_stability_2d(config):
        response = input("FDM may be unstable. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Experiment aborted.")
            return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # =========================================================================
    # STEP 1: Generate Ground Truth with 2D FDM
    # =========================================================================
    print("="*80)
    print("STEP 1: Generating Ground Truth with 2D FDM")
    print("="*80 + "\n")
    
    fdm_solver = WaveFDM2D(
        c=config['c'],
        x_domain=config['x_domain'],
        y_domain=config['y_domain'],
        t_domain=config['t_domain'],
        nx=config['fdm_nx'],
        ny=config['fdm_ny'],
        nt=config['fdm_nt'],
        kx=config['kx'],
        ky=config['ky']
    )
    u_exact, x_grid, y_grid, t_grid = fdm_solver.solve()
    print("  ✓ Ground truth generated\n")
    
    # Total epochs for fair comparison
    total_epochs = sum(stage['epochs'] for stage in config['curriculum_stages'])
    
    # =========================================================================
    # STEP 2: Train Standard PINN
    # =========================================================================
    print("="*80)
    print("STEP 2: Training Standard PINN")
    print("="*80 + "\n")
    
    model_std = create_model(
        model_type=config['model_type'],
        layers=config['layers'],
        activation=config['activation']
    )
    
    standard_config = config.copy()
    standard_config['epochs'] = total_epochs
    
    pinn_standard = WavePINN2D(model_std, standard_config, device=device)
    pinn_standard.train(verbose=True, save_interval=2000)
    print("  ✓ Standard PINN training complete\n")
    
    # =========================================================================
    # STEP 3: Train Curriculum PINN
    # =========================================================================
    print("="*80)
    print("STEP 3: Training Curriculum PINN (Stage-based)")
    print("="*80 + "\n")
    
    model_curr = create_model(
        model_type=config['model_type'],
        layers=config['layers'],
        activation=config['activation']
    )
    
    pinn_curriculum = CurriculumWavePINN2D(model_curr, config, device=device)
    pinn_curriculum.train(verbose=True, save_interval=2000)
    print("  ✓ Curriculum PINN training complete\n")
    
    # =========================================================================
    # STEP 4: Generate Standard Visualizations
    # =========================================================================
    print("="*80)
    print("STEP 4: Generating Standard Visualizations and Saving Data")
    print("="*80 + "\n")
    
    results = {
        'config': config,
        'standard': {'solver': pinn_standard, 'label': 'Standard PINN'},
        'curriculum': {'solver': pinn_curriculum, 'label': 'Curriculum PINN'},
        'exact': {
            'solution': u_exact,
            'x': x_grid,
            'y': y_grid,
            't': t_grid,
            'label': 'Exact (FDM)'
        }
    }
    
    os.makedirs(output_dir, exist_ok=True)
    create_all_wave_plots_2d(results, output_dir=output_dir)
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
    
    pinn_standard.save_model(os.path.join(output_dir, 'standard_pinn_2d.pt'))
    pinn_curriculum.save_model(os.path.join(output_dir, 'curriculum_pinn_2d.pt'))
    
    with open(os.path.join(output_dir, 'config_2d.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("  ✓ Models and configuration saved\n")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("="*80)
    print("2D WAVE EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Standard PINN model: standard_pinn_2d.pt")
    print(f"  - Curriculum PINN model: curriculum_pinn_2d.pt")
    print(f"  - Configuration: config_2d.json")
    print(f"  - Visualizations: *.png")
    if enable_landscape:
        print(f"  - Loss landscapes: loss_landscape_*.png")
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run 2D Wave equation experiment with curriculum learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='default',
        help='Configuration name from wave_config_2d.py'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/wave_2d',
        help='Output directory for results'
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
    
    run_wave_experiment_2d(
        config_name=args.config,
        output_dir=args.output,
        enable_landscape=not args.no_landscape,
        landscape_steps=args.landscape_steps,
        landscape_distance=args.landscape_distance
    )