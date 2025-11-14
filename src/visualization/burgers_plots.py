"""
Visualization utilities for Burgers equation experiments, with data export functionality.
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib import cm

# --- Helper Function for Saving Data ---

def _save_data(data_dict, filename, output_dir, format='csv'):
    """
    Saves data to a specified format in a 'data' subdirectory.

    Args:
        data_dict (dict or np.ndarray): Data to save. If dict, keys are columns.
        filename (str): Name of the file (e.g., 'loss_curves.csv').
        output_dir (str): The base output directory for the experiment.
        format (str): 'csv', 'npy', or 'json'.
    """
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)

    try:
        if format == 'csv':
            df = pd.DataFrame(data_dict)
            df.to_csv(filepath, index=False)
        elif format == 'npy':
            np.save(filepath, data_dict)
        elif format == 'json':
            with open(filepath, 'w') as f:
                json.dump(data_dict, f, indent=4)
        else:
            print(f"  [Warning] Unsupported save format: {format}")
            return

        print(f"  -> Data saved to {filepath}")
    except Exception as e:
        print(f"  [Error] Failed to save data to {filepath}: {e}")

# --- Modified Plotting and Analysis Functions ---

def plot_burgers_comparison(results, output_dir='outputs/burgers'):
    """
    Comprehensive comparison plot for Burgers equation.
    Also saves the data used for each subplot.

    Args:
        results: Dictionary containing all experimental results.
        output_dir: Directory to save plots and data.
    """
    print("1. Generating comprehensive comparison plot...")
    config = results['config']
    nu = config['nu']
    t_final = config.get('plot_t_final', 0.99)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'PINN Performance for 1D Burgers Equation (ν={nu:.2e})',
                 fontsize=16, fontweight='bold')

    # --- 1. Loss Curve Comparison ---
    ax = axes[0, 0]
    std_loss = results['standard']['solver'].loss_history
    curr_loss = results['curriculum']['solver'].loss_history
    # Ensure epoch arrays are of the same length as loss histories
    epochs_std = np.arange(len(std_loss))
    epochs_curr = np.arange(len(curr_loss))
    
    ax.plot(epochs_std, std_loss, label=results['standard']['label'], alpha=0.7, linewidth=2)
    ax.plot(epochs_curr, curr_loss, label=results['curriculum']['label'], linewidth=2)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('Loss Curve Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    # Save loss data
    max_len = max(len(std_loss), len(curr_loss))
    loss_data = {
        'epoch': np.arange(max_len),
        'standard_pinn_loss': pd.Series(std_loss),
        'curriculum_pinn_loss': pd.Series(curr_loss)
    }
    _save_data(loss_data, 'loss_curves.csv', output_dir)

    # --- 2. Nu Evolution (Curriculum Learning) - UPDATED FOR STAGE-BASED LEARNING ---
    ax = axes[0, 1]
    curriculum_solver = results['curriculum']['solver']

    # Check for the new 'nu_history' attribute from the stage-based solver
    if hasattr(curriculum_solver, 'nu_history') and curriculum_solver.nu_history:
        nu_history = curriculum_solver.nu_history
        epochs_nu = np.arange(len(nu_history))
        
        # Use 'steps-post' to create a clear step-function plot
        ax.plot(epochs_nu, nu_history, color='green', linewidth=2, drawstyle='steps-post',
                label='Actual ν value per epoch')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('ν (Viscosity Coefficient)', fontsize=12)
        ax.set_title('Curriculum Learning: Staged ν Schedule',
                     fontsize=13, fontweight='bold')
        ax.grid(True, ls="--", alpha=0.5)
        
        # Add vertical lines to indicate stage transitions
        stage_epochs = curriculum_solver.config.get('curriculum_stages', [])
        transition_epochs = np.cumsum([s['epochs'] for s in stage_epochs])
        
        current_ylim = ax.get_ylim()
        for i, epoch in enumerate(transition_epochs[:-1]): # Don't mark the final end
            ax.axvline(x=epoch, color='orange', linestyle=':', alpha=0.9, linewidth=1.5)
            # Place text annotation for stage transition
            ax.text(epoch + 50, current_ylim[1] * 0.9, f'Stage {i+2} Start',
                    ha='left', va='top', fontsize=9, rotation=90,
                    bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))

        ax.legend(fontsize=10)
        # Set y-axis to log scale if the nu values span a large range
        if max(nu_history) / min(nu_history) > 10:
            ax.set_yscale('log')

        # Save nu evolution data
        nu_data = {'epoch': epochs_nu, 'viscosity_nu': nu_history}
        _save_data(nu_data, 'nu_evolution.csv', output_dir)
    else:
        ax.text(0.5, 0.5, 'No curriculum data available',
                ha='center', va='center', fontsize=12)
        ax.axis('off')

    # --- 3. Solution Comparison at t_final ---
    ax = axes[1, 0]
    exact_data = results['exact']
    t_idx = np.argmin(np.abs(exact_data['t'] - t_final))
    u_exact_final = exact_data['solution'][:, t_idx]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_test = torch.linspace(config['x_domain'][0], config['x_domain'][1],
                           config.get('plot_x_points', 512)).view(-1, 1)
    t_test = torch.full_like(x_test, t_final)
    X_test_tensor = torch.cat([x_test, t_test], dim=1).to(device)
    
    u_pred_std = results['standard']['solver'].predict(X_test_tensor)
    u_pred_curr = results['curriculum']['solver'].predict(X_test_tensor)
    x_test_np = x_test.cpu().numpy().flatten()

    ax.plot(exact_data['x'], u_exact_final, 'k-',
            label=exact_data['label'], linewidth=3, alpha=0.8)
    ax.plot(x_test_np, u_pred_std, 'b--',
            label=results['standard']['label'], linewidth=2)
    ax.plot(x_test_np, u_pred_curr, 'r-.',
            label=results['curriculum']['label'], linewidth=2)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel(f'u(x, t={t_final})', fontsize=12)
    ax.set_title(f'Solution Comparison at t={t_final}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, ls="--", alpha=0.5)

    # --- 4. Absolute Error Comparison ---
    ax = axes[1, 1]
    u_exact_interp = np.interp(x_test_np, exact_data['x'], u_exact_final)
    error_std = np.abs(u_pred_std.flatten() - u_exact_interp)
    error_curr = np.abs(u_pred_curr.flatten() - u_exact_interp)

    ax.plot(x_test_np, error_std, 'b--',
            label=f"Error ({results['standard']['label']})", linewidth=2)
    ax.plot(x_test_np, error_curr, 'r-.',
            label=f"Error ({results['curriculum']['label']})", linewidth=2)

    ax.set_yscale('log')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title(f'Absolute Error vs. Numerical Solution at t={t_final}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.5)

    # Save solution and error data from plots 3 and 4
    solution_error_data = {
        'x': x_test_np,
        'u_exact_interpolated': u_exact_interp,
        'u_standard_pinn': u_pred_std.flatten(),
        'u_curriculum_pinn': u_pred_curr.flatten(),
        'error_standard_pinn': error_std,
        'error_curriculum_pinn': error_curr
    }
    _save_data(solution_error_data, f'solution_and_error_t_{t_final:.2f}.csv', output_dir)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    filepath = os.path.join(output_dir, 'comprehensive_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to {filepath}")
    # plt.show()


def plot_burgers_spacetime(results, output_dir='outputs/burgers'):
    """
    Plot space-time contour of solutions and save the underlying grid data.

    Args:
        results: Dictionary containing experimental results.
        output_dir: Directory to save plots and data.
    """
    print("2. Generating space-time solution plot...")
    config = results['config']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create space-time grid for PINN prediction
    nx, nt = 200, 200
    x = np.linspace(config['x_domain'][0], config['x_domain'][1], nx)
    t = np.linspace(config['t_domain'][0], config['t_domain'][1], nt)
    X_grid, T_grid = np.meshgrid(x, t)

    X_flat = np.c_[X_grid.ravel(), T_grid.ravel()]
    X_tensor = torch.tensor(X_flat, dtype=torch.float32).to(device)

    # Get predictions
    u_std = results['standard']['solver'].predict(X_tensor).reshape(nt, nx)
    u_curr = results['curriculum']['solver'].predict(X_tensor).reshape(nt, nx)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Exact solution
    exact_data = results['exact']
    im0 = axes[0].contourf(exact_data['x'], exact_data['t'],
                            exact_data['solution'].T, levels=20, cmap='RdBu_r')
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('t', fontsize=12)
    axes[0].set_title('Ground Truth (FDM)', fontsize=13, fontweight='bold')
    plt.colorbar(im0, ax=axes[0])

    # Standard PINN
    im1 = axes[1].contourf(X_grid, T_grid, u_std, levels=20, cmap='RdBu_r')
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('t', fontsize=12)
    axes[1].set_title('Standard PINN', fontsize=13, fontweight='bold')
    plt.colorbar(im1, ax=axes[1])

    # Curriculum PINN
    im2 = axes[2].contourf(X_grid, T_grid, u_curr, levels=20, cmap='RdBu_r')
    axes[2].set_xlabel('x', fontsize=12)
    axes[2].set_ylabel('t', fontsize=12)
    axes[2].set_title('Curriculum PINN', fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle('Space-Time Solution Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save plot
    filepath = os.path.join(output_dir, 'spacetime_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Space-time plot saved to {filepath}")

    # Save space-time data
    print("  -> Saving space-time grid data...")
    # Exact data
    _save_data(exact_data['x'], 'spacetime_exact_x_grid.npy', output_dir, format='npy')
    _save_data(exact_data['t'], 'spacetime_exact_t_grid.npy', output_dir, format='npy')
    _save_data(exact_data['solution'], 'spacetime_exact_solution.npy', output_dir, format='npy')
    # PINN data
    _save_data(x, 'spacetime_pinn_x_grid.npy', output_dir, format='npy')
    _save_data(t, 'spacetime_pinn_t_grid.npy', output_dir, format='npy')
    _save_data(u_std, 'spacetime_standard_pinn_solution.npy', output_dir, format='npy')
    _save_data(u_curr, 'spacetime_curriculum_pinn_solution.npy', output_dir, format='npy')
    
    # plt.show()


def print_burgers_metrics(results, output_dir='outputs/burgers'):
    """
    Print quantitative metrics and save them to a JSON file.

    Args:
        results: Dictionary containing experimental results.
        output_dir: Directory to save the metrics file.
    """
    print("3. Calculating and saving quantitative metrics...")
    config = results['config']
    t_final = config.get('plot_t_final', 0.99) # Use the same t_final as plots

    exact_data = results['exact']
    t_idx = np.argmin(np.abs(exact_data['t'] - t_final))
    u_exact = exact_data['solution'][:, t_idx]
    x_exact = exact_data['x']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_test = torch.tensor(x_exact, dtype=torch.float32).view(-1, 1)
    t_test = torch.full_like(x_test, t_final)
    X_test = torch.cat([x_test, t_test], dim=1).to(device)

    u_std = results['standard']['solver'].predict(X_test).flatten()
    u_curr = results['curriculum']['solver'].predict(X_test).flatten()

    error_std = u_std - u_exact
    error_curr = u_curr - u_exact

    metrics = {
        'evaluation_time_t': t_final,
        'standard_pinn': {
            'l2_relative_error': np.linalg.norm(error_std) / np.linalg.norm(u_exact),
            'linf_absolute_error': np.max(np.abs(error_std))
        },
        'curriculum_pinn': {
            'l2_relative_error': np.linalg.norm(error_curr) / np.linalg.norm(u_exact),
            'linf_absolute_error': np.max(np.abs(error_curr))
        }
    }
    
    # Calculate improvement
    l2_std = metrics['standard_pinn']['l2_relative_error']
    l2_curr = metrics['curriculum_pinn']['l2_relative_error']
    linf_std = metrics['standard_pinn']['linf_absolute_error']
    linf_curr = metrics['curriculum_pinn']['linf_absolute_error']
    
    metrics['improvement'] = {
        'l2_error_reduction_percent': (l2_std - l2_curr) / l2_std * 100 if l2_std > 0 else 0,
        'linf_error_reduction_percent': (linf_std - linf_curr) / linf_std * 100 if linf_std > 0 else 0
    }

    # Print to console
    print("\n" + "="*80)
    print(f"QUANTITATIVE ERROR ANALYSIS (at t={t_final:.2f})")
    print("="*80)
    print("\n--- Standard PINN ---")
    print(f"L2 Relative Error: {metrics['standard_pinn']['l2_relative_error']:.6e}")
    print(f"L∞ Absolute Error: {metrics['standard_pinn']['linf_absolute_error']:.6e}")
    print("\n--- Curriculum PINN ---")
    print(f"L2 Relative Error: {metrics['curriculum_pinn']['l2_relative_error']:.6e}")
    print(f"L∞ Absolute Error: {metrics['curriculum_pinn']['linf_absolute_error']:.6e}")
    print("\n--- Improvement by Curriculum Learning ---")
    print(f"L2 Error Reduction: {metrics['improvement']['l2_error_reduction_percent']:+.2f}%")
    print(f"L∞ Error Reduction: {metrics['improvement']['linf_error_reduction_percent']:+.2f}%")
    print("="*80 + "\n")

    # Save metrics to JSON
    _save_data(metrics, 'quantitative_metrics.json', output_dir, format='json')


def create_all_burgers_plots(results, output_dir='outputs/burgers'):
    """
    Generate all plots and data exports for Burgers equation.

    Args:
        results: Dictionary containing experimental results.
        output_dir: Directory to save plots and data.
    """
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS & DATA FOR BURGERS EQUATION")
    print("="*80 + "\n")

    plot_burgers_comparison(results, output_dir)
    plot_burgers_spacetime(results, output_dir)
    print_burgers_metrics(results, output_dir)

    print("✓ All visualizations and data exports completed!\n")