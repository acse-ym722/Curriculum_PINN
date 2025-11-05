"""
Visualization utilities for Burgers equation experiments
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_burgers_comparison(results, output_dir='outputs/burgers'):
    """
    Comprehensive comparison plot for Burgers equation
    
    Args:
        results: Dictionary containing all experimental results
        output_dir: Directory to save plots
    """
    config = results['config']
    nu = config['nu']
    t_final = config.get('plot_t_final', 0.01)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'PINN Performance for 1D Burgers Equation (ν={nu:.2e})', 
                 fontsize=16, fontweight='bold')
    
    # --- 1. Loss Curve Comparison ---
    ax = axes[0, 0]
    ax.plot(results['standard']['solver'].loss_history, 
            label=results['standard']['label'], alpha=0.7, linewidth=2)
    ax.plot(results['curriculum']['solver'].loss_history, 
            label=results['curriculum']['label'], linewidth=2)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('Loss Curve Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    # --- 2. Nu Evolution (Curriculum Learning) ---
    ax = axes[0, 1]
    curriculum_solver = results['curriculum']['solver']
    
    # Check if this is a curriculum solver with nu history
    if hasattr(curriculum_solver, 'nu_target') and hasattr(curriculum_solver, 'ramp_epochs'):
        # Reconstruct nu evolution
        epochs = np.arange(curriculum_solver.total_epochs)
        nu_history = []
        
        for epoch in epochs:
            if epoch < curriculum_solver.ramp_epochs:
                progress = epoch / curriculum_solver.ramp_epochs
                nu = curriculum_solver.nu_target * 10 * (1 - progress) + curriculum_solver.nu_target * progress
            else:
                nu = curriculum_solver.nu_target
            nu_history.append(nu)
        
        ax.plot(epochs, nu_history, color='green', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('ν (Viscosity Coefficient)', fontsize=12)
        ax.set_title('Curriculum Learning: ν Annealing Schedule', 
                     fontsize=13, fontweight='bold')
        ax.grid(True, ls="--", alpha=0.5)
        ax.axhline(y=curriculum_solver.nu_target, color='red', 
                  linestyle='--', alpha=0.5, label=f'Target ν={curriculum_solver.nu_target:.2e}')
        ax.axhline(y=curriculum_solver.nu_target * 10, color='blue', 
                  linestyle='--', alpha=0.5, label=f'Initial ν={curriculum_solver.nu_target * 10:.2e}')
        ax.legend(fontsize=10)
        
        # Add annotation for ramp period
        ax.axvline(x=curriculum_solver.ramp_epochs, color='orange', 
                  linestyle=':', alpha=0.7, linewidth=1.5)
        ax.text(curriculum_solver.ramp_epochs, ax.get_ylim()[1] * 0.95, 
               f'Ramp ends\n(epoch {curriculum_solver.ramp_epochs})', 
               ha='center', va='top', fontsize=9, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'No curriculum data available', 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
    
    # --- 3. Solution Comparison at t_final ---
    ax = axes[1, 0]
    
    # Extract exact solution at t_final
    exact_data = results['exact']
    t_idx = np.argmin(np.abs(exact_data['t'] - t_final))
    u_exact_final = exact_data['solution'][:, t_idx]
    
    # Get PINN predictions
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_test = torch.linspace(config['x_domain'][0], config['x_domain'][1], 
                           config.get('plot_x_points', 512)).view(-1, 1)
    t_test = torch.full_like(x_test, t_final)
    X_test_tensor = torch.cat([x_test, t_test], dim=1).to(device)
    
    u_pred_std = results['standard']['solver'].predict(X_test_tensor)
    u_pred_curr = results['curriculum']['solver'].predict(X_test_tensor)
    
    ax.plot(exact_data['x'], u_exact_final, 'k-', 
            label=exact_data['label'], linewidth=3, alpha=0.8)
    ax.plot(x_test.cpu().numpy(), u_pred_std, 'b--', 
            label=results['standard']['label'], linewidth=2)
    ax.plot(x_test.cpu().numpy(), u_pred_curr, 'r-.', 
            label=results['curriculum']['label'], linewidth=2)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel(f'u(x, t={t_final})', fontsize=12)
    ax.set_title(f'Solution Comparison at t={t_final}', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, ls="--", alpha=0.5)
    
    # --- 4. Absolute Error Comparison ---
    ax = axes[1, 1]
    
    # Interpolate exact solution to test points
    u_exact_interp = np.interp(x_test.cpu().numpy().flatten(), 
                                exact_data['x'], u_exact_final)
    
    error_std = np.abs(u_pred_std.flatten() - u_exact_interp)
    error_curr = np.abs(u_pred_curr.flatten() - u_exact_interp)
    
    ax.plot(x_test.cpu().numpy(), error_std, 'b--', 
            label=f"Error ({results['standard']['label']})", linewidth=2)
    ax.plot(x_test.cpu().numpy(), error_curr, 'r-.', 
            label=f"Error ({results['curriculum']['label']})", linewidth=2)
    
    ax.set_yscale('log')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title(f'Absolute Error vs. Numerical Solution at t={t_final}', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figure
    import os
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'comprehensive_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to {filepath}")
    
    plt.show()


def plot_burgers_spacetime(results, output_dir='outputs/burgers'):
    """
    Plot space-time contour of solutions
    
    Args:
        results: Dictionary containing experimental results
        output_dir: Directory to save plots
    """
    config = results['config']
    
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create space-time grid
    nx, nt = 200, 200
    x = np.linspace(config['x_domain'][0], config['x_domain'][1], nx)
    t = np.linspace(config['t_domain'][0], config['t_domain'][1], nt)
    X_grid, T_grid = np.meshgrid(x, t)
    
    # Prepare input tensor
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
    
    # Save
    import os
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'spacetime_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Space-time plot saved to {filepath}")
    
    plt.show()


def print_burgers_metrics(results):
    """
    Print quantitative metrics
    
    Args:
        results: Dictionary containing experimental results
    """
    config = results['config']
    t_final = config.get('plot_t_final', 0.01)
    
    print("\n" + "="*80)
    print("QUANTITATIVE ERROR ANALYSIS")
    print("="*80)
    
    # Extract exact solution
    exact_data = results['exact']
    t_idx = np.argmin(np.abs(exact_data['t'] - t_final))
    u_exact = exact_data['solution'][:, t_idx]
    
    # Get predictions
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_test = torch.linspace(config['x_domain'][0], config['x_domain'][1], 
                           len(exact_data['x'])).view(-1, 1)
    t_test = torch.full_like(x_test, t_final)
    X_test = torch.cat([x_test, t_test], dim=1).to(device)
    
    u_std = results['standard']['solver'].predict(X_test).flatten()
    u_curr = results['curriculum']['solver'].predict(X_test).flatten()
    
    # Compute errors
    error_std = u_std - u_exact
    error_curr = u_curr - u_exact
    
    # Metrics for Standard PINN
    print("\n--- Standard PINN ---")
    l2_std = np.linalg.norm(error_std) / np.linalg.norm(u_exact)
    linf_std = np.max(np.abs(error_std))
    print(f"L2 Relative Error: {l2_std:.6e}")
    print(f"L∞ Absolute Error: {linf_std:.6e}")
    
    # Metrics for Curriculum PINN
    print("\n--- Curriculum PINN ---")
    l2_curr = np.linalg.norm(error_curr) / np.linalg.norm(u_exact)
    linf_curr = np.max(np.abs(error_curr))
    print(f"L2 Relative Error: {l2_curr:.6e}")
    print(f"L∞ Absolute Error: {linf_curr:.6e}")
    
    # Improvement
    print("\n--- Improvement by Curriculum Learning ---")
    improvement_l2 = (l2_std - l2_curr) / l2_std * 100
    improvement_linf = (linf_std - linf_curr) / linf_std * 100
    print(f"L2 Error Reduction: {improvement_l2:+.2f}%")
    print(f"L∞ Error Reduction: {improvement_linf:+.2f}%")
    
    print("="*80 + "\n")


def create_all_burgers_plots(results, output_dir='outputs/burgers'):
    """
    Generate all plots for Burgers equation
    
    Args:
        results: Dictionary containing experimental results
        output_dir: Directory to save plots
    """
    print("\n" + "="*80)
    print("GENERATING VISUALIZATION FOR BURGERS EQUATION")
    print("="*80 + "\n")
    
    plot_burgers_comparison(results, output_dir)
    plot_burgers_spacetime(results, output_dir)
    print_burgers_metrics(results)
    
    print("✓ All visualizations completed!\n")