"""
Visualization utilities for lid-driven cavity flow experiments
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import RectBivariateSpline


def plot_cavity_comparison(fdm_solver, standard_pinn, curriculum_pinn, 
                           config, output_dir='outputs/cavity'):
    """
    Comprehensive comparison plot for cavity flow
    
    Args:
        fdm_solver: FDM solver object
        standard_pinn: Standard PINN solver
        curriculum_pinn: Curriculum PINN solver
        config: Configuration dictionary
        output_dir: Directory to save plots
    """
    import torch
    import os
    
    Re = config['Re_target']
    resolution = 100
    
    # Create prediction grid
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X_grid, Y_grid = np.meshgrid(x, y)
    
    # FDM solution (interpolated)
    fdm_u_interp = RectBivariateSpline(fdm_solver.y, fdm_solver.x, fdm_solver.u)
    fdm_v_interp = RectBivariateSpline(fdm_solver.y, fdm_solver.x, fdm_solver.v)
    fdm_p_interp = RectBivariateSpline(fdm_solver.y, fdm_solver.x, fdm_solver.p)
    
    U_fdm = fdm_u_interp(y, x)
    V_fdm = fdm_v_interp(y, x)
    P_fdm = fdm_p_interp(y, x)
    
    # PINN predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_flat = torch.tensor(np.c_[X_grid.ravel(), Y_grid.ravel()], 
                         dtype=torch.float32).to(device)
    
    UVP_standard = standard_pinn.predict(X_flat)
    U_standard = UVP_standard[:, 0].reshape(resolution, resolution)
    V_standard = UVP_standard[:, 1].reshape(resolution, resolution)
    P_standard = UVP_standard[:, 2].reshape(resolution, resolution)
    
    UVP_curriculum = curriculum_pinn.predict(X_flat)
    U_curriculum = UVP_curriculum[:, 0].reshape(resolution, resolution)
    V_curriculum = UVP_curriculum[:, 1].reshape(resolution, resolution)
    P_curriculum = UVP_curriculum[:, 2].reshape(resolution, resolution)
    
    # Compute errors
    error_u_std = np.abs(U_fdm - U_standard)
    error_v_std = np.abs(V_fdm - V_standard)
    error_u_curr = np.abs(U_fdm - U_curriculum)
    error_v_curr = np.abs(V_fdm - V_curriculum)
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    
    titles = [
        'Ground Truth (FDM)\nu velocity',
        'Standard PINN\nu velocity',
        'Curriculum PINN\nu velocity',
        'Standard PINN\nError in u',
        'Curriculum PINN\nError in u',
        
        'Ground Truth (FDM)\nv velocity',
        'Standard PINN\nv velocity',
        'Curriculum PINN\nv velocity',
        'Standard PINN\nError in v',
        'Curriculum PINN\nError in v',
        
        'Ground Truth (FDM)\nPressure',
        'Standard PINN\nPressure',
        'Curriculum PINN\nPressure',
        'Velocity Magnitude\n(Ground Truth)',
        'Training Loss\nComparison',
    ]
    
    data = [
        U_fdm, U_standard, U_curriculum, error_u_std, error_u_curr,
        V_fdm, V_standard, V_curriculum, error_v_std, error_v_curr,
        P_fdm, P_standard, P_curriculum, np.sqrt(U_fdm**2 + V_fdm**2), None
    ]
    
    for i in range(15):
        ax = plt.subplot(3, 5, i+1)
        
        if i == 14:  # Loss comparison
            ax.semilogy(standard_pinn.loss_history, label='Standard PINN', linewidth=2)
            ax.semilogy(curriculum_pinn.loss_history, label='Curriculum PINN', linewidth=2)
            
            # Mark curriculum stages
            if hasattr(curriculum_pinn, 'Re_history'):
                epoch_count = 0
                for stage in config.get('curriculum_stages', []):
                    epoch_count += stage['epochs']
                    ax.axvline(epoch_count, color='red', linestyle='--', alpha=0.5)
                    ax.text(epoch_count, ax.get_ylim()[1]*0.5, 
                           f"Re={stage['Re']}", rotation=90, fontsize=8)
            
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Loss', fontsize=11)
            ax.set_title('Training Loss Comparison', fontsize=10, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            im = ax.contourf(X_grid, Y_grid, data[i], levels=20, cmap='jet')
            ax.set_title(titles[i], fontsize=9, fontweight='bold')
            ax.set_xlabel('x', fontsize=10)
            ax.set_ylabel('y', fontsize=10)
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Add streamlines for velocity fields
            if i < 3 or (5 <= i < 8):
                if i < 3:
                    U_plot, V_plot = data[i], V_fdm
                elif 5 <= i < 8:
                    U_plot, V_plot = data[i], V_fdm
                else:
                    U_plot, V_plot = U_fdm, V_fdm
                    
                ax.streamplot(X_grid, Y_grid, U_plot, V_plot, 
                             color='white', linewidth=0.5, density=1.2, arrowsize=0.6)
    
    plt.suptitle(f'Comprehensive Comparison: FDM vs Standard PINN vs Curriculum PINN (Re = {Re})', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'comprehensive_comparison.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison plot saved to {filepath}")
    
    plt.show()


def plot_cavity_centerlines(fdm_solver, standard_pinn, curriculum_pinn, 
                            config, output_dir='outputs/cavity'):
    """
    Plot velocity profiles along centerlines
    
    Args:
        fdm_solver: FDM solver object
        standard_pinn: Standard PINN solver
        curriculum_pinn: Curriculum PINN solver
        config: Configuration dictionary
        output_dir: Directory to save plots
    """
    import torch
    import os
    
    resolution = 200
    
    # Horizontal centerline (y=0.5)
    x_h = np.linspace(0, 1, resolution)
    y_h = np.full_like(x_h, 0.5)
    
    # Vertical centerline (x=0.5)
    x_v = np.full(resolution, 0.5)
    y_v = np.linspace(0, 1, resolution)
    
    # FDM interpolation
    fdm_u_interp = RectBivariateSpline(fdm_solver.y, fdm_solver.x, fdm_solver.u)
    fdm_v_interp = RectBivariateSpline(fdm_solver.y, fdm_solver.x, fdm_solver.v)
    
    u_fdm_h = fdm_u_interp(y_h, x_h, grid=False)
    v_fdm_v = fdm_v_interp(y_v, x_v, grid=False)
    
    # PINN predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_h = torch.tensor(np.c_[x_h, y_h], dtype=torch.float32).to(device)
    X_v = torch.tensor(np.c_[x_v, y_v], dtype=torch.float32).to(device)
    
    UVP_std_h = standard_pinn.predict(X_h)
    UVP_std_v = standard_pinn.predict(X_v)
    UVP_curr_h = curriculum_pinn.predict(X_h)
    UVP_curr_v = curriculum_pinn.predict(X_v)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Horizontal centerline (u velocity)
    axes[0].plot(x_h, u_fdm_h, 'k-', linewidth=2, label='FDM (Ground Truth)')
    axes[0].plot(x_h, UVP_std_h[:, 0], 'b--', linewidth=2, label='Standard PINN')
    axes[0].plot(x_h, UVP_curr_h[:, 0], 'r:', linewidth=2, label='Curriculum PINN')
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('u velocity', fontsize=12)
    axes[0].set_title('Horizontal Centerline (y = 0.5)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Vertical centerline (v velocity)
    axes[1].plot(v_fdm_v, y_v, 'k-', linewidth=2, label='FDM (Ground Truth)')
    axes[1].plot(UVP_std_v[:, 1], y_v, 'b--', linewidth=2, label='Standard PINN')
    axes[1].plot(UVP_curr_v[:, 1], y_v, 'r:', linewidth=2, label='Curriculum PINN')
    axes[1].set_xlabel('v velocity', fontsize=12)
    axes[1].set_ylabel('y', fontsize=12)
    axes[1].set_title('Vertical Centerline (x = 0.5)', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'centerline_comparison.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✓ Centerline plot saved to {filepath}")
    
    plt.show()


def print_cavity_metrics(fdm_solver, standard_pinn, curriculum_pinn, config):
    """
    Print quantitative metrics for cavity flow
    
    Args:
        fdm_solver: FDM solver object
        standard_pinn: Standard PINN solver
        curriculum_pinn: Curriculum PINN solver
        config: Configuration dictionary
    """
    import torch
    
    resolution = 100
    
    # Create grid
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X_grid, Y_grid = np.meshgrid(x, y)
    
    # FDM interpolation
    fdm_u_interp = RectBivariateSpline(fdm_solver.y, fdm_solver.x, fdm_solver.u)
    fdm_v_interp = RectBivariateSpline(fdm_solver.y, fdm_solver.x, fdm_solver.v)
    
    U_fdm = fdm_u_interp(y, x)
    V_fdm = fdm_v_interp(y, x)
    
    # PINN predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_flat = torch.tensor(np.c_[X_grid.ravel(), Y_grid.ravel()], 
                         dtype=torch.float32).to(device)
    
    UVP_std = standard_pinn.predict(X_flat)
    U_std = UVP_std[:, 0].reshape(resolution, resolution)
    V_std = UVP_std[:, 1].reshape(resolution, resolution)
    
    UVP_curr = curriculum_pinn.predict(X_flat)
    U_curr = UVP_curr[:, 0].reshape(resolution, resolution)
    V_curr = UVP_curr[:, 1].reshape(resolution, resolution)
    
    print("\n" + "="*80)
    print("QUANTITATIVE ERROR ANALYSIS")
    print("="*80)
    
    # Standard PINN
    print("\n--- Standard PINN ---")
    l2_u_std = np.linalg.norm(U_std - U_fdm) / np.linalg.norm(U_fdm)
    l2_v_std = np.linalg.norm(V_std - V_fdm) / np.linalg.norm(V_fdm)
    max_u_std = np.max(np.abs(U_std - U_fdm))
    max_v_std = np.max(np.abs(V_std - V_fdm))
    
    print(f"L2 Relative Error in u: {l2_u_std:.6f}")
    print(f"L2 Relative Error in v: {l2_v_std:.6f}")
    print(f"Max Absolute Error in u: {max_u_std:.6f}")
    print(f"Max Absolute Error in v: {max_v_std:.6f}")
    
    # Curriculum PINN
    print("\n--- Curriculum PINN ---")
    l2_u_curr = np.linalg.norm(U_curr - U_fdm) / np.linalg.norm(U_fdm)
    l2_v_curr = np.linalg.norm(V_curr - V_fdm) / np.linalg.norm(V_fdm)
    max_u_curr = np.max(np.abs(U_curr - U_fdm))
    max_v_curr = np.max(np.abs(V_curr - V_fdm))
    
    print(f"L2 Relative Error in u: {l2_u_curr:.6f}")
    print(f"L2 Relative Error in v: {l2_v_curr:.6f}")
    print(f"Max Absolute Error in u: {max_u_curr:.6f}")
    print(f"Max Absolute Error in v: {max_v_curr:.6f}")
    
    # Improvement
    print("\n--- Improvement by Curriculum Learning ---")
    improvement_l2_u = (l2_u_std - l2_u_curr) / l2_u_std * 100
    improvement_l2_v = (l2_v_std - l2_v_curr) / l2_v_std * 100
    improvement_max_u = (max_u_std - max_u_curr) / max_u_std * 100
    improvement_max_v = (max_v_std - max_v_curr) / max_v_std * 100
    
    print(f"L2 Error Reduction in u: {improvement_l2_u:+.2f}%")
    print(f"L2 Error Reduction in v: {improvement_l2_v:+.2f}%")
    print(f"Max Error Reduction in u: {improvement_max_u:+.2f}%")
    print(f"Max Error Reduction in v: {improvement_max_v:+.2f}%")
    
    print("="*80 + "\n")


def create_all_cavity_plots(fdm_solver, standard_pinn, curriculum_pinn, 
                            config, output_dir='outputs/cavity'):
    """
    Generate all plots for cavity flow
    
    Args:
        fdm_solver: FDM solver object
        standard_pinn: Standard PINN solver
        curriculum_pinn: Curriculum PINN solver
        config: Configuration dictionary
        output_dir: Directory to save plots
    """
    print("\n" + "="*80)
    print("GENERATING VISUALIZATION FOR CAVITY FLOW")
    print("="*80 + "\n")
    
    plot_cavity_comparison(fdm_solver, standard_pinn, curriculum_pinn, 
                          config, output_dir)
    plot_cavity_centerlines(fdm_solver, standard_pinn, curriculum_pinn, 
                           config, output_dir)
    print_cavity_metrics(fdm_solver, standard_pinn, curriculum_pinn, config)
    
    print("✓ All visualizations completed!\n")