"""
Visualization utilities for cylinder flow experiments
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import os


def plot_cylinder_comparison(fdm_solver, pinn_standard, pinn_curriculum, 
                            config, output_dir='outputs/cylinder'):
    """
    Comprehensive comparison plot for cylinder flow
    
    Args:
        fdm_solver: FDM solver object (ground truth)
        pinn_standard: Standard PINN solver
        pinn_curriculum: Curriculum PINN solver
        config: Configuration dictionary
        output_dir: Directory to save plots
    """
    Re = config['Re_target']
    cx, cy = config['cylinder_center']
    radius = config['cylinder_radius']
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'Flow Past Cylinder (Re={Re})', fontsize=18, fontweight='bold')
    
    # Prepare grid for PINN evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nx_plot, ny_plot = config.get('plot_resolution', 150), config.get('plot_resolution', 150) // 2
    x_plot = np.linspace(config['x_domain'][0], config['x_domain'][1], nx_plot)
    y_plot = np.linspace(config['y_domain'][0], config['y_domain'][1], ny_plot)
    X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
    
    # Flatten and create tensor
    X_flat = np.c_[X_plot.ravel(), Y_plot.ravel()]
    X_tensor = torch.tensor(X_flat, dtype=torch.float32).to(device)
    
    # Get PINN predictions
    UVP_std = pinn_standard.predict(X_tensor)
    UVP_curr = pinn_curriculum.predict(X_tensor)
    
    U_std = UVP_std[:, 0].reshape(ny_plot, nx_plot)
    V_std = UVP_std[:, 1].reshape(ny_plot, nx_plot)
    P_std = UVP_std[:, 2].reshape(ny_plot, nx_plot)
    
    U_curr = UVP_curr[:, 0].reshape(ny_plot, nx_plot)
    V_curr = UVP_curr[:, 1].reshape(ny_plot, nx_plot)
    P_curr = UVP_curr[:, 2].reshape(ny_plot, nx_plot)
    
    # Interpolate FDM to same grid
    from scipy.interpolate import RegularGridInterpolator
    u_fdm_interp = RegularGridInterpolator((fdm_solver.y, fdm_solver.x), fdm_solver.u)
    v_fdm_interp = RegularGridInterpolator((fdm_solver.y, fdm_solver.x), fdm_solver.v)
    p_fdm_interp = RegularGridInterpolator((fdm_solver.y, fdm_solver.x), fdm_solver.p)
    
    points = np.c_[Y_plot.ravel(), X_plot.ravel()]
    U_fdm = u_fdm_interp(points).reshape(ny_plot, nx_plot)
    V_fdm = v_fdm_interp(points).reshape(ny_plot, nx_plot)
    P_fdm = p_fdm_interp(points).reshape(ny_plot, nx_plot)
    
    # Compute velocity magnitude
    Speed_fdm = np.sqrt(U_fdm**2 + V_fdm**2)
    Speed_std = np.sqrt(U_std**2 + V_std**2)
    Speed_curr = np.sqrt(U_curr**2 + V_curr**2)
    
    # --- Row 1: Velocity Magnitude ---
    vmin, vmax = 0, config['U_inf'] * 1.5
    
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.contourf(X_plot, Y_plot, Speed_fdm, levels=20, cmap='jet', vmin=vmin, vmax=vmax)
    _plot_cylinder(ax1, cx, cy, radius)
    ax1.set_title('FDM: Velocity Magnitude', fontsize=13, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1, label='|V|')
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.contourf(X_plot, Y_plot, Speed_std, levels=20, cmap='jet', vmin=vmin, vmax=vmax)
    _plot_cylinder(ax2, cx, cy, radius)
    ax2.set_title('Standard PINN: Velocity Magnitude', fontsize=13, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2, label='|V|')
    
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.contourf(X_plot, Y_plot, Speed_curr, levels=20, cmap='jet', vmin=vmin, vmax=vmax)
    _plot_cylinder(ax3, cx, cy, radius)
    ax3.set_title('Curriculum PINN: Velocity Magnitude', fontsize=13, fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.colorbar(im3, ax=ax3, label='|V|')
    
    # --- Row 2: Pressure ---
    pmin, pmax = np.min(P_fdm), np.max(P_fdm)
    
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.contourf(X_plot, Y_plot, P_fdm, levels=20, cmap='RdBu_r', vmin=pmin, vmax=pmax)
    _plot_cylinder(ax4, cx, cy, radius)
    ax4.set_title('FDM: Pressure', fontsize=13, fontweight='bold')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    plt.colorbar(im4, ax=ax4, label='p')
    
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.contourf(X_plot, Y_plot, P_std, levels=20, cmap='RdBu_r', vmin=pmin, vmax=pmax)
    _plot_cylinder(ax5, cx, cy, radius)
    ax5.set_title('Standard PINN: Pressure', fontsize=13, fontweight='bold')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    plt.colorbar(im5, ax=ax5, label='p')
    
    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.contourf(X_plot, Y_plot, P_curr, levels=20, cmap='RdBu_r', vmin=pmin, vmax=pmax)
    _plot_cylinder(ax6, cx, cy, radius)
    ax6.set_title('Curriculum PINN: Pressure', fontsize=13, fontweight='bold')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    plt.colorbar(im6, ax=ax6, label='p')
    
    # --- Row 3: Errors ---
    error_std_vel = np.abs(Speed_std - Speed_fdm)
    error_curr_vel = np.abs(Speed_curr - Speed_fdm)
    
    ax7 = fig.add_subplot(gs[2, 0])
    im7 = ax7.contourf(X_plot, Y_plot, error_std_vel, levels=20, cmap='hot')
    _plot_cylinder(ax7, cx, cy, radius)
    ax7.set_title('Standard PINN: Velocity Error', fontsize=13, fontweight='bold')
    ax7.set_xlabel('x')
    ax7.set_ylabel('y')
    plt.colorbar(im7, ax=ax7, label='|Error|')
    
    ax8 = fig.add_subplot(gs[2, 1])
    im8 = ax8.contourf(X_plot, Y_plot, error_curr_vel, levels=20, cmap='hot')
    _plot_cylinder(ax8, cx, cy, radius)
    ax8.set_title('Curriculum PINN: Velocity Error', fontsize=13, fontweight='bold')
    ax8.set_xlabel('x')
    ax8.set_ylabel('y')
    plt.colorbar(im8, ax=ax8, label='|Error|')
    
    # --- Loss curves ---
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(pinn_standard.loss_history, label='Standard PINN', linewidth=2, alpha=0.7)
    ax9.plot(pinn_curriculum.loss_history, label='Curriculum PINN', linewidth=2)
    ax9.set_yscale('log')
    ax9.set_xlabel('Epoch')
    ax9.set_ylabel('Loss')
    ax9.set_title('Training Loss Comparison', fontsize=13, fontweight='bold')
    ax9.legend()
    ax9.grid(True, which='both', ls='--', alpha=0.5)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'comprehensive_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to {filepath}")
    
    # plt.show()


def plot_cylinder_streamlines(fdm_solver, pinn_standard, pinn_curriculum,
                              config, output_dir='outputs/cylinder'):
    """Plot streamlines around cylinder"""
    Re = config['Re_target']
    cx, cy = config['cylinder_center']
    radius = config['cylinder_radius']
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(f'Streamlines: Flow Past Cylinder (Re={Re})', fontsize=16, fontweight='bold')
    
    # Prepare grid
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nx_plot, ny_plot = 200, 100
    x_plot = np.linspace(config['x_domain'][0], config['x_domain'][1], nx_plot)
    y_plot = np.linspace(config['y_domain'][0], config['y_domain'][1], ny_plot)
    X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
    
    # Get PINN predictions
    X_flat = np.c_[X_plot.ravel(), Y_plot.ravel()]
    X_tensor = torch.tensor(X_flat, dtype=torch.float32).to(device)
    
    UVP_std = pinn_standard.predict(X_tensor)
    UVP_curr = pinn_curriculum.predict(X_tensor)
    
    U_std = UVP_std[:, 0].reshape(ny_plot, nx_plot)
    V_std = UVP_std[:, 1].reshape(ny_plot, nx_plot)
    U_curr = UVP_curr[:, 0].reshape(ny_plot, nx_plot)
    V_curr = UVP_curr[:, 1].reshape(ny_plot, nx_plot)
    
    # FDM data
    U_fdm = fdm_solver.u
    V_fdm = fdm_solver.v
    X_fdm, Y_fdm = fdm_solver.X, fdm_solver.Y
    
    # Plot FDM
    speed_fdm = np.sqrt(U_fdm**2 + V_fdm**2)
    axes[0].streamplot(X_fdm, Y_fdm, U_fdm, V_fdm, color=speed_fdm, 
                      cmap='jet', density=1.5, linewidth=1, arrowsize=1.5)
    _plot_cylinder(axes[0], cx, cy, radius)
    axes[0].set_title('FDM', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_xlim(config['x_domain'])
    axes[0].set_ylim(config['y_domain'])
    
    # Plot Standard PINN
    speed_std = np.sqrt(U_std**2 + V_std**2)
    axes[1].streamplot(X_plot, Y_plot, U_std, V_std, color=speed_std,
                      cmap='jet', density=1.5, linewidth=1, arrowsize=1.5)
    _plot_cylinder(axes[1], cx, cy, radius)
    axes[1].set_title('Standard PINN', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_xlim(config['x_domain'])
    axes[1].set_ylim(config['y_domain'])
    
    # Plot Curriculum PINN
    speed_curr = np.sqrt(U_curr**2 + V_curr**2)
    axes[2].streamplot(X_plot, Y_plot, U_curr, V_curr, color=speed_curr,
                      cmap='jet', density=1.5, linewidth=1, arrowsize=1.5)
    _plot_cylinder(axes[2], cx, cy, radius)
    axes[2].set_title('Curriculum PINN', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_xlim(config['x_domain'])
    axes[2].set_ylim(config['y_domain'])
    
    plt.tight_layout()
    
    # Save
    filepath = os.path.join(output_dir, 'streamlines.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Streamlines plot saved to {filepath}")
    
    # plt.show()


def print_cylinder_metrics(fdm_solver, pinn_standard, pinn_curriculum, config):
    """Print quantitative error metrics"""
    print("\n" + "="*80)
    print("QUANTITATIVE ERROR ANALYSIS FOR CYLINDER FLOW")
    print("="*80)
    
    # Prepare evaluation grid
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nx_eval, ny_eval = 100, 50
    x_eval = np.linspace(config['x_domain'][0], config['x_domain'][1], nx_eval)
    y_eval = np.linspace(config['y_domain'][0], config['y_domain'][1], ny_eval)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval)
    
    X_flat = np.c_[X_eval.ravel(), Y_eval.ravel()]
    X_tensor = torch.tensor(X_flat, dtype=torch.float32).to(device)
    
    # Get predictions
    UVP_std = pinn_standard.predict(X_tensor)
    UVP_curr = pinn_curriculum.predict(X_tensor)
    
    # Interpolate FDM
    from scipy.interpolate import RegularGridInterpolator
    u_fdm_interp = RegularGridInterpolator((fdm_solver.y, fdm_solver.x), fdm_solver.u)
    v_fdm_interp = RegularGridInterpolator((fdm_solver.y, fdm_solver.x), fdm_solver.v)
    
    points = np.c_[Y_eval.ravel(), X_eval.ravel()]
    U_fdm = u_fdm_interp(points)
    V_fdm = v_fdm_interp(points)
    
    # Compute errors
    u_error_std = np.abs(UVP_std[:, 0] - U_fdm)
    v_error_std = np.abs(UVP_std[:, 1] - V_fdm)
    u_error_curr = np.abs(UVP_curr[:, 0] - U_fdm)
    v_error_curr = np.abs(UVP_curr[:, 1] - V_fdm)
    
    # Standard PINN
    print("\n--- Standard PINN ---")
    print(f"u L2 Error: {np.linalg.norm(u_error_std) / np.linalg.norm(U_fdm):.6e}")
    print(f"v L2 Error: {np.linalg.norm(v_error_std) / np.linalg.norm(V_fdm):.6e}")
    print(f"u L∞ Error: {np.max(u_error_std):.6e}")
    print(f"v L∞ Error: {np.max(v_error_std):.6e}")
    
    # Curriculum PINN
    print("\n--- Curriculum PINN ---")
    print(f"u L2 Error: {np.linalg.norm(u_error_curr) / np.linalg.norm(U_fdm):.6e}")
    print(f"v L2 Error: {np.linalg.norm(v_error_curr) / np.linalg.norm(V_fdm):.6e}")
    print(f"u L∞ Error: {np.max(u_error_curr):.6e}")
    print(f"v L∞ Error: {np.max(v_error_curr):.6e}")
    
    print("="*80 + "\n")


def _plot_cylinder(ax, cx, cy, radius):
    """Helper function to plot cylinder"""
    circle = plt.Circle((cx, cy), radius, color='white', ec='black', linewidth=2, zorder=10)
    ax.add_patch(circle)


def create_all_cylinder_plots(fdm_solver, pinn_standard, pinn_curriculum,
                              config, output_dir='outputs/cylinder'):
    """Generate all plots for cylinder flow"""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS FOR CYLINDER FLOW")
    print("="*80 + "\n")
    
    plot_cylinder_comparison(fdm_solver, pinn_standard, pinn_curriculum, config, output_dir)
    plot_cylinder_streamlines(fdm_solver, pinn_standard, pinn_curriculum, config, output_dir)
    print_cylinder_metrics(fdm_solver, pinn_standard, pinn_curriculum, config)
    
    print("✓ All visualizations completed!\n")