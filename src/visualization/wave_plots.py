"""
Visualization utilities for Wave equation experiments
"""
import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_wave_comparison(results, output_dir='outputs/wave'):
    """Comprehensive comparison plot for Wave equation"""
    config = results['config']
    c = config['c']
    k = config['k']
    t_snapshots = config.get('plot_t_snapshots', [0.0, 0.5, 1.0, 1.5, 2.0])
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'PINN Performance for 1D Wave Equation (c={c}, k={k})',
                 fontsize=16, fontweight='bold')
    
    # --- 1. Loss Curve Comparison ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(results['standard']['solver'].loss_history,
            label=results['standard']['label'], alpha=0.7, linewidth=2)
    ax1.plot(results['curriculum']['solver'].loss_history,
            label=results['curriculum']['label'], linewidth=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Total Loss', fontsize=12)
    ax1.set_title('Loss Curve Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    # --- 2. Wave Speed Evolution (Curriculum) ---
    ax2 = fig.add_subplot(gs[0, 1])
    curriculum_solver = results['curriculum']['solver']
    
    if hasattr(curriculum_solver, 'c_target') and hasattr(curriculum_solver, 'ramp_epochs'):
        epochs = np.arange(curriculum_solver.total_epochs)
        c_history = []
        
        for epoch in epochs:
            if epoch < curriculum_solver.ramp_epochs:
                progress = epoch / curriculum_solver.ramp_epochs
                c_curr = curriculum_solver.c_target * 0.2 * (1 - progress) + \
                         curriculum_solver.c_target * progress
            else:
                c_curr = curriculum_solver.c_target
            c_history.append(c_curr)
        
        ax2.plot(epochs, c_history, color='green', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Wave Speed (c)', fontsize=12)
        ax2.set_title('Curriculum: Wave Speed Annealing', fontsize=13, fontweight='bold')
        ax2.grid(True, ls="--", alpha=0.5)
        ax2.axhline(y=curriculum_solver.c_target, color='red',
                   linestyle='--', alpha=0.5, label=f'Target c={curriculum_solver.c_target:.2f}')
        ax2.axhline(y=curriculum_solver.c_target * 0.2, color='blue',
                   linestyle='--', alpha=0.5, label=f'Initial c={curriculum_solver.c_target * 0.2:.2f}')
        ax2.legend(fontsize=9)
        ax2.axvline(x=curriculum_solver.ramp_epochs, color='orange',
                   linestyle=':', alpha=0.7, linewidth=1.5)
    
    # --- 3. Space-time contour (Exact) ---
    ax3 = fig.add_subplot(gs[0, 2])
    exact_data = results['exact']
    im = ax3.contourf(exact_data['x'], exact_data['t'], exact_data['solution'].T,
                     levels=20, cmap='RdBu_r')
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('t', fontsize=12)
    ax3.set_title('Exact Solution (Analytical/FDM)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax3)
    
    # --- 4-8. Snapshots at different times ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_test = torch.linspace(config['x_domain'][0], config['x_domain'][1],
                           config.get('plot_x_points', 512)).view(-1, 1)
    
    for idx, t_snap in enumerate(t_snapshots):
        row = 1 + idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        # Get exact solution at this time
        t_idx = np.argmin(np.abs(exact_data['t'] - t_snap))
        u_exact = exact_data['solution'][:, t_idx]
        
        # Get PINN predictions
        t_test = torch.full_like(x_test, t_snap)
        X_test = torch.cat([x_test, t_test], dim=1).to(device)
        
        u_std = results['standard']['solver'].predict(X_test)
        u_curr = results['curriculum']['solver'].predict(X_test)
        
        # Plot
        ax.plot(exact_data['x'], u_exact, 'k-',
               label='Exact', linewidth=3, alpha=0.8)
        ax.plot(x_test.cpu().numpy(), u_std, 'b--',
               label='Standard PINN', linewidth=2)
        ax.plot(x_test.cpu().numpy(), u_curr, 'r-.',
               label='Curriculum PINN', linewidth=2)
        
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel(f'u(x, t={t_snap:.2f})', fontsize=11)
        ax.set_title(f't = {t_snap:.2f}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, ls="--", alpha=0.5)
    
    plt.tight_layout()
    
    # Save
    import os
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'comprehensive_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to {filepath}")
    
    # plt.show()


def plot_wave_spacetime(results, output_dir='outputs/wave'):
    """Plot space-time contour of solutions"""
    config = results['config']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create space-time grid
    nx, nt = 200, 200
    x = np.linspace(config['x_domain'][0], config['x_domain'][1], nx)
    t = np.linspace(config['t_domain'][0], config['t_domain'][1], nt)
    X_grid, T_grid = np.meshgrid(x, t)
    
    # Prepare input
    X_flat = np.c_[X_grid.ravel(), T_grid.ravel()]
    X_tensor = torch.tensor(X_flat, dtype=torch.float32).to(device)
    
    # Get predictions
    u_std = results['standard']['solver'].predict(X_tensor).reshape(nt, nx)
    u_curr = results['curriculum']['solver'].predict(X_tensor).reshape(nt, nx)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    # Exact
    exact_data = results['exact']
    im0 = axes[0].contourf(exact_data['x'], exact_data['t'],
                           exact_data['solution'].T, levels=20, cmap='RdBu_r')
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('t', fontsize=12)
    axes[0].set_title('Exact Solution', fontsize=13, fontweight='bold')
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
    
    # plt.show()


def print_wave_metrics(results):
    """Print quantitative metrics"""
    config = results['config']
    t_final = config.get('plot_t_snapshots', [2.0])[-1]
    
    print("\n" + "="*80)
    print("QUANTITATIVE ERROR ANALYSIS (Wave Equation)")
    print("="*80)
    
    # Extract exact solution
    exact_data = results['exact']
    t_idx = np.argmin(np.abs(exact_data['t'] - t_final))
    u_exact = exact_data['solution'][:, t_idx]
    
    # Get predictions
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
    
    # Metrics
    print(f"\nAt t = {t_final}:")
    print("\n--- Standard PINN ---")
    l2_std = np.linalg.norm(error_std) / np.linalg.norm(u_exact)
    linf_std = np.max(np.abs(error_std))
    print(f"L2 Relative Error: {l2_std:.6e}")
    print(f"L∞ Absolute Error: {linf_std:.6e}")
    
    print("\n--- Curriculum PINN ---")
    l2_curr = np.linalg.norm(error_curr) / np.linalg.norm(u_exact)
    linf_curr = np.max(np.abs(error_curr))
    print(f"L2 Relative Error: {l2_curr:.6e}")
    print(f"L∞ Absolute Error: {linf_curr:.6e}")
    
    print("\n--- Improvement ---")
    improvement_l2 = (l2_std - l2_curr) / l2_std * 100
    improvement_linf = (linf_std - linf_curr) / linf_std * 100
    print(f"L2 Error Reduction: {improvement_l2:+.2f}%")
    print(f"L∞ Error Reduction: {improvement_linf:+.2f}%")
    
    print("="*80 + "\n")


def create_all_wave_plots(results, output_dir='outputs/wave'):
    """Generate all plots for Wave equation"""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATION FOR WAVE EQUATION")
    print("="*80 + "\n")
    
    plot_wave_comparison(results, output_dir)
    plot_wave_spacetime(results, output_dir)
    print_wave_metrics(results)
    
    print("✓ All visualizations completed!\n")