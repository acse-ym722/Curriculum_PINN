# 文件名: src/visualization/wave_plots_2d.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os

def save_loss_to_csv(results, output_dir):
    """Saves the loss history of standard and curriculum PINNs to a CSV file."""
    std_loss = results['standard']['solver'].loss_history
    curr_loss = results['curriculum']['solver'].loss_history
    
    # 确保两个列表等长，方便写入DataFrame
    max_len = max(len(std_loss), len(curr_loss))
    epochs = list(range(1, max_len + 1))
    
    std_loss.extend([np.nan] * (max_len - len(std_loss)))
    curr_loss.extend([np.nan] * (max_len - len(curr_loss)))
    
    df = pd.DataFrame({
        'epoch': epochs,
        'standard_pinn_loss': std_loss,
        'curriculum_pinn_loss': curr_loss
    })
    
    filepath = os.path.join(output_dir, 'loss_history.csv')
    df.to_csv(filepath, index=False)
    print(f"✓ Loss history saved to {filepath}")

def save_raw_data(results, output_dir):
    """Saves the raw data used for plotting (FDM solution, PINN predictions)."""
    config = results['config']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 网格
    exact_data = results['exact']
    x_grid, y_grid, t_grid = exact_data['x'], exact_data['y'], exact_data['t']
    u_exact = exact_data['solution']
    
    # 在FDM网格上获取PINN预测
    nx, ny, nt = u_exact.shape
    X, Y, T = np.meshgrid(x_grid, y_grid, t_grid, indexing='ij')
    
    grid_tensor = torch.tensor(np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=-1), 
                               dtype=torch.float32).to(device)
    
    print("Generating PINN predictions on the full grid for data saving...")
    u_std_pred = results['standard']['solver'].predict(grid_tensor).reshape(nx, ny, nt)
    u_curr_pred = results['curriculum']['solver'].predict(grid_tensor).reshape(nx, ny, nt)
    print("✓ Predictions generated.")

    filepath = os.path.join(output_dir, 'raw_solution_data.npz')
    np.savez(filepath,
             x=x_grid,
             y=y_grid,
             t=t_grid,
             u_exact=u_exact,
             u_standard_pinn=u_std_pred,
             u_curriculum_pinn=u_curr_pred)
    print(f"✓ Raw solution data saved to {filepath}")


def plot_wave_comparison_2d(results, output_dir='outputs/wave_2d'):
    """Comprehensive comparison plot for 2D Wave equation."""
    config = results['config']
    c = config['c']
    kx, ky = config['kx'], config['ky']
    t_snapshots = config.get('plot_t_snapshots', [0.0, 0.5, 1.0])
    
    num_snapshots = len(t_snapshots)
    fig = plt.figure(figsize=(20, 5 * num_snapshots))
    gs = fig.add_gridspec(num_snapshots, 3, hspace=0.4, wspace=0.3)
    
    fig.suptitle(f'PINN for 2D Wave Equation (c_target={c}, kx={kx}, ky={ky})',
                 fontsize=18, fontweight='bold')
    
    # --- Top Row: Loss, Curriculum, and Metrics ---
    # 1. Loss Curve Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(results['standard']['solver'].loss_history, label=results['standard']['label'], alpha=0.8)
    ax1.plot(results['curriculum']['solver'].loss_history, label=results['curriculum']['label'])
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Loss Curve Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    # 2. Wave Speed Evolution (Curriculum)
    ax2 = fig.add_subplot(gs[0, 1])
    curriculum_solver = results['curriculum']['solver']
    ax2.plot(curriculum_solver.c_history, color='green', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Wave Speed (c)')
    ax2.set_title('Curriculum: Wave Speed Evolution', fontweight='bold')
    ax2.grid(True, ls="--", alpha=0.5)
    
    # 3. Final Error Map
    ax3 = fig.add_subplot(gs[0, 2])
    exact_data = results['exact']
    t_final_idx = np.argmin(np.abs(exact_data['t'] - t_snapshots[-1]))
    u_exact_final = exact_data['solution'][:, :, t_final_idx]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_test = torch.linspace(config['x_domain'][0], config['x_domain'][1], config['plot_xy_points']).to(device)
    y_test = torch.linspace(config['y_domain'][0], config['y_domain'][1], config['plot_xy_points']).to(device)
    X_test, Y_test = torch.meshgrid(x_test, y_test, indexing='ij')
    T_test = torch.full_like(X_test, t_snapshots[-1])
    
    grid_test = torch.stack([X_test.ravel(), Y_test.ravel(), T_test.ravel()], dim=-1)
    u_curr_pred = curriculum_solver.predict(grid_test).reshape(X_test.shape)
    
    error_map = np.abs(u_curr_pred - u_exact_final)
    im = ax3.contourf(X_test.cpu(), Y_test.cpu(), error_map, levels=20, cmap='Reds')
    ax3.set_title(f'Curriculum PINN Abs Error at t={t_snapshots[-1]}', fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_aspect('equal', adjustable='box')
    plt.colorbar(im, ax=ax3, label='|u_pred - u_exact|')

    # --- Subsequent Rows: Snapshots at different times ---
    for idx, t_snap in enumerate(t_snapshots[1:], start=1):
        # Get exact solution at this time
        t_idx = np.argmin(np.abs(exact_data['t'] - t_snap))
        u_exact_snap = exact_data['solution'][:, :, t_idx]
        
        # Get PINN predictions
        T_snap_tensor = torch.full_like(X_test, t_snap)
        grid_snap = torch.stack([X_test.ravel(), Y_test.ravel(), T_snap_tensor.ravel()], dim=-1)
        
        u_std_snap = results['standard']['solver'].predict(grid_snap).reshape(X_test.shape)
        u_curr_snap = results['curriculum']['solver'].predict(grid_snap).reshape(X_test.shape)
        
        solutions = {
            'Exact (FDM)': u_exact_snap,
            'Standard PINN': u_std_snap,
            'Curriculum PINN': u_curr_snap
        }
        
        for col, (title, u_sol) in enumerate(solutions.items()):
            ax = fig.add_subplot(gs[idx, col])
            im = ax.contourf(X_test.cpu(), Y_test.cpu(), u_sol, levels=20, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title(f'{title} at t={t_snap:.2f}', fontweight='bold')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal', adjustable='box')
            if col == 2:
                plt.colorbar(im, ax=ax)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save
    filepath = os.path.join(output_dir, 'comprehensive_comparison_2d.png')
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    print(f"✓ Comparison plot saved to {filepath}")
    plt.close(fig)

def print_wave_metrics_2d(results):
    """Print quantitative metrics for 2D wave equation."""
    config = results['config']
    t_final = config.get('plot_t_snapshots', [1.0])[-1]
    
    print("\n" + "="*80)
    print("QUANTITATIVE ERROR ANALYSIS (2D Wave Equation)")
    print("="*80)
    
    exact_data = results['exact']
    t_idx = np.argmin(np.abs(exact_data['t'] - t_final))
    u_exact = exact_data['solution'][:, :, t_idx]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_test = torch.linspace(config['x_domain'][0], config['x_domain'][1], u_exact.shape[0]).to(device)
    y_test = torch.linspace(config['y_domain'][0], config['y_domain'][1], u_exact.shape[1]).to(device)
    X_test, Y_test = torch.meshgrid(x_test, y_test, indexing='ij')
    T_test = torch.full_like(X_test, t_final)
    
    grid_test = torch.stack([X_test.ravel(), Y_test.ravel(), T_test.ravel()], dim=-1)
    
    u_std = results['standard']['solver'].predict(grid_test).reshape(u_exact.shape)
    u_curr = results['curriculum']['solver'].predict(grid_test).reshape(u_exact.shape)
    
    error_std = u_std - u_exact
    error_curr = u_curr - u_exact
    
    print(f"\nMetrics at final time t = {t_final}:")
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
    if l2_std > 0:
        improvement_l2 = (l2_std - l2_curr) / l2_std * 100
        print(f"L2 Error Reduction: {improvement_l2:+.2f}%")
    if linf_std > 0:
        improvement_linf = (linf_std - linf_curr) / linf_std * 100
        print(f"L∞ Error Reduction: {improvement_linf:+.2f}%")
    
    print("="*80 + "\n")

def create_all_wave_plots_2d(results, output_dir='outputs/wave_2d'):
    """Generate all plots and save data for 2D Wave equation."""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS & SAVING DATA FOR 2D WAVE EQUATION")
    print("="*80 + "\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot comparisons
    plot_wave_comparison_2d(results, output_dir)
    
    # 2. Print quantitative metrics
    print_wave_metrics_2d(results)
    
    # 3. Save loss history to CSV
    save_loss_to_csv(results, output_dir)
    
    # 4. Save raw solution data
    save_raw_data(results, output_dir)
    
    print("✓ All visualizations and data saving completed!\n")