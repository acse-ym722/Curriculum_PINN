"""
Visualization utilities for Darcy equation experiments
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from scipy.interpolate import RegularGridInterpolator


def plot_darcy_comparison(results, output_dir='outputs/darcy_flow'):
    """
    Create comparison plots for FDM, Standard PINN, and Curriculum PINN
    
    Args:
        results: Dictionary containing:
            - 'fdm': FDM solution results
            - 'standard_pinn': Standard PINN results
            - 'curriculum_pinn': Curriculum PINN results
            - 'config': Configuration dictionary
            - 'stages': Curriculum stages (optional)
    """
    # Extract results
    fdm_result = results['fdm']
    std_result = results['standard_pinn']
    curr_result = results['curriculum_pinn']
    config = results.get('config', {})
    stages = results.get('stages', [])
    
    # Get domain info
    x_domain = config.get('x_domain', [0.0, 1.0])
    y_domain = config.get('y_domain', [0.0, 0.5])
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('2D Darcy Flow: Comparison of Methods', fontsize=16, fontweight='bold')
    
    # Determine color scale limits from all solutions
    vmin = min(
        fdm_result['p'].min(),
        std_result['p_pred'].min(),
        curr_result['p_pred'].min()
    )
    vmax = max(
        fdm_result['p'].max(),
        std_result['p_pred'].max(),
        curr_result['p_pred'].max()
    )
    
    # ========== Row 1: Pressure fields ==========
    
    # FDM Solution
    im1 = axes[0, 0].contourf(
        fdm_result['x'], fdm_result['y'], fdm_result['p'],
        levels=20, cmap='RdYlBu_r', vmin=vmin, vmax=vmax
    )
    axes[0, 0].set_title('FDM Solution (Ground Truth)', fontweight='bold')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, 0], label='Pressure')
    
    # Add source/sink markers
    if stages:
        for stage in stages:
            if 'source' in stage:
                src = stage['source']
                axes[0, 0].plot(src['x'], src['y'], 'r*', markersize=15, 
                              markeredgecolor='black', markeredgewidth=1.5,
                              label='Source' if not axes[0, 0].get_legend_handles_labels()[0] else '')
            if 'sink' in stage:
                sink = stage['sink']
                axes[0, 0].plot(sink['x'], sink['y'], 'bs', markersize=10,
                              markeredgecolor='black', markeredgewidth=1.5,
                              label='Sink' if 'Sink' not in str(axes[0, 0].get_legend_handles_labels()[1]) else '')
        
        if axes[0, 0].get_legend_handles_labels()[0]:
            axes[0, 0].legend(loc='upper right', fontsize=9)
    
    # Standard PINN
    im2 = axes[0, 1].contourf(
        std_result['x'], std_result['y'], std_result['p_pred'],
        levels=20, cmap='RdYlBu_r', vmin=vmin, vmax=vmax
    )
    axes[0, 1].set_title('Standard PINN', fontweight='bold')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0, 1], label='Pressure')
    
    # Curriculum PINN
    im3 = axes[0, 2].contourf(
        curr_result['x'], curr_result['y'], curr_result['p_pred'],
        levels=20, cmap='RdYlBu_r', vmin=vmin, vmax=vmax
    )
    axes[0, 2].set_title('Curriculum PINN', fontweight='bold')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    axes[0, 2].set_aspect('equal')
    plt.colorbar(im3, ax=axes[0, 2], label='Pressure')
    
    # ========== Row 2: Error fields ==========
    
    std_error = np.abs(std_result['p_pred'] - fdm_result['p'])
    curr_error = np.abs(curr_result['p_pred'] - fdm_result['p'])
    
    error_max = max(std_error.max(), curr_error.max())
    
    # Placeholder for first column
    axes[1, 0].text(0.5, 0.5, 'Reference\n(FDM Solution)', 
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   transform=axes[1, 0].transAxes)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].axis('off')
    
    # Standard PINN error
    im4 = axes[1, 1].contourf(
        std_result['x'], std_result['y'], std_error,
        levels=20, cmap='hot_r', vmin=0, vmax=error_max
    )
    axes[1, 1].set_title(f'Standard PINN Error\nMax: {std_error.max():.2e}', fontweight='bold')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_aspect('equal')
    plt.colorbar(im4, ax=axes[1, 1], label='|Error|')
    
    # Curriculum PINN error
    im5 = axes[1, 2].contourf(
        curr_result['x'], curr_result['y'], curr_error,
        levels=20, cmap='hot_r', vmin=0, vmax=error_max
    )
    axes[1, 2].set_title(f'Curriculum PINN Error\nMax: {curr_error.max():.2e}', fontweight='bold')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('y')
    axes[1, 2].set_aspect('equal')
    plt.colorbar(im5, ax=axes[1, 2], label='|Error|')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved comparison plot to {filepath}")


def plot_training_history(results, output_dir='outputs/darcy_flow'):
    """Plot training loss history comparison"""
    
    std_history = results['standard_pinn'].get('history', {})
    curr_history = results['curriculum_pinn'].get('history', {})
    
    if not std_history or not curr_history:
        print("  ⚠ Warning: No training history available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training History Comparison', fontsize=16, fontweight='bold')
    
    # Extract loss components
    std_total = std_history.get('total_loss', [])
    std_pde = std_history.get('pde_loss', [])
    std_bc = std_history.get('bc_loss', [])
    std_data = std_history.get('data_loss', [])
    
    curr_total = curr_history.get('total_loss', [])
    curr_pde = curr_history.get('pde_loss', [])
    curr_bc = curr_history.get('bc_loss', [])
    curr_data = curr_history.get('data_loss', [])
    
    # Total loss
    axes[0, 0].semilogy(std_total, 'b-', label='Standard PINN', linewidth=1.5, alpha=0.7)
    axes[0, 0].semilogy(curr_total, 'r-', label='Curriculum PINN', linewidth=1.5)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (log scale)')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # PDE loss
    axes[0, 1].semilogy(std_pde, 'b-', label='Standard PINN', linewidth=1.5, alpha=0.7)
    axes[0, 1].semilogy(curr_pde, 'r-', label='Curriculum PINN', linewidth=1.5)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss (log scale)')
    axes[0, 1].set_title('PDE Residual Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # BC loss
    axes[1, 0].semilogy(std_bc, 'b-', label='Standard PINN', linewidth=1.5, alpha=0.7)
    axes[1, 0].semilogy(curr_bc, 'r-', label='Curriculum PINN', linewidth=1.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss (log scale)')
    axes[1, 0].set_title('Boundary Condition Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Data loss
    axes[1, 1].semilogy(std_data, 'b-', label='Standard PINN', linewidth=1.5, alpha=0.7)
    axes[1, 1].semilogy(curr_data, 'r-', label='Curriculum PINN', linewidth=1.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss (log scale)')
    axes[1, 1].set_title('Data Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'training_history.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved training history to {filepath}")


def plot_curriculum_stages(results, output_dir='outputs/darcy_flow'):
    """Plot curriculum learning stages progression"""
    
    stages = results.get('stages', [])
    if not stages:
        print("  ⚠ Warning: No curriculum stages found")
        return
    
    config = results.get('config', {})
    x_domain = config.get('x_domain', [0.0, 1.0])
    y_domain = config.get('y_domain', [0.0, 0.5])
    
    n_stages = len(stages)
    fig, axes = plt.subplots(1, n_stages, figsize=(6 * n_stages, 5))
    
    if n_stages == 1:
        axes = [axes]
    
    fig.suptitle('Curriculum Learning Stages', fontsize=16, fontweight='bold')
    
    for idx, stage in enumerate(stages):
        ax = axes[idx]
        
        # Draw domain boundary
        ax.set_xlim(x_domain[0] - 0.05, x_domain[1] + 0.05)
        ax.set_ylim(y_domain[0] - 0.05, y_domain[1] + 0.05)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f"{stage.get('name', f'Stage {idx + 1}')}\n({stage.get('epochs', 0)} epochs)", 
                    fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Draw domain rectangle
        from matplotlib.patches import Rectangle
        rect = Rectangle(
            (x_domain[0], y_domain[0]),
            x_domain[1] - x_domain[0],
            y_domain[1] - y_domain[0],
            linewidth=2, edgecolor='black', facecolor='lightblue', alpha=0.2
        )
        ax.add_patch(rect)
        
        # Draw source
        if 'source' in stage:
            src = stage['source']
            ax.plot(src['x'], src['y'], 'r*', markersize=25, 
                   markeredgecolor='black', markeredgewidth=2,
                   label=f"Source\n(Q={src['strength']:.1f})", zorder=5)
            
            # Add text annotation
            ax.text(src['x'], src['y'] + 0.03, f"Q={src['strength']:.1f}",
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Draw sink
        if 'sink' in stage:
            sink = stage['sink']
            ax.plot(sink['x'], sink['y'], 'bs', markersize=18,
                   markeredgecolor='black', markeredgewidth=2,
                   label=f"Sink\n(Q={sink['strength']:.1f})", zorder=5)
            
            # Add text annotation
            ax.text(sink['x'], sink['y'] + 0.03, f"Q={sink['strength']:.1f}",
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'curriculum_stages.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved curriculum stages to {filepath}")


def plot_error_metrics(results, output_dir='outputs/darcy_flow'):
    """Plot error metrics comparison"""
    
    fdm_result = results['fdm']
    std_result = results['standard_pinn']
    curr_result = results['curriculum_pinn']
    
    # Compute errors
    std_error = np.abs(std_result['p_pred'] - fdm_result['p'])
    curr_error = np.abs(curr_result['p_pred'] - fdm_result['p'])
    
    # Compute metrics
    std_l2 = np.sqrt(np.mean(std_error**2))
    std_linf = np.max(std_error)
    std_mean = np.mean(std_error)
    
    curr_l2 = np.sqrt(np.mean(curr_error**2))
    curr_linf = np.max(curr_error)
    curr_mean = np.mean(curr_error)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['L2 Error', 'L∞ Error', 'Mean Error']
    std_values = [std_l2, std_linf, std_mean]
    curr_values = [curr_l2, curr_linf, curr_mean]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, std_values, width, label='Standard PINN', 
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, curr_values, width, label='Curriculum PINN',
                   color='coral', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2e}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Error Magnitude')
    ax.set_title('Error Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add improvement percentages
    improvements = [(std_values[i] - curr_values[i]) / std_values[i] * 100 
                   for i in range(len(metrics))]
    
    textstr = '\n'.join([
        f'{metrics[i]}: {improvements[i]:+.1f}% improvement'
        for i in range(len(metrics))
    ])
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'error_metrics.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved error metrics to {filepath}")


def print_quantitative_metrics(results):
    """Print detailed quantitative metrics"""
    
    fdm_result = results['fdm']
    std_result = results['standard_pinn']
    curr_result = results['curriculum_pinn']
    
    # Compute errors
    std_error = std_result['p_pred'] - fdm_result['p']
    curr_error = curr_result['p_pred'] - fdm_result['p']
    
    print("\n" + "="*80)
    print("QUANTITATIVE ERROR ANALYSIS")
    print("="*80)
    
    # Standard PINN
    print("\n--- Standard PINN ---")
    std_l2_rel = np.linalg.norm(std_error) / np.linalg.norm(fdm_result['p'])
    std_l2_abs = np.sqrt(np.mean(std_error**2))
    std_linf = np.max(np.abs(std_error))
    std_mean = np.mean(np.abs(std_error))
    
    print(f"  L2 Relative Error: {std_l2_rel:.6e}")
    print(f"  L2 Absolute Error: {std_l2_abs:.6e}")
    print(f"  L∞ Error:          {std_linf:.6e}")
    print(f"  Mean Absolute Err: {std_mean:.6e}")
    
    # Curriculum PINN
    print("\n--- Curriculum PINN ---")
    curr_l2_rel = np.linalg.norm(curr_error) / np.linalg.norm(fdm_result['p'])
    curr_l2_abs = np.sqrt(np.mean(curr_error**2))
    curr_linf = np.max(np.abs(curr_error))
    curr_mean = np.mean(np.abs(curr_error))
    
    print(f"  L2 Relative Error: {curr_l2_rel:.6e}")
    print(f"  L2 Absolute Error: {curr_l2_abs:.6e}")
    print(f"  L∞ Error:          {curr_linf:.6e}")
    print(f"  Mean Absolute Err: {curr_mean:.6e}")
    
    # Improvement
    print("\n--- Improvement by Curriculum Learning ---")
    imp_l2_rel = (std_l2_rel - curr_l2_rel) / std_l2_rel * 100
    imp_l2_abs = (std_l2_abs - curr_l2_abs) / std_l2_abs * 100
    imp_linf = (std_linf - curr_linf) / std_linf * 100
    imp_mean = (std_mean - curr_mean) / std_mean * 100
    
    print(f"  L2 Relative Error: {imp_l2_rel:+.2f}%")
    print(f"  L2 Absolute Error: {imp_l2_abs:+.2f}%")
    print(f"  L∞ Error:          {imp_linf:+.2f}%")
    print(f"  Mean Absolute Err: {imp_mean:+.2f}%")
    
    print("="*80 + "\n")


def create_all_darcy_plots(results, output_dir='outputs/darcy_flow'):
    """Generate all visualization plots for Darcy experiment"""
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATION FOR 2D DARCY FLOW")
    print("="*80 + "\n")
    
    # Create all plots
    plot_darcy_comparison(results, output_dir)
    plot_training_history(results, output_dir)
    plot_curriculum_stages(results, output_dir)
    plot_error_metrics(results, output_dir)
    
    # Print quantitative metrics
    print_quantitative_metrics(results)
    
    print("="*80)
    print("✓ All visualizations completed!")
    print("="*80 + "\n")