# 文件名: src/visualization/loss_landscape_generic.py

"""
Generic Loss Landscape Visualization for PINNs

This module provides a unified interface for loss landscape visualization
that works with any PINN solver.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import os
import copy

try:
    import loss_landscapes
    import loss_landscapes.metrics
    LOSS_LANDSCAPES_AVAILABLE = True
except ImportError:
    LOSS_LANDSCAPES_AVAILABLE = False
    print("⚠ Warning: loss_landscapes not installed. Install with: pip install loss-landscapes")


class LossLandscapeVisualizer:
    """
    Generic loss landscape visualizer for any PINN solver.
    
    Features:
    - Works with any PDE (Burgers, Wave, Navier-Stokes, etc.)
    - Supports both standard and curriculum learning
    - Generates 2D and 3D visualizations
    - Tracks training trajectories
    - Handles multi-stage curriculum learning
    """
    
    def __init__(self, output_dir='outputs', seed=42):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save plots
            seed: Random seed for reproducibility
        """
        self.output_dir = output_dir
        self.seed = seed
        os.makedirs(output_dir, exist_ok=True)
        
        if not LOSS_LANDSCAPES_AVAILABLE:
            raise ImportError("loss-landscapes library is required. Install with: pip install loss-landscapes")
    
    @staticmethod
    def get_flat_params(model):
        """Get model parameters as a flat numpy array."""
        params = []
        for param in model.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)
    
    @staticmethod
    def generate_random_directions(model, seed=42):
        """Generate two random orthonormal directions in parameter space."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        params = LossLandscapeVisualizer.get_flat_params(model)
        n_params = len(params)
        
        # Generate random directions
        direction1 = np.random.randn(n_params)
        direction2 = np.random.randn(n_params)
        
        # Gram-Schmidt orthogonalization
        direction1 = direction1 / np.linalg.norm(direction1)
        direction2 = direction2 - np.dot(direction2, direction1) * direction1
        direction2 = direction2 / np.linalg.norm(direction2)
        
        return direction1, direction2
    
    @staticmethod
    def project_trajectory_to_plane(trajectory_states, final_model, direction1, direction2, model_class, model_kwargs):
        """
        Project the parameter trajectory onto a 2D plane.
        
        Args:
            trajectory_states: List of model state_dicts
            final_model: The final trained model (center of projection)
            direction1, direction2: Basis vectors for the plane
            model_class: Class to instantiate temporary models
            model_kwargs: Kwargs for model initialization
        
        Returns:
            np.array of (x, y) coordinates
        """
        final_params = LossLandscapeVisualizer.get_flat_params(final_model)
        coords = []
        
        for state in trajectory_states:
            # Create a temporary model and load the state
            temp_model = model_class(**model_kwargs)
            temp_model.load_state_dict(state)
            temp_params = LossLandscapeVisualizer.get_flat_params(temp_model)
            
            # Get displacement from final model
            displacement = temp_params - final_params
            
            # Project onto the two directions
            x = np.dot(displacement, direction1)
            y = np.dot(displacement, direction2)
            
            coords.append([x, y])
        
        return np.array(coords)
    
    def compute_loss_landscape(self, model, X_eval, u_target, distance=1.0, steps=40):
        """
        Compute loss landscape around a model.
        
        Args:
            model: The model to evaluate
            X_eval: Input data for evaluation
            u_target: Target output for evaluation
            distance: How far to explore in parameter space
            steps: Resolution of the landscape grid
        
        Returns:
            2D numpy array of loss values
        """
        print(f"    Computing loss landscape (steps={steps})...")
        
        # Ensure data is on CPU
        X_eval = X_eval.cpu()
        u_target = u_target.cpu()
        model = model.cpu()
        
        # Define loss metric
        criterion = torch.nn.MSELoss()
        metric = loss_landscapes.metrics.Loss(criterion, X_eval, u_target)
        
        # Compute landscape
        landscape = loss_landscapes.random_plane(
            model=model,
            metric=metric,
            distance=distance,
            steps=steps,
            normalization='filter',
            deepcopy_model=True
        )
        
        return landscape
    
    def plot_trajectory_2d(self, ax, trajectory_coords, trajectory_epochs, 
                          color='blue', label='Training Path', 
                          stage_boundaries=None, stage_names=None):
        """
        Plot 2D training trajectory on loss landscape.
        
        Args:
            ax: Matplotlib axis
            trajectory_coords: Array of (x, y) coordinates
            trajectory_epochs: List of epoch numbers
            color: Trajectory color
            label: Trajectory label
            stage_boundaries: List of epoch numbers where stages change (for curriculum)
            stage_names: List of stage names
        """
        if len(trajectory_coords) < 2:
            return
        
        trajectory_coords = np.array(trajectory_coords)
        
        # Plot trajectory line
        ax.plot(trajectory_coords[:, 0], trajectory_coords[:, 1], 
               color=color, linewidth=2.5, alpha=0.8, label=label, zorder=5)
        
        # Mark start
        ax.scatter(trajectory_coords[0, 0], trajectory_coords[0, 1], 
                  c='green', s=250, marker='^', edgecolors='darkgreen', 
                  linewidths=2.5, label='Start', zorder=10)
        
        # Mark end
        ax.scatter(trajectory_coords[-1, 0], trajectory_coords[-1, 1], 
                  c='red', s=250, marker='X', edgecolors='darkred', 
                  linewidths=2.5, label='Final Model Parameters', zorder=10)
        
        # Mark stage boundaries (for curriculum learning)
        if stage_boundaries is not None and len(stage_boundaries) > 0:
            for stage_idx, epoch in enumerate(stage_boundaries):
                # Find closest trajectory point
                if len(trajectory_epochs) == 0:
                    continue
                    
                idx = min(range(len(trajectory_epochs)), 
                         key=lambda i: abs(trajectory_epochs[i] - epoch))
                
                if idx < len(trajectory_coords):
                    ax.scatter(trajectory_coords[idx, 0], trajectory_coords[idx, 1],
                              c='yellow', s=200, marker='s', edgecolors='orange',
                              linewidths=2, zorder=9)
                    
                    # Add annotation
                    stage_label = stage_names[stage_idx] if stage_names and stage_idx < len(stage_names) else f'Stage {stage_idx+1}'
                    ax.annotate(stage_label, 
                              xy=(trajectory_coords[idx, 0], trajectory_coords[idx, 1]),
                              xytext=(12, 12), textcoords='offset points',
                              fontsize=9, color='orange', fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                       edgecolor='orange', alpha=0.8))
        
        # Add direction arrows
        n_arrows = min(10, len(trajectory_coords) - 1)
        if n_arrows > 0:
            arrow_indices = np.linspace(0, len(trajectory_coords) - 2, n_arrows, dtype=int)
            
            for idx in arrow_indices:
                dx = trajectory_coords[idx + 1, 0] - trajectory_coords[idx, 0]
                dy = trajectory_coords[idx + 1, 1] - trajectory_coords[idx, 1]
                
                distance = np.sqrt(dx**2 + dy**2)
                if distance > 1e-4:  # Only draw if movement is significant
                    ax.arrow(trajectory_coords[idx, 0], trajectory_coords[idx, 1], 
                            dx * 0.4, dy * 0.4, 
                            head_width=0.04, head_length=0.04, 
                            fc=color, ec=color, alpha=0.6, zorder=8, 
                            length_includes_head=True)
    
    def create_2d_landscape_comparison(self, results_dict, X_eval, u_target,
                                      model_class, model_kwargs,
                                      filename='loss_landscape_2d.png', 
                                      steps=40, distance=1.0):
        """
        Create 2D loss landscape comparison plot.
        
        Args:
            results_dict: Dictionary with structure:
                {
                    'method_name': {
                        'model': trained model,
                        'trajectory_states': list of state_dicts,
                        'trajectory_epochs': list of epochs,
                        'color': trajectory color,
                        'stage_boundaries': list of epochs (optional),
                        'stage_names': list of names (optional)
                    },
                    ...
                }
            X_eval: Evaluation input tensor
            u_target: Evaluation target tensor
            model_class: Model class for trajectory projection
            model_kwargs: Model initialization kwargs
            filename: Output filename (can be full path)
            steps: Resolution of landscape grid
            distance: Exploration distance in parameter space
        """
        print("\n" + "="*80)
        print("GENERATING 2D LOSS LANDSCAPE COMPARISON")
        print("="*80)
        
        n_methods = len(results_dict)
        fig, axes = plt.subplots(1, n_methods, figsize=(10*n_methods, 9))
        
        if n_methods == 1:
            axes = [axes]
        
        fig.suptitle('Loss Landscape with Training Trajectories (2D)', 
                    fontsize=18, fontweight='bold')
        
        for idx, (method_name, method_data) in enumerate(results_dict.items()):
            ax = axes[idx]
            
            print(f"\n  Processing: {method_name}")
            
            model = method_data['model']
            trajectory_states = method_data.get('trajectory_states', [])
            trajectory_epochs = method_data.get('trajectory_epochs', [])
            color = method_data.get('color', 'cyan')
            stage_boundaries = method_data.get('stage_boundaries', None)
            stage_names = method_data.get('stage_names', None)
            
            # Compute loss landscape
            landscape = self.compute_loss_landscape(model, X_eval, u_target, distance, steps)
            
            # Plot landscape
            x_coords = np.linspace(-distance, distance, steps)
            y_coords = np.linspace(-distance, distance, steps)
            X, Y = np.meshgrid(x_coords, y_coords)
            
            landscape_safe = np.copy(landscape)
            landscape_safe[landscape_safe <= 0] = 1e-10
            Z = np.log(landscape_safe)  # Natural log for better visualization
            
            contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
            plt.colorbar(contour, ax=ax, label='Log(Loss)')
            
            # Project and plot trajectory
            if len(trajectory_states) > 0:
                print(f"    Projecting trajectory ({len(trajectory_states)} points)...")
                direction1, direction2 = self.generate_random_directions(model, seed=self.seed + idx)
                
                trajectory_2d = self.project_trajectory_to_plane(
                    trajectory_states, model, direction1, direction2,
                    model_class, model_kwargs
                )
                
                self.plot_trajectory_2d(
                    ax, trajectory_2d, trajectory_epochs, 
                    color=color, label='Training Path',
                    stage_boundaries=stage_boundaries,
                    stage_names=stage_names
                )
            else:
                # Just mark final position
                ax.plot([0], [0], 'rX', markersize=18, markeredgewidth=3, 
                       label='Final Model Parameters', zorder=15)
            
            ax.set_title(f'{method_name}', fontsize=16, fontweight='bold')
            ax.set_xlabel('Direction α', fontsize=13)
            ax.set_ylabel('Direction β', fontsize=13)
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Handle both full path and filename
        if os.path.dirname(filename):
            filepath = filename
        else:
            filepath = os.path.join(self.output_dir, filename)
            
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\n  ✓ 2D landscape saved to {filepath}")
        plt.close(fig)
    
    def create_3d_landscape_comparison(self, results_dict, X_eval, u_target,
                                      model_class, model_kwargs,
                                      filename='loss_landscape_3d.png', 
                                      steps=30, distance=1.0):
        """
        Create 3D loss landscape surface plots.
        
        Args:
            results_dict: Same structure as create_2d_landscape_comparison
            X_eval: Evaluation input tensor
            u_target: Evaluation target tensor
            model_class: Model class (not used in 3D but kept for consistency)
            model_kwargs: Model kwargs (not used in 3D but kept for consistency)
            filename: Output filename (can be full path)
            steps: Resolution of landscape grid
            distance: Exploration distance
        """
        print("\n" + "="*80)
        print("GENERATING 3D LOSS LANDSCAPE COMPARISON")
        print("="*80)
        
        n_methods = len(results_dict)
        fig = plt.figure(figsize=(10*n_methods, 8))
        
        fig.suptitle('Loss Landscape with Training Trajectories (3D)', 
                    fontsize=18, fontweight='bold')
        
        for idx, (method_name, method_data) in enumerate(results_dict.items()):
            ax = fig.add_subplot(1, n_methods, idx + 1, projection='3d')
            
            print(f"\n  Processing: {method_name}")
            
            model = method_data['model']
            
            # Compute loss landscape
            landscape = self.compute_loss_landscape(model, X_eval, u_target, distance, steps)
            
            # Plot 3D surface
            x_coords = np.linspace(-distance, distance, steps)
            y_coords = np.linspace(-distance, distance, steps)
            X, Y = np.meshgrid(x_coords, y_coords)
            
            landscape_safe = np.copy(landscape)
            landscape_safe[landscape_safe <= 0] = 1e-10
            Z = np.log(landscape_safe)
            
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.85, 
                                  edgecolor='none', antialiased=True)
            fig.colorbar(surf, ax=ax, shrink=0.5, label='Log(Loss)', pad=0.1)
            
            # Mark final model
            criterion = torch.nn.MSELoss()
            with torch.no_grad():
                u_pred = model(X_eval)
                z_center = np.log(criterion(u_pred, u_target).item() + 1e-10)
            
            ax.scatter([0], [0], [z_center], c='red', s=250, marker='X',
                      edgecolors='darkred', linewidths=2.5, 
                      label='Final Model Parameters', zorder=10)
            
            ax.set_title(f'{method_name}', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Direction α', fontsize=12, labelpad=10)
            ax.set_ylabel('Direction β', fontsize=12, labelpad=10)
            ax.set_zlabel('Log(Loss)', fontsize=12, labelpad=10)
            ax.view_init(elev=25, azim=45)
            ax.legend(loc='upper left', fontsize=10)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Handle both full path and filename
        if os.path.dirname(filename):
            filepath = filename
        else:
            filepath = os.path.join(self.output_dir, filename)
            
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\n  ✓ 3D landscape saved to {filepath}")
        plt.close(fig)
    
    def _infer_pde_type(self, model):
        """Infer PDE type from model architecture (heuristic)."""
        # This is a simple heuristic - you can make it more sophisticated
        n_params = sum(p.numel() for p in model.parameters())
        
        if n_params < 10000:
            return "PDE Problem"
        else:
            return "Complex PDE Problem"
    
    @staticmethod
    def extract_stage_info(solver):
        """
        Extract curriculum stage information from a solver.
        
        Args:
            solver: A PINN solver object
        
        Returns:
            (stage_boundaries, stage_names) or (None, None) if not curriculum
        """
        if not hasattr(solver, 'curriculum_stages'):
            return None, None
        
        stages = solver.curriculum_stages
        boundaries = []
        names = []
        cumulative_epochs = 0
        
        for stage in stages:
            boundaries.append(cumulative_epochs)
            names.append(stage.get('name', f"Stage {len(names)+1}"))
            cumulative_epochs += stage['epochs']
        
        return boundaries[1:], names[1:]  # Skip first boundary (epoch 0)


# ==================== Helper Function for Easy Integration ====================

def visualize_loss_landscape(solvers_dict, eval_data, output_dir='outputs', 
                             pde_name='PDE', seed=42, steps_2d=40, steps_3d=30):
    """
    Convenient wrapper function to visualize loss landscapes.
    
    Args:
        solvers_dict: Dictionary of trained solvers
            {
                'method_name': {
                    'solver': trained PINN solver object,
                    'label': display label (optional),
                    'color': trajectory color (optional)
                }
            }
        eval_data: Tuple of (X_eval, u_target) for loss computation
        output_dir: Output directory
        pde_name: Name of the PDE for filename
        seed: Random seed
        steps_2d: Resolution for 2D plots
        steps_3d: Resolution for 3D plots
    
    Example:
        ```python
        solvers = {
            'standard': {'solver': pinn_std, 'label': 'Standard PINN', 'color': 'yellow'},
            'curriculum': {'solver': pinn_curr, 'label': 'Curriculum PINN', 'color': 'cyan'}
        }
        
        X_eval = torch.randn(1000, 2)
        u_target = torch.randn(1000, 1)
        
        visualize_loss_landscape(solvers, (X_eval, u_target), 
                                output_dir='outputs/wave_2d',
                                pde_name='Wave2D')
        ```
    """
    if not LOSS_LANDSCAPES_AVAILABLE:
        print("⚠ Loss landscape visualization skipped (library not installed)")
        return
    
    visualizer = LossLandscapeVisualizer(output_dir=output_dir, seed=seed)
    
    # Convert solver dict to results dict
    results_dict = {}
    
    for method_name, method_info in solvers_dict.items():
        solver = method_info['solver']
        
        # Extract model info
        model_class = type(solver.model)
        model_kwargs = {
            'layers': getattr(solver.model, 'layers', [3, 64, 64, 1]),
            'activation': getattr(solver.config, 'activation', 'tanh')
        }
        
        # Extract curriculum info if available
        stage_boundaries, stage_names = LossLandscapeVisualizer.extract_stage_info(solver)
        
        results_dict[method_name] = {
            'model': solver.model,
            'trajectory_states': getattr(solver, 'trajectory_states', []),
            'trajectory_epochs': getattr(solver, 'trajectory_epochs', []),
            'eval_data': eval_data,
            'label': method_info.get('label', method_name),
            'color': method_info.get('color', 'blue'),
            'stage_boundaries': stage_boundaries,
            'stage_names': stage_names,
            'model_class': model_class,
            'model_kwargs': model_kwargs
        }
    
    # Generate visualizations
    X_eval, u_target = eval_data
    
    visualizer.create_2d_landscape_comparison(
        results_dict=results_dict,
        X_eval=X_eval,
        u_target=u_target,
        model_class=list(results_dict.values())[0]['model_class'],
        model_kwargs=list(results_dict.values())[0]['model_kwargs'],
        filename=f'loss_landscape_2d_{pde_name.lower()}.png',
        steps=steps_2d
    )
    
    visualizer.create_3d_landscape_comparison(
        results_dict=results_dict,
        X_eval=X_eval,
        u_target=u_target,
        model_class=list(results_dict.values())[0]['model_class'],
        model_kwargs=list(results_dict.values())[0]['model_kwargs'],
        filename=f'loss_landscape_3d_{pde_name.lower()}.png',
        steps=steps_3d
    )
    
    print("\n" + "="*80)
    print("LOSS LANDSCAPE VISUALIZATION COMPLETED")
    print("="*80 + "\n")