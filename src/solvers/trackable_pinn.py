# 文件名: src/solvers/trackable_pinn.py

"""
Trackable PINN Mixin

Adds trajectory tracking capability to any PINN solver.
"""

import copy
import time


class TrackableMixin:
    """
    Mixin class to add trajectory tracking to any PINN solver.
    
    Usage:
        class MyTrackablePINN(TrackableMixin, MyBasePINN):
            pass
    """
    
    def __init__(self, *args, track_interval=100, **kwargs):
        """
        Args:
            track_interval: How often to record model state (in epochs)
        """
        self.track_interval = track_interval
        self.trajectory_states = []
        self.trajectory_epochs = []
        super().__init__(*args, **kwargs)
        
        print(f"  ✓ Trajectory tracking enabled (interval={track_interval})")
    
    def _record_trajectory(self, epoch):
        """Record model state at specified intervals."""
        if epoch % self.track_interval == 0 or epoch == 0:
            self.trajectory_states.append(copy.deepcopy(self.model.state_dict()))
            self.trajectory_epochs.append(epoch)
    
    def train(self, verbose=True, save_interval=1000):
        """
        Override train method to add trajectory tracking.
        
        This method wraps the parent's train method.
        """
        # Get total epochs
        total_epochs = getattr(self, 'total_epochs', getattr(self, 'epochs', 10000))
        
        print("\n" + "="*80)
        print(f"Training {self.__class__.__name__} with Trajectory Tracking")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Total epochs: {total_epochs}")
        print(f"Track interval: {self.track_interval}")
        print("-"*80)
        
        start_time = time.time()
        
        # Record initial state
        self._record_trajectory(0)
        
        # Check if this is a curriculum solver
        is_curriculum = hasattr(self, 'curriculum_stages')
        
        if is_curriculum:
            # Curriculum training loop
            global_epoch = 0
            
            for stage_idx, stage in enumerate(self.curriculum_stages):
                stage_epochs = stage['epochs']
                
                print(f"\n{'='*25} STAGE {stage_idx + 1}/{len(self.curriculum_stages)}: "
                      f"{stage.get('name', 'Stage ' + str(stage_idx+1))} {'='*25}")
                print(f"Epochs in this stage: {stage_epochs}\n")
                
                for epoch_in_stage in range(stage_epochs):
                    # Update curriculum parameter if method exists
                    if hasattr(self, '_update_curriculum_stage'):
                        self._update_curriculum_stage(stage_idx, epoch_in_stage, stage_epochs)
                    elif hasattr(self, 'update_curriculum'):
                        self.update_curriculum(global_epoch)
                    
                    # Standard training step
                    loss, loss_dict = self.train_step()
                    
                    # Record history
                    self.loss_history.append(loss.item())
                    for key, value in loss_dict.items():
                        if key not in self.loss_components_history:
                            self.loss_components_history[key] = []
                        self.loss_components_history[key].append(value)
                    
                    # Record trajectory
                    self._record_trajectory(global_epoch + 1)
                    
                    # Verbose output
                    if verbose and ((global_epoch + 1) % save_interval == 0 or 
                                   epoch_in_stage == stage_epochs - 1):
                        elapsed = time.time() - start_time
                        lr = self.optimizer.param_groups[0]['lr']
                        
                        loss_str = f"Loss = {loss_dict.get('total', loss.item()):.4e}"
                        for key in ['ic', 'bc', 'pde', 'data']:
                            if key in loss_dict:
                                loss_str += f", {key.upper()} = {loss_dict[key]:.2e}"
                        
                        print(f"Epoch {global_epoch+1:6d}/{total_epochs} "
                              f"[Stage {stage_idx+1}, {epoch_in_stage+1:5d}/{stage_epochs}]: "
                              f"{loss_str}, LR = {lr:.2e}, Time = {elapsed:.1f}s")
                    
                    global_epoch += 1
            
            # Record final state
            if total_epochs not in self.trajectory_epochs:
                self._record_trajectory(total_epochs)
        
        else:
            # Standard training loop
            for epoch in range(total_epochs):
                # Standard training step
                loss, loss_dict = self.train_step()
                
                # Record history
                self.loss_history.append(loss.item())
                for key, value in loss_dict.items():
                    if key not in self.loss_components_history:
                        self.loss_components_history[key] = []
                    self.loss_components_history[key].append(value)
                
                # Record trajectory
                self._record_trajectory(epoch + 1)
                
                # Verbose output
                if verbose and ((epoch + 1) % save_interval == 0 or epoch == total_epochs - 1):
                    elapsed = time.time() - start_time
                    lr = self.optimizer.param_groups[0]['lr']
                    
                    loss_str = f"Loss = {loss_dict.get('total', loss.item()):.4e}"
                    for key in ['ic', 'bc', 'pde', 'data']:
                        if key in loss_dict:
                            loss_str += f", {key.upper()} = {loss_dict[key]:.2e}"
                    
                    print(f"Epoch {epoch+1:6d}/{total_epochs}: {loss_str}, "
                          f"LR = {lr:.2e}, Time = {elapsed:.1f}s")
            
            # Record final state
            if total_epochs not in self.trajectory_epochs:
                self._record_trajectory(total_epochs)
        
        total_time = time.time() - start_time
        print("-"*80)
        print(f"✓ Training completed in {total_time:.1f}s")
        print(f"✓ Tracked {len(self.trajectory_states)} trajectory points")
        print("="*80 + "\n")