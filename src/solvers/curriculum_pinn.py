"""
Curriculum Learning PINN Solvers
"""
import torch
import numpy as np
import copy
from .standard_pinn import BurgersPINN, NavierStokesPINN
import time

class CurriculumBurgersPINN(BurgersPINN):
    """
    Curriculum learning PINN for Burgers equation
    Gradually increases viscosity from high to low (easy to hard)
    """
    
    def __init__(self, model, config, device='cuda'):
        # Store target nu and total epochs BEFORE calling super().__init__
        self.nu_target = config.get('nu', 0.01)
        self.curriculum_ramp_ratio = config.get('curriculum_ramp_ratio', 0.4)
        self.total_epochs = config.get('epochs', 20000)
        self.ramp_epochs = int(self.total_epochs * self.curriculum_ramp_ratio)
        
        # Make a copy of config to avoid modifying the original
        curriculum_config = copy.deepcopy(config)
        curriculum_config['nu'] = self.nu_target * 10  # Start easier
        
        # Call parent constructor
        super().__init__(model, curriculum_config, device)
        
        # Override nu to start with easier problem
        self.nu = self.nu_target * 10
        self.current_epoch = 0
        
        print(f"\nCurriculum Learning Setup:")
        print(f"  Target nu: {self.nu_target:.6f}")
        print(f"  Initial nu: {self.nu:.6f}")
        print(f"  Ramp epochs: {self.ramp_epochs}")
        print(f"  Total epochs: {self.total_epochs}\n")
    
    def update_curriculum(self, epoch):
        """Update curriculum parameter (nu) based on epoch"""
        self.current_epoch = epoch
        
        if epoch < self.ramp_epochs:
            # Linearly decrease nu from high to target
            progress = epoch / self.ramp_epochs
            self.nu = self.nu_target * 10 * (1 - progress) + self.nu_target * progress
        else:
            self.nu = self.nu_target
    
    def train(self, verbose=True, save_interval=1000):
        """
        Override train to include curriculum updates
        """
        print("="*80)
        print("Training CurriculumBurgersPINN")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.total_epochs}")
        print(f"Curriculum ramp: {self.ramp_epochs} epochs")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        print("-"*80)
        
        import time
        start_time = time.time()
        
        for epoch in range(self.total_epochs):
            # Update curriculum
            self.update_curriculum(epoch)
            
            # Standard training step
            loss, loss_dict = self.train_step()
            
            # Record history (use the correct attribute names from base_solver.py)
            self.loss_history.append(loss.item())
            for key, value in loss_dict.items():
                self.loss_components_history[key].append(value)
            
            # Verbose output
            if verbose and ((epoch + 1) % save_interval == 0 or epoch == self.total_epochs - 1):
                elapsed = time.time() - start_time
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:6d}/{self.total_epochs}: "
                      f"Loss = {loss_dict['total']:.4e}, "
                      f"IC = {loss_dict['ic']:.2e}, "
                      f"BC = {loss_dict['bc']:.2e}, "
                      f"PDE = {loss_dict['pde']:.2e}, "
                      f"nu = {self.nu:.6f}, "
                      f"LR = {lr:.2e}, "
                      f"Time = {elapsed:.1f}s")
        
        total_time = time.time() - start_time
        print("-"*80)
        print(f"✓ Training completed in {total_time:.1f}s")
        print("="*80 + "\n")


class CurriculumNavierStokesPINN(NavierStokesPINN):
    """
    Curriculum learning PINN for Navier-Stokes equations
    Gradually increases Reynolds number from low to high (easy to hard)
    """
    
    def __init__(self, model, config, device='cuda'):
        # Curriculum stages: list of (Re, epochs)
        self.curriculum_stages = config.get('curriculum_stages', [])
        
        # Store total epochs
        self.total_epochs = sum(stage['epochs'] for stage in self.curriculum_stages)
        
        # Make a copy of config to avoid modifying the original
        curriculum_config = copy.deepcopy(config)
        
        # Start with first stage
        curriculum_config['Re'] = self.curriculum_stages[0]['Re']
        
        # Call parent constructor
        super().__init__(model, curriculum_config, device)
        
        self.current_stage = 0
        self.stage_epoch = 0
        
        print(f"\nCurriculum Learning Setup:")
        print(f"  Total stages: {len(self.curriculum_stages)}")
        print(f"  Total epochs: {self.total_epochs}")
        for i, stage in enumerate(self.curriculum_stages):
            print(f"  Stage {i}: Re={stage['Re']}, epochs={stage['epochs']}")
        print()
    
    def train(self, verbose=True, save_interval=500):
        """
        Override train to include curriculum updates
        """
        print("="*80)
        print("Training CurriculumNavierStokesPINN")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Total epochs: {self.total_epochs}")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        print("-"*80)
        
        import time
        start_time = time.time()
        global_epoch = 0
        
        for stage_idx, stage in enumerate(self.curriculum_stages):
            self.current_stage = stage_idx
            self.Re = stage['Re']
            self.nu = 1.0 / self.Re
            stage_epochs = stage['epochs']
            
            print(f"\n{'='*80}")
            print(f"Stage {stage_idx}: Re = {self.Re}, Epochs = {stage_epochs}")
            print(f"{'='*80}\n")
            
            for epoch in range(stage_epochs):
                # Standard training step
                loss, loss_dict = self.train_step()
                
                # Record history (use the correct attribute names from base_solver.py)
                self.loss_history.append(loss.item())
                for key, value in loss_dict.items():
                    self.loss_components_history[key].append(value)
                
                # Verbose output
                if verbose and ((epoch + 1) % save_interval == 0 or epoch == stage_epochs - 1):
                    elapsed = time.time() - start_time
                    lr = self.optimizer.param_groups[0]['lr']
                    print(f"Stage {stage_idx} | Epoch {epoch+1:6d}/{stage_epochs} | "
                          f"Global {global_epoch+1:6d} | Loss: {loss_dict['total']:.6e} | "
                          f"BC: {loss_dict['bc']:.6e} | "
                          f"PDE: {loss_dict['pde']:.6e} | "
                          f"Re: {self.Re:.1f} | LR: {lr:.6e} | "
                          f"Time: {elapsed:.1f}s")
                
                global_epoch += 1
        
        total_time = time.time() - start_time
        print("-"*80)
        print(f"✓ Training completed in {total_time:.1f}s")
        print("="*80 + "\n")


from .standard_pinn import CylinderPINN
class CurriculumCylinderPINN(CylinderPINN):
    """
    Curriculum learning PINN for cylinder flow
    Gradually increases Reynolds number
    """
    
    def __init__(self, model, config, device='cuda'):
        self.curriculum_stages = config.get('curriculum_stages', [])
        self.current_stage = 0
        self.total_epochs_completed = 0
        
        # Initialize with first stage Reynolds number
        config_copy = config.copy()
        config_copy['Re'] = self.curriculum_stages[0]['Re']
        # Set epochs to total of all stages
        config_copy['epochs'] = sum(stage['epochs'] for stage in self.curriculum_stages)
        
        super().__init__(model, config_copy, device)
        
        # Store original config for later stages
        self.base_config = config
    
    def train(self, verbose=True, save_interval=500):
        """Train with curriculum learning"""
        print("\n" + "="*80)
        print("CURRICULUM LEARNING FOR CYLINDER FLOW")
        print("="*80)
        print(f"Total stages: {len(self.curriculum_stages)}")
        for i, stage in enumerate(self.curriculum_stages):
            print(f"  Stage {i+1}: Re = {stage['Re']}, Epochs = {stage['epochs']}")
        print("="*80 + "\n")
        
        start_time = time.time()
        
        for stage_idx, stage in enumerate(self.curriculum_stages):
            self.current_stage = stage_idx
            self.Re = stage['Re']
            self.nu = 1.0 / self.Re
            
            print(f"\n{'='*80}")
            print(f"STAGE {stage_idx + 1}/{len(self.curriculum_stages)}: Re = {self.Re}")
            print(f"{'='*80}\n")
            
            epochs = stage['epochs']
            
            for epoch in range(epochs):
                # Use train_step from base class
                loss, loss_dict = self.train_step()
                
                # Store history
                self.loss_history.append(loss.item())
                for key, val in loss_dict.items():
                    if key not in self.loss_components_history:
                        self.loss_components_history[key] = []
                    self.loss_components_history[key].append(val)
                
                self.total_epochs_completed += 1
                
                if verbose and ((epoch + 1) % save_interval == 0 or epoch == 0):
                    lr = self.optimizer.param_groups[0]['lr']
                    elapsed = time.time() - start_time
                    
                    # Build loss string
                    loss_str_parts = [f"Total = {loss_dict.get('total', loss.item()):.2e}"]
                    for key in ['bc_inlet', 'bc_outlet', 'bc_wall', 'bc_cylinder', 'pde', 'data']:
                        if key in loss_dict and loss_dict[key] > 0:
                            loss_str_parts.append(f"{key.upper()} = {loss_dict[key]:.2e}")
                    
                    loss_str = ", ".join(loss_str_parts)
                    
                    print(f"  Stage {stage_idx+1} | Epoch {epoch+1:4d}/{epochs} | "
                          f"Re={self.Re} | {loss_str} | LR={lr:.2e} | Time={elapsed:.1f}s")
        
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print("CURRICULUM TRAINING COMPLETED")
        print(f"Total epochs: {self.total_epochs_completed}")
        print(f"Final Re: {self.Re}")
        print(f"Total time: {total_time:.1f}s")
        print(f"{'='*80}\n")