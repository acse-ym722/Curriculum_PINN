"""
Curriculum Learning PINN Solvers
"""
import torch
import numpy as np
import copy
from .standard_pinn import BurgersPINN, NavierStokesPINN
import time
import copy


class CurriculumBurgersPINN(BurgersPINN):
    """
    Curriculum learning PINN for Burgers equation.
    Uses a multi-stage approach defined in the config:
    - Stage 1: Train with a high, fixed viscosity.
    - Stage 2: Linearly anneal viscosity from high to the target value.
    - Stage 3: Train with the final, low target viscosity.
    """
    
    def __init__(self, model, config, device='cuda'):
        # --- NEW: Read stage-based curriculum parameters ---
        self.curriculum_stages = config.get('curriculum_stages', [])
        if not self.curriculum_stages:
            raise ValueError("CurriculumBurgersPINN requires 'curriculum_stages' to be defined in the config.")
            
        self.nu_initial = config.get('nu_initial', config['nu'] * 10)
        self.nu_target = config.get('nu_target', config['nu'])
        
        # Total epochs is the sum of epochs from all stages
        self.total_epochs = sum(stage['epochs'] for stage in self.curriculum_stages)
        
        # Make a copy of config to avoid modifying the original
        curriculum_config = copy.deepcopy(config)
        # Start with the initial high viscosity
        curriculum_config['nu'] = self.nu_initial
        
        # Call parent constructor
        super().__init__(model, curriculum_config, device)
        
        # Override nu to start with the initial curriculum value
        self.nu = self.nu_initial
        
        print(f"\nCurriculum Learning Setup (Stage-based for Burgers):")
        print(f"  Initial nu: {self.nu_initial:.6f}")
        print(f"  Target nu:  {self.nu_target:.6f}")
        print(f"  Total epochs: {self.total_epochs}")
        for i, stage in enumerate(self.curriculum_stages):
            print(f"  - Stage {i+1} ('{stage['name']}'): {stage['epochs']} epochs")
        print()

    def _update_curriculum_stage(self, stage_idx, epoch_in_stage, epochs_for_stage):
        """Update curriculum parameter (nu) based on the current stage and epoch."""
        if stage_idx == 0:  # Stage 1: High Viscosity
            self.nu = self.nu_initial
        elif stage_idx == 1:  # Stage 2: Annealing
            progress = (epoch_in_stage + 1) / epochs_for_stage
            self.nu = self.nu_initial * (1 - progress) + self.nu_target * progress
        elif stage_idx == 2:  # Stage 3: Target Viscosity
            self.nu = self.nu_target
        else: # Fallback for any additional stages
            self.nu = self.nu_target

    def train(self, verbose=True, save_interval=1000):
        """
        Override train to implement the multi-stage curriculum.
        """
        print("="*80)
        print("Training CurriculumBurgersPINN (Stage-based)")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Total epochs: {self.total_epochs}")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        print("-"*80)
        
        start_time = time.time()
        global_epoch = 0
        
        # --- NEW: Loop through each curriculum stage ---
        for stage_idx, stage in enumerate(self.curriculum_stages):
            stage_epochs = stage['epochs']
            
            print(f"\n{'='*30} STAGE {stage_idx + 1}/{len(self.curriculum_stages)}: {stage['name']} {'='*30}")
            print(f"Epochs in this stage: {stage_epochs}\n")
            
            for epoch_in_stage in range(stage_epochs):
                # Update curriculum parameter 'nu' for the current step
                self._update_curriculum_stage(stage_idx, epoch_in_stage, stage_epochs)
                
                # Standard training step from the base class
                loss, loss_dict = self.train_step()
                
                # Record history
                self.loss_history.append(loss.item())
                for key, value in loss_dict.items():
                    self.loss_components_history[key].append(value)
                
                # Verbose output
                if verbose and ((global_epoch + 1) % save_interval == 0 or epoch_in_stage == stage_epochs - 1):
                    elapsed = time.time() - start_time
                    lr = self.optimizer.param_groups[0]['lr']
                    print(f"Epoch {global_epoch+1:6d}/{self.total_epochs} [Stage {stage_idx+1}, {epoch_in_stage+1:5d}/{stage_epochs}]: "
                          f"Loss = {loss_dict['total']:.4e}, "
                          f"IC = {loss_dict['ic']:.2e}, "
                          f"BC = {loss_dict['bc']:.2e}, "
                          f"PDE = {loss_dict['pde']:.2e}, "
                          f"nu = {self.nu:.6f}, "
                          f"LR = {lr:.2e}, "
                          f"Time = {elapsed:.1f}s")
                
                global_epoch += 1
        
        total_time = time.time() - start_time
        print("-"*80)
        print(f"✓ Curriculum training completed in {total_time:.1f}s")
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


from .standard_pinn import WavePINN
class CurriculumWavePINN(WavePINN):
    """
    Curriculum learning PINN for Wave equation
    Gradually increases wave speed from low to high (easy to hard)
    """
    
    def __init__(self, model, config, device='cuda'):
        # Store target c and total epochs
        self.c_target = config.get('c', 1.0)
        self.curriculum_ramp_ratio = config.get('curriculum_ramp_ratio', 0.4)
        self.total_epochs = config.get('epochs', 20000)
        self.ramp_epochs = int(self.total_epochs * self.curriculum_ramp_ratio)
        
        # Make a copy of config
        curriculum_config = copy.deepcopy(config)
        curriculum_config['c'] = self.c_target * 0.2  # Start with 20% of target speed
        
        # Call parent constructor
        super().__init__(model, curriculum_config, device)
        
        # Override c to start easier
        self.c = self.c_target * 0.2
        self.current_epoch = 0
        
        print(f"\nCurriculum Learning Setup (Wave Equation):")
        print(f"  Target wave speed (c): {self.c_target:.2f}")
        print(f"  Initial wave speed (c): {self.c:.2f}")
        print(f"  Ramp epochs: {self.ramp_epochs}")
        print(f"  Total epochs: {self.total_epochs}\n")
    
    def update_curriculum(self, epoch):
        """Update curriculum parameter (wave speed c) based on epoch"""
        self.current_epoch = epoch
        
        if epoch < self.ramp_epochs:
            # Linearly increase c from initial to target
            progress = epoch / self.ramp_epochs
            self.c = self.c_target * 0.2 * (1 - progress) + self.c_target * progress
        else:
            self.c = self.c_target
    
    def train(self, verbose=True, save_interval=1000):
        """Override train to include curriculum updates"""
        print("="*80)
        print("Training CurriculumWavePINN")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.total_epochs}")
        print(f"Curriculum ramp: {self.ramp_epochs} epochs")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        print("-"*80)
        
        start_time = time.time()
        
        for epoch in range(self.total_epochs):
            # Update curriculum
            self.update_curriculum(epoch)
            
            # Standard training step
            loss, loss_dict = self.train_step()
            
            # Record history
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
                      f"c = {self.c:.2f}, "
                      f"LR = {lr:.2e}, "
                      f"Time = {elapsed:.1f}s")
        
        total_time = time.time() - start_time
        print("-"*80)
        print(f"✓ Training completed in {total_time:.1f}s")
        print("="*80 + "\n")


from .standard_pinn import WavePINN2D 
class CurriculumWavePINN2D(WavePINN2D):
    """
    Curriculum learning PINN for 2D Wave equation.
    Uses a multi-stage approach defined in the config to gradually increase wave speed 'c'.
    """
    
    def __init__(self, model, config, device='cuda'):
        self.curriculum_stages = config.get('curriculum_stages', [])
        if not self.curriculum_stages:
            raise ValueError("CurriculumWavePINN2D requires 'curriculum_stages' to be defined in the config.")
            
        # Total epochs is the sum of epochs from all stages
        self.total_epochs = sum(stage['epochs'] for stage in self.curriculum_stages)
        
        # Make a copy of config to avoid modifying the original
        curriculum_config = copy.deepcopy(config)
        
        # Start with the parameters from the first stage
        first_stage = self.curriculum_stages[0]
        curriculum_config['c'] = first_stage.get('c', first_stage.get('c_start', 1.0))
        
        # Call parent constructor with initial config
        super().__init__(model, curriculum_config, device)
        
        # Store c history for plotting
        self.c_history = []
        
        print(f"\nCurriculum Learning Setup (Stage-based for 2D Wave Equation):")
        print(f"  Total epochs: {self.total_epochs}")
        for i, stage in enumerate(self.curriculum_stages):
            stage_info = f"  - Stage {i+1} ('{stage['name']}'): {stage['epochs']} epochs"
            if 'c' in stage:
                stage_info += f", c = {stage['c']:.2f}"
            elif 'c_start' in stage and 'c_end' in stage:
                stage_info += f", c anneals from {stage['c_start']:.2f} to {stage['c_end']:.2f}"
            print(stage_info)
        print()

    def train(self, verbose=True, save_interval=1000):
        """
        Override train to implement the multi-stage curriculum.
        """
        print("="*80)
        print("Training CurriculumWavePINN2D (Stage-based)")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Total epochs: {self.total_epochs}")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        print("-"*80)
        
        start_time = time.time()
        global_epoch = 0
        
        # Loop through each curriculum stage
        for stage_idx, stage in enumerate(self.curriculum_stages):
            stage_epochs = stage['epochs']
            
            print(f"\n{'='*25} STAGE {stage_idx + 1}/{len(self.curriculum_stages)}: {stage['name']} {'='*25}")
            print(f"Epochs in this stage: {stage_epochs}\n")
            
            c_start = stage.get('c_start')
            c_end = stage.get('c_end')
            is_annealing_stage = c_start is not None and c_end is not None

            for epoch_in_stage in range(stage_epochs):
                # Update curriculum parameter 'c' for the current step
                if is_annealing_stage:
                    progress = (epoch_in_stage + 1) / stage_epochs
                    self.c = c_start * (1 - progress) + c_end * progress
                else:
                    self.c = stage['c']
                
                self.c_history.append(self.c)

                # Standard training step from the base class
                loss, loss_dict = self.train_step()
                
                # Record history
                self.loss_history.append(loss.item())
                for key, value in loss_dict.items():
                    self.loss_components_history[key].append(value)
                
                # Verbose output
                if verbose and ((global_epoch + 1) % save_interval == 0 or epoch_in_stage == stage_epochs - 1):
                    elapsed = time.time() - start_time
                    lr = self.optimizer.param_groups[0]['lr']
                    print(f"Epoch {global_epoch+1:6d}/{self.total_epochs} [Stage {stage_idx+1}, {epoch_in_stage+1:5d}/{stage_epochs}]: "
                          f"Loss = {loss_dict['total']:.4e}, "
                          f"IC = {loss_dict['ic']:.2e}, "
                          f"BC = {loss_dict['bc']:.2e}, "
                          f"PDE = {loss_dict['pde']:.2e}, "
                          f"c = {self.c:.4f}, "
                          f"LR = {lr:.2e}, "
                          f"Time = {elapsed:.1f}s")
                
                global_epoch += 1
        
        total_time = time.time() - start_time
        print("-"*80)
        print(f"✓ Curriculum training completed in {total_time:.1f}s")
        print("="*80 + "\n")


from .standard_pinn import DarcyPINN
class CurriculumDarcyPINN(DarcyPINN):
    """
    Curriculum PINN: Single Source → Source + Sink
    """
    
    def __init__(self, model, config, device='cuda'):
        self.curriculum_stages = config.get('curriculum_stages', [])
        self.current_stage = 0
        
        # 初始化使用第一阶段
        config_copy = config.copy()
        config_copy['f_source_torch'] = self.curriculum_stages[0]['f_source_torch']
        config_copy['epochs'] = sum(s['epochs'] for s in self.curriculum_stages)
        
        super().__init__(model, config_copy, device)
        self.base_config = config
    
    def train(self, verbose=True, save_interval=500):
        """课程学习训练"""
        print("\n" + "="*80)
        print("CURRICULUM LEARNING: Single Source → Source + Sink")
        print("="*80)
        
        start_time = time.time()
        total_epochs_done = 0
        
        for stage_idx, stage in enumerate(self.curriculum_stages):
            self.current_stage = stage_idx
            
            # 更新源项
            self.f_source = stage['f_source_torch']
            
            # 重新生成训练数据
            print(f"\n{'='*80}")
            print(f"STAGE {stage_idx + 1}: {stage['name']}")
            print(f"{'='*80}\n")
            
            # 生成对应的FDM解
            from src.numerics.fdm_darcy import DarcyFDM
            fdm = DarcyFDM(
                K=self.K,
                f_source=stage['f_source_np'],
                x_domain=self.config['x_domain'],
                y_domain=self.config['y_domain'],
                nx=self.config['fdm_nx'],
                ny=self.config['fdm_ny'],
                bc_left=self.config['bc_left'],
                bc_right=self.config['bc_right'],
                bc_top=self.config.get('bc_top', 0.0),
                bc_bottom=self.config.get('bc_bottom', 0.0)
            )
            p_fdm, x_fdm, y_fdm = fdm.solve()
            
            # 更新data loss的参考解
            fdm_solution = (p_fdm, x_fdm, y_fdm)
            self.config['fdm_solution'] = fdm_solution
            self._generate_training_data()
            
            epochs = stage['epochs']
            
            for epoch in range(epochs):
                loss, loss_dict = self.train_step()
                
                self.loss_history.append(loss.item())
                for key, val in loss_dict.items():
                    if key not in self.loss_components_history:
                        self.loss_components_history[key] = []
                    self.loss_components_history[key].append(val)
                
                total_epochs_done += 1
                
                if verbose and ((epoch + 1) % save_interval == 0 or epoch == 0):
                    lr = self.optimizer.param_groups[0]['lr']
                    elapsed = time.time() - start_time
                    print(f"  Stage {stage_idx+1} | Epoch {epoch+1:4d}/{epochs} | "
                          f"Loss = {loss_dict['total']:.2e}, "
                          f"PDE = {loss_dict['pde']:.2e}, "
                          f"BC = {loss_dict['bc']:.2e}, "
                          f"Data = {loss_dict['data']:.2e}, "
                          f"LR = {lr:.2e}, Time = {elapsed:.1f}s")
        
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"CURRICULUM TRAINING COMPLETED in {total_time:.1f}s")
        print(f"{'='*80}\n")


