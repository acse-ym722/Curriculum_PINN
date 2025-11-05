# Copilot Instructions for Curriculum_PINN

This repository implements Physics-Informed Neural Networks (PINNs) with curriculum learning for solving partial differential equations (PDEs). Here's what you need to know to work effectively with this codebase:

## Project Architecture

- Each PDE problem has its own dedicated Python file with a complete solver implementation
- `burgers.py` - Solves 1D Burgers equation
- `Lid_Cavity_Flow.py` - Solves 2D lid-driven cavity flow problem

## Key Design Patterns

### Configuration Management
- All hyperparameters and configuration settings are centralized in a `CONFIG` dictionary at the top of each solver file
- Parameters are grouped by category (physical parameters, network architecture, training settings, etc.)

Example from `burgers.py`:
```python
CONFIG = {
    "nu": 0.002,  # Viscosity coefficient
    "layers": [2, 32, 32, 32, 32, 32, 1],  # Neural network architecture
    "curriculum_ramp_ratio": 0.4,  # Curriculum learning parameter
    # ...
}
```

### Solver Components
Each solver implements:
1. Ground truth solution using Finite Difference Method (FDM)
2. PINN-based solution with curriculum learning
3. Comprehensive validation and visualization utilities

### Device Management
- Code automatically detects and uses CUDA if available, falling back to CPU
- Device configuration is handled early in each file:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Critical Workflows

### Numerical Solution Stability
- FDM parameters are pre-evaluated using CFL conditions
- Check `evaluate_fdm_params()` in `burgers.py` for stability criteria
- Adjust `NX` (spatial points) and `NT` (time steps) if stability warnings appear

### Training Process
1. Ground truth solution is generated first using FDM
2. PINN training proceeds with curriculum learning
3. Intermediate results are visualized for monitoring convergence

## Project-Specific Conventions

1. Curriculum Learning Implementation
   - Problems are solved progressively from simple to complex configurations
   - `curriculum_stages` in config defines the progression (e.g., increasing Reynolds number)

2. Validation Approach
   - Solutions are validated against FDM ground truth
   - Visualization routines are built into each solver for immediate feedback

## Integration Points

1. PyTorch Integration
   - Neural networks are implemented using PyTorch
   - Custom loss functions combine physics-based and data-based terms

2. External Dependencies
   - Core: torch, numpy, matplotlib
   - Scientific computing: scipy (for sparse matrix operations in FDM)