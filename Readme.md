# Curriculum Learning-based Physics-Informed Neural Networks (课程学习优化的物理信息神经网络)

## Overview 概述

This project implements a systematic training framework for Physics-Informed Neural Networks (PINNs) using curriculum learning strategies to solve complex physical problems. The framework specifically addresses training difficulties in problems involving strong nonlinearity and multiple scales.

本项目实现了一个基于课程学习的物理信息神经网络(PINN)系统性训练框架，旨在解决PINN在处理强非线性、多尺度等复杂物理问题时的训练困难。

## Features 特性

- **Multiple PDE Solvers**: Burgers equation and Navier-Stokes equations
- **Curriculum Learning**: Gradual training strategies for improved convergence
- **Ground Truth Generation**: Finite Difference Method (FDM) solvers for validation
- **Comprehensive Visualization**: Detailed plots and metrics for analysis
- **Modular Design**: Easy to extend with new PDEs and training strategies

## Quick Start 快速开始

### Environment Setup 环境配置

1. Create conda environment 创建conda环境:
```bash
conda env create -f environment.yml
conda activate pinn
```

2. Install PyTorch 安装PyTorch:
```bash
# Visit https://pytorch.org/ for the correct command for your system
# code tested on Ubuntu 22.04 with Cuda version 12.6
# run the code visualize current cuda version 
(pinn) yang@DESKTOP-TNJH297:~/Downloads/Curriculum_PINN$ nvidia-smi
Sat Nov 29 15:08:12 2025       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 520.61.05    Driver Version: 560.94       CUDA Version: 12.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:01:00.0  On |                  Off |
|  0%   32C    P8    10W / 500W |   1141MiB / 24564MiB |      2%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Project Structure 项目结构

```
project_root/
├── src/
│   ├── models/          # Neural network architectures
│   │   ├── __init__.py
│   │   └── neural_networks.py
│   ├── solvers/         # PINN solvers
│   │   ├── base_solver.py
│   │   ├── standard_pinn.py
│   │   └── curriculum_pinn.py
│   ├── numerics/        # FDM numerical solvers
│   │   ├── fdm_burgers.py
│   │   └── fdm_cavity.py
│   ├── visualization/   # Plotting utilities
│   │   ├── burgers_plots.py
│   │   └── cavity_plots.py
│   ├── configs/         # Experiment configurations
│   │   ├── burgers_config.py
│   │   └── cavity_config.py
│   ├── experiment_burgers.py
│   └── experiment_cavity.py
├── outputs/
│   ├── burgers/
│   └── cavity/
└── README.md
```

## Research Phases 研究阶段

### Phase 1: Empirical Study - Linear Curriculum Learning
第一阶段：实证研究 - 线性课程学习

#### Representative Problems 代表性问题:

1. **1D Burgers' Equation 一维伯格斯方程**
   - Challenge: Solution develops shock waves as viscosity ν decreases
   - Curriculum: Linear increase of β from 0 to 1
   - Implementation: See `experiment_burgers.py`

2. **2D Lid-Driven Cavity Flow 二维顶盖驱动流**
   - Challenge: Complex vortex structures at high Reynolds numbers
   - Curriculum: Progressive increase in Reynolds number
   - Implementation: See `experiment_cavity.py`

3. **2D Cylinder Flow 二维圆柱绕流**
   - Challenge: Complex flow phenomena including:
     * Vortex shedding and von Kármán vortex street
     * Flow separation and wake formation
     * Reynolds number dependent flow regimes
   - Curriculum: Multi-stage learning strategy increase in Reynolds number
   - Physics Features:
     * Incompressible Navier-Stokes equations
     * No-slip boundary condition on cylinder surface
     * Far-field uniform flow conditions
   - Implementation: See `experiment_cylinder.py`

4. **2D Wave Equation 二维波动方程**
### Phase 2: Theoretical Analysis
第三阶段：理论分析

Theoretical investigation focuses on:
- Loss landscape smoothing effects
- Optimization dynamics and gradient flow
- Spectral bias and frequency curriculum

## Usage Examples 使用示例

### Quick Run Commands

```bash
# Run Burgers equation experiment
python -m src.experiment_burgers --config default --output outputs/burgers_mlp
python -m src.experiment_burgers --config resnet --output outputs/burgers_resnet
python -m src.experiment_burgers --config fourier --output outputs/burgers_fourier

# Run Cavity flow experiment
python -m src.experiment_cavity --config low_re --output outputs/cavity_low_re
python -m src.experiment_cavity --config high_re --output outputs/cavity_high_re

# Run wave eqn experiment
python -m src.experiment_wave
python -m src.experiment_wave_2d --config default --output outputs/wave2d
python -m src.experiment_wave_2d --config default --output outputs/wave2d_3step
```

### Configuration Examples

```python
# Burgers equation configuration
CONFIG = {
    "nu": 0.002,              # Viscosity coefficient
    "curriculum_ramp_ratio": 0.4,  # Curriculum learning parameter
    "layers": [2, 32, 32, 32, 32, 32, 1]  # Network architecture
}

# Lid-driven cavity flow configuration
CONFIG = {
    "curriculum_stages": [
        {"Re": 10, "epochs": 2000},
        {"Re": 30, "epochs": 2000},
        {"Re": 60, "epochs": 2000},
        {"Re": 100, "epochs": 14000},
    ]
}
```

## Core Research Questions 核心科学问题

How to design a systematic training framework to overcome PINN's training difficulties in complex physical problems involving strong nonlinearity and multiple scales, while theoretically explaining its effectiveness?

如何设计一个系统性的训练框架，克服PINN在求解强非线性、多尺度等复杂物理问题时的训练失败问题，并从理论上解释其有效性？

## Output Files 输出文件

Each experiment generates:
- `comprehensive_comparison.png`: Main results visualization
- `spacetime_comparison.png` (Burgers) / `centerline_comparison.png` (Cavity)
- `standard_pinn.pt`: Trained standard PINN model
- `curriculum_pinn.pt`: Trained curriculum PINN model
- `config.json`: Experiment configuration


## Contributing 贡献

Contributions are welcome! Please feel free to:
- Open issues for bugs or feature requests
- Submit pull requests for improvements
- Add new physical problems or curriculum strategies
- Enhance visualization capabilities
- Contribute to theoretical analysis

## License 许可

This project is licensed under the MIT License.

## Citation 引用

If you use this framework in your research, please cite:

```bibtex
@software{curriculum_pinn,
  title={Curriculum Learning Physics-Informed Neural Networks for Parametric PDEs},
  author={Yang Meng},
  year={2025},
  url={https://github.com/acse-ym722/Curriculum_PINN}
}
```