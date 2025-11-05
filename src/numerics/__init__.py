"""
Numerical solvers (FDM)
"""

from .fdm_burgers import BurgersFDM
from .fdm_cavity import CavityFDM

__all__ = [
    'BurgersFDM',
    'CavityFDM',
]