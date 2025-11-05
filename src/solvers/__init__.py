"""
PINN solvers
"""

from .base_solver import BasePINNSolver
from .standard_pinn import BurgersPINN, NavierStokesPINN
from .curriculum_pinn import CurriculumBurgersPINN, CurriculumNavierStokesPINN

__all__ = [
    'BasePINNSolver',
    'BurgersPINN',
    'NavierStokesPINN',
    'CurriculumBurgersPINN',
    'CurriculumNavierStokesPINN',
]