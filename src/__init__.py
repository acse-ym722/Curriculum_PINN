"""
PINN Framework for CFD Problems
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from . import models
from . import solvers
from . import numerics
from . import visualization
from . import configs

__all__ = [
    'models',
    'solvers',
    'numerics',
    'visualization',
    'configs',
]