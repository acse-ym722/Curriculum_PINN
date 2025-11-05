"""
Experiment configurations
"""

from .burgers_config import get_config as get_burgers_config
from .cavity_config import get_config as get_cavity_config

__all__ = [
    'get_burgers_config',
    'get_cavity_config',
]