"""
Provides DeepONet, DeepOKAN, FNO, and CNO architectures for operator learning.
"""

from .base_deepo import BaseDeepO
from .cno import CNO
from .deepokan import DeepOKAN
from .deeponet import DeepONet
from .fno import FNO
from .model_factory import create_model, model_supports_per_timestep

__all__ = [
    "BaseDeepO",
    "DeepONet",
    "DeepOKAN",
    "FNO",
    "CNO",
    "create_model",
    "model_supports_per_timestep",
]

__version__ = "0.1.0"
