"""
Custom optimizers for training.

Available optimizers:
- SOAP: Shampoo with Adam in the Preconditioner
"""

from .soap import SOAP

__all__ = ["SOAP"]
