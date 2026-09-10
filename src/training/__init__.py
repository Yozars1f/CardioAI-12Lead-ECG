"""
CardioAI Training Package.
Clinical model trainers, evaluators, and checkpointing routines.
"""

from .trainer import ClinicalECGTrainer

__all__ = ["ClinicalECGTrainer"]
