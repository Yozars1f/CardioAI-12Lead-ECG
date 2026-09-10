"""
CardioAI Calibration Package.
Post-hoc probability calibration routines and Reliability Diagram visualizations.
"""

from .temperature_scaling import (
    ModelWithTemperature,
    compute_binary_ece,
    compute_multilabel_ece,
    plot_reliability_diagrams
)

__all__ = [
    "ModelWithTemperature",
    "compute_binary_ece",
    "compute_multilabel_ece",
    "plot_reliability_diagrams"
]
