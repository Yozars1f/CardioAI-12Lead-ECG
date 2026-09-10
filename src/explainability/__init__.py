"""
CardioAI Explainability Package.
1D Grad-CAM implementation for transparent 12-lead ECG neural interpretation.
"""

from .gradcam1d import GradCAM1D, plot_12lead_gradcam

__all__ = ["GradCAM1D", "plot_12lead_gradcam"]
