"""
CardioAI_12Lead_ECG: 1D Grad-CAM (Gradient-Weighted Class Activation Mapping).
Provides transparent, lead-specific visual explanations for 12-lead ECG diagnoses.
Author: Youssef Ahmad, MD (Cardiovascular AI Track)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from pathlib import Path
from typing import List, Optional, Tuple

STANDARD_12_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


class GradCAM1D:
    """
    1D Grad-CAM implementation for ECGResNet1D.
    Computes saliency heatmaps along the temporal axis to reveal clinical justifications.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or model.stages[-1]

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        signal: torch.Tensor,
        target_class_idx: int,
        target_length: int = 1000
    ) -> np.ndarray:
        """
        Generates 1D heatmap of length target_length (values in [0.0, 1.0]).
        Args:
            signal: Tensor of shape (1, 12, 1000) or (12, 1000)
            target_class_idx: Int class index (e.g. 1 for MI)
        """
        self.model.eval()
        if signal.dim() == 2:
            signal = signal.unsqueeze(0)  # (1, 12, 1000)

        device = next(self.model.parameters()).device
        signal = signal.to(device)

        # Forward pass
        logits = self.model(signal)
        score = logits[0, target_class_idx]

        # Backward pass for target class
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Activations: (1, C, L_reduced), Gradients: (1, C, L_reduced)
        activations = self.activations[0]  # (C, L_red)
        gradients = self.gradients[0]      # (C, L_red)

        # Global average pooling of gradients per channel (weights alpha_k)
        weights = torch.mean(gradients, dim=1, keepdim=True)  # (C, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=0)  # (L_red)

        # Apply ReLU to keep only positive evidence supporting the diagnosis
        cam = F.relu(cam)

        # Interpolate 1D CAM back to full ECG signal length (1000 timesteps)
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, L_red)
        cam = F.interpolate(cam, size=target_length, mode="linear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Min-Max Normalization to [0.0, 1.0]
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def plot_12lead_gradcam(
    signal: np.ndarray,
    heatmap: np.ndarray,
    predicted_class: str,
    confidence: float,
    ecg_id: int,
    output_path: Path,
    lead_names: Optional[List[str]] = None
):
    """
    Overlays 1D Grad-CAM heatmap directly onto 12-lead ECG strips.
    Red/Warm highlights show exactly which QRS complexes and ST segments justified the diagnosis.
    """
    if lead_names is None:
        lead_names = STANDARD_12_LEADS

    time = np.linspace(0, 10, signal.shape[1])
    fig, axes = plt.subplots(6, 2, figsize=(15, 11), sharex=True, dpi=300)
    axes = axes.flatten()

    cmap = plt.get_cmap("jet")
    norm = Normalize(vmin=0.0, vmax=1.0)

    for idx, lead_name in enumerate(lead_names):
        ax = axes[idx]
        lead_sig = signal[idx]

        # Construct colored line segments based on Grad-CAM heatmap intensity
        points = np.array([time, lead_sig]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # LineCollection colored by Grad-CAM heatmap values
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(heatmap[:-1])
        lc.set_linewidth(1.8)
        ax.add_collection(lc)

        # Thin gray background signal for baseline reference
        ax.plot(time, lead_sig, color="#555555", alpha=0.3, linewidth=0.7)

        ax.set_xlim(0, 10)
        ax.set_ylim(-3.5, 3.5)
        ax.set_ylabel(lead_name, fontsize=10, fontweight="bold", rotation=0, labelpad=15)
        ax.grid(True, linestyle=":", color="#cccccc", alpha=0.7)

    axes[-2].set_xlabel("Time (seconds)", fontsize=10, fontweight="bold")
    axes[-1].set_xlabel("Time (seconds)", fontsize=10, fontweight="bold")

    # Add Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("1D Grad-CAM Importance (Neural Activation)", fontsize=10, fontweight="bold")

    plt.suptitle(
        f"CardioAI 1D Grad-CAM Saliency Explanation | Record #{ecg_id:05d}\n"
        f"Diagnosed Target: {predicted_class} (Confidence: {confidence*100:.1f}%) | Breaking the Black-Box",
        fontsize=13, fontweight="bold", y=0.98
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" [OK] Saved 12-lead Grad-CAM visualization to: {output_path}")
