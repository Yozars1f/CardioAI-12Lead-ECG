"""
CardioAI_12Lead_ECG: Probability Calibration via Post-Hoc Temperature Scaling.
Recalibrates overconfident neural networks and computes Expected Calibration Error (ECE).
Author: Youssef Ahmad, MD (Cardiovascular AI Track)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, List, Optional


def compute_binary_ece(probs: np.ndarray, targets: np.ndarray, num_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) for multi-label / binary predictions.
    Formula: ECE = sum_m (|B_m| / N) * |accuracy(B_m) - confidence(B_m)|
    """
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    total_samples = len(probs)

    for i in range(num_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        # Identify samples falling into confidence bin
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            avg_confidence = np.mean(probs[in_bin])
            avg_accuracy = np.mean(targets[in_bin])
            ece += (bin_size / total_samples) * np.abs(avg_accuracy - avg_confidence)

    return float(ece)


def compute_multilabel_ece(probs: np.ndarray, targets: np.ndarray, num_bins: int = 10) -> Dict[str, float]:
    """Computes ECE for each diagnostic superclass and the macro average."""
    eces = []
    num_classes = probs.shape[1]
    for c in range(num_classes):
        c_ece = compute_binary_ece(probs[:, c], targets[:, c], num_bins=num_bins)
        eces.append(c_ece)
    return {
        "class_eces": eces,
        "macro_ece": float(np.mean(eces))
    }


class ModelWithTemperature(nn.Module):
    """
    Wraps trained 1D-ResNet with Temperature Scaling parameter T.
    Output: calibrated probabilities = sigmoid(logits / T).
    Preserves AUROC/AUPRC discrimination while drastically lowering ECE.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # Initialize temperature parameter T = 1.0 (Log scale for positivity enforcement)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return self.scale_logits(logits)

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales logits by temperature T."""
        temp = self.temperature.clamp(min=0.01)
        return logits / temp

    def calibrate(self, val_loader, device: Optional[torch.device] = None, max_iter: int = 50) -> float:
        """
        Learns optimal temperature T on validation set using L-BFGS optimizer.
        """
        if device is None:
            device = next(self.model.parameters()).device

        self.model.eval()
        logits_list = []
        labels_list = []

        with torch.no_grad():
            for batch in val_loader:
                signals = batch["signal"].to(device)
                labels = batch["label"].to(device)
                logits = self.model(signals)
                logits_list.append(logits)
                labels_list.append(labels)

        all_logits = torch.cat(logits_list).to(device)
        all_labels = torch.cat(labels_list).to(device)

        # Multi-label binary cross entropy with logits
        bce_criterion = nn.BCEWithLogitsLoss()

        optimizer = optim.LBFGS([self.temperature], lr=0.05, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            scaled_logits = self.scale_logits(all_logits)
            loss = bce_criterion(scaled_logits, all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        optimal_t = float(self.temperature.item())
        return optimal_t


def plot_reliability_diagrams(
    uncalibrated_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    targets: np.ndarray,
    superclasses: List[str],
    output_path: Path,
    num_bins: int = 10
):
    """
    Generates clinical Reliability Diagrams (Calibration Curves) comparing
    uncalibrated vs calibrated model against perfect calibration line.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Perfect calibration reference
    for ax in axes:
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7, linewidth=1.5)

    # 1. Uncalibrated Reliability
    uncal_eces = []
    for c_idx, c_name in enumerate(superclasses):
        p = uncalibrated_probs[:, c_idx]
        y = targets[:, c_idx]
        accs = []
        confs = []
        for b in range(num_bins):
            mask = (p >= bins[b]) & (p < bins[b+1])
            if np.sum(mask) > 0:
                accs.append(np.mean(y[mask]))
                confs.append(np.mean(p[mask]))
        axes[0].plot(confs, accs, marker='o', label=c_name, linewidth=1.5)
        uncal_eces.append(compute_binary_ece(p, y, num_bins))

    macro_uncal_ece = np.mean(uncal_eces)
    axes[0].set_title(f"Uncalibrated 1D-ResNet\n(Macro ECE: {macro_uncal_ece*100:.2f}%)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Mean Predicted Confidence", fontsize=10)
    axes[0].set_ylabel("Empirical Disease Prevalence", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(frameon=True, fontsize=8)

    # 2. Calibrated Reliability (Post-hoc Temperature Scaling)
    cal_eces = []
    for c_idx, c_name in enumerate(superclasses):
        p = calibrated_probs[:, c_idx]
        y = targets[:, c_idx]
        accs = []
        confs = []
        for b in range(num_bins):
            mask = (p >= bins[b]) & (p < bins[b+1])
            if np.sum(mask) > 0:
                accs.append(np.mean(y[mask]))
                confs.append(np.mean(p[mask]))
        axes[1].plot(confs, accs, marker='s', label=c_name, linewidth=1.5)
        cal_eces.append(compute_binary_ece(p, y, num_bins))

    macro_cal_ece = np.mean(cal_eces)
    axes[1].set_title(f"Calibrated 1D-ResNet (Temperature Scaling)\n(Macro ECE: {macro_cal_ece*100:.2f}%)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Calibrated Predicted Probability", fontsize=10)
    axes[1].set_ylabel("Empirical Disease Prevalence", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(frameon=True, fontsize=8)

    plt.suptitle("CardioAI Clinical Reliability Diagrams: Eliminating Alert Fatigue", fontsize=13, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
