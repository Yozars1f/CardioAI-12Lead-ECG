"""
CardioAI_12Lead_ECG: Asymmetric Focal Loss (AFL) Implementation.
Tackles severe multi-label clinical class imbalance in 12-lead ECG records.
Author: Youssef Ahmad, MD (Cardiovascular AI Track)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss (AFL) for Multi-Label Cardiac Classification.
    
    References:
    - Ridnik et al., 'Asymmetric Loss For Multi-Label Classification' (ICCV 2021)
    - Clinical Deep Learning ECG AI Formulation
    
    Formula:
      L_+ = (1 - p)^gamma_pos * log(p)
      L_- = (max(p - margin, 0))^gamma_neg * log(1 - max(p - margin, 0))
      Total Loss = - sum [ y * pos_weight * L_+ + (1 - y) * L_- ]
    """
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip_margin: float = 0.05,
        eps: float = 1e-8,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip_margin = clip_margin
        self.eps = eps
        self.reduction = reduction

        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Model raw outputs of shape (B, num_classes)
            targets (torch.Tensor): Multi-hot ground truth labels of shape (B, num_classes)
        Returns:
            torch.Tensor: Computed loss scalar or tensor
        """
        # Sigmoid probabilities
        probs = torch.sigmoid(logits)

        # 1. Positive Loss: y == 1
        probs_pos = probs.clamp(min=self.eps, max=1.0 - self.eps)
        focal_pos = (1.0 - probs_pos) ** self.gamma_pos
        loss_pos = targets * focal_pos * torch.log(probs_pos)

        # Apply class-specific positive weights if provided
        if self.pos_weight is not None:
            loss_pos = loss_pos * self.pos_weight

        # 2. Negative Loss: y == 0 (with asymmetric focusing and margin clipping)
        probs_neg = probs
        if self.clip_margin > 0.0:
            probs_neg = (probs_neg - self.clip_margin).clamp(min=0.0)
        probs_neg = probs_neg.clamp(min=self.eps, max=1.0 - self.eps)

        focal_neg = (probs_neg) ** self.gamma_neg
        loss_neg = (1.0 - targets) * focal_neg * torch.log(1.0 - probs_neg)

        # Total Asymmetric Loss
        loss = - (loss_pos + loss_neg)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
