"""
CardioAI_12Lead_ECG: 1D-ResNet Neural Network Architecture.
Clinical-grade Residual Network designed for 12-Lead ECG Signal Classification.
Author: Youssef Ahmad, MD (Cardiovascular AI Track)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ConvBlock1D(nn.Module):
    """
    Standard 1D Convolutional block with Batch Normalization and ReLU.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = False
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block with identity skip connection:
    F(x) + x, preserves high-frequency QRS & ST morphology.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dropout_p: float = 0.2
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_p)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size,
            stride=1, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut / Skip Connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class ECGResNet1D(nn.Module):
    """
    1D-ResNet Architecture for 12-Lead ECG Multi-Label Diagnosis.
    Input Contract: (Batch_Size, in_channels=12, seq_length=1000)
    Output Contract: (Batch_Size, num_classes=5) [Logits]
    """
    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        base_filters: int = 64,
        kernel_size: int = 7,
        dropout_p: float = 0.2,
        layers: Optional[List[int]] = None
    ):
        super().__init__()
        # Default layers: 4 residual stages [2, 2, 2, 2] -> 18-layer equivalent
        if layers is None:
            layers = [2, 2, 2, 2]

        self.in_channels = in_channels
        self.num_classes = num_classes

        # 1. Initial Stem Convolution (Captures broad temporal baseline and rhythm)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # 2. Residual Stages (Progressive channel expansion and temporal downsampling)
        current_channels = base_filters
        self.stages = nn.ModuleList()

        for stage_idx, num_blocks in enumerate(layers):
            out_channels = base_filters * (2 ** stage_idx)
            stride = 1 if stage_idx == 0 else 2
            
            blocks = []
            # First block in stage handles channel change and stride downsampling
            blocks.append(
                ResidualBlock1D(
                    current_channels, out_channels,
                    kernel_size=kernel_size, stride=stride, dropout_p=dropout_p
                )
            )
            # Subsequent blocks keep channels and resolution
            for _ in range(1, num_blocks):
                blocks.append(
                    ResidualBlock1D(
                        out_channels, out_channels,
                        kernel_size=kernel_size, stride=1, dropout_p=dropout_p
                    )
                )
            current_channels = out_channels
            self.stages.append(nn.Sequential(*blocks))

        # 3. Global Pooling & Multi-Label Head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(current_channels, num_classes)
        )

        self._initialize_weights()

    @property
    def classifier(self):
        """Property alias for self.fc for backward compatibility."""
        return self.fc

    def _initialize_weights(self):
        """Kaiming (He) normal initialization suited for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def load_state_dict(self, state_dict, strict=True):
        """Custom state_dict loader that remaps 'classifier' <-> 'fc' automatically."""
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith("classifier."):
                new_state[k.replace("classifier.", "fc.")] = v
            else:
                new_state[k] = v
        return super().load_state_dict(new_state, strict=strict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Shape (B, 12, 1000)
        Returns:
            torch.Tensor: Logits of shape (B, 5)
        """
        out = self.stem(x)
        for stage in self.stages:
            out = stage(out)
        out = self.global_pool(out)      # Shape: (B, Channels, 1)
        out = out.flatten(1)             # Shape: (B, Channels)
        logits = self.fc(out)            # Shape: (B, 5)
        return logits

    def get_features_and_logits(self, x: torch.Tensor) -> tuple:
        """
        Helper method specifically designed for Phase 3 (1D Grad-CAM Explainability).
        Returns both the feature maps of the final convolutional stage and the logits.
        """
        out = self.stem(x)
        for stage in self.stages:
            out = stage(out)
        feature_maps = out               # (B, Channels, L_reduced)
        pooled = self.global_pool(out).flatten(1)
        logits = self.fc(pooled)
        return feature_maps, logits


def build_cardio_resnet1d(num_classes: int = 5, in_channels: int = 12) -> ECGResNet1D:
    """Factory helper to build clinical standard 1D-ResNet."""
    return ECGResNet1D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=64,
        kernel_size=7,
        dropout_p=0.2,
        layers=[2, 2, 2, 2]
    )
