"""Small 1D-CNN modulation classifier.

Trained on IQ samples — operates on complex signals represented as
[B, 2, T] tensors (I/Q channels). Small enough to train on a laptop,
big enough that SAE analysis is non-trivial.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ModClassifier(nn.Module):
    def __init__(self, n_classes: int = 11, n_samples: int = 1024):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 64, 7, padding=3), nn.ReLU(),
            nn.Conv1d(64, 64, 7, padding=3), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(),
            nn.Conv1d(128, 128, 5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        feats = self.features(x)
        return self.head(feats)

    def residual_stream(self, x):
        """Hook point for SAE training — last pre-head activation."""
        return self.features(x)
