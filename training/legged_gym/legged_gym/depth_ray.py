"""Depth-image to navigation-ray model shared by training and simulation."""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn


class DepthRayNet(nn.Module):
    """Single-channel ResNet-18 regressor for log2 depth-ray distances."""

    def __init__(self, num_rays: int = 21) -> None:
        super().__init__()
        try:
            from torchvision.models import resnet18
        except ImportError as exc:
            raise ImportError(
                "DepthRayNet requires torchvision. Install the ABS training dependencies first."
            ) from exc

        self.num_rays = int(num_rays)
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, self.num_rays)

    def forward(self, depth_log2: torch.Tensor) -> torch.Tensor:
        if depth_log2.ndim == 3:
            depth_log2 = depth_log2.unsqueeze(1)
        if depth_log2.ndim != 4 or depth_log2.shape[1] != 1:
            raise ValueError(
                "DepthRayNet expects [batch, 1, height, width] log2 depth tensors."
            )
        return self.backbone(depth_log2)


def load_depth_ray_model(
    model_path: str, device: torch.device | str, num_rays: int
) -> DepthRayNet:
    """Load a model produced by train_depth_rays.py and validate its ray layout."""
    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError(
            "Depth-ray model not found. Train one with train_depth_rays.py and pass "
            "--depth_model <path>, or use --depth_mode oracle for oracle-ray training."
        )

    checkpoint: Any = torch.load(model_path, map_location=device)
    checkpoint_rays = int(checkpoint.get("num_rays", num_rays)) if isinstance(checkpoint, dict) else num_rays
    if checkpoint_rays != int(num_rays):
        raise ValueError(
            f"Depth-ray model has {checkpoint_rays} outputs but task expects {num_rays}."
        )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = DepthRayNet(num_rays=num_rays).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
