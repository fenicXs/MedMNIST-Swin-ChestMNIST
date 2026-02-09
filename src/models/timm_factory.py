from __future__ import annotations

import timm
import torch


def create_model(name: str, num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    """Create a timm model for multi-label classification."""
    model = timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=3,
    )
    return model
