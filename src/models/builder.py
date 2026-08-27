"""Model factory: pretrained backbones with a 2-class head.

Transfer learning from ImageNet weights is used for every backbone, which is
essential given the modest dataset size and CPU-only training budget.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

BACKBONES = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "efficientnet_b0": models.efficientnet_b0,
    "efficientnet_b1": models.efficientnet_b1,
    "efficientnet_b3": models.efficientnet_b3,
    "mobilenet_v3_small": models.mobilenet_v3_small,
    "mobilenet_v3_large": models.mobilenet_v3_large,
    "convnext_tiny": models.convnext_tiny,
}


def build_model(backbone: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone '{backbone}'. Available: {sorted(BACKBONES)}")
    weights = "DEFAULT" if pretrained else None
    model = BACKBONES[backbone](weights=weights)

    if backbone.startswith("resnet"):
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    elif backbone.startswith("efficientnet"):
        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_f, num_classes)
    elif backbone.startswith("mobilenet_v3"):
        in_f = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_f, num_classes)
    elif backbone.startswith("convnext"):
        in_f = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_f, num_classes)
    else:
        raise ValueError(f"Head replacement not implemented for {backbone}")
    return model
