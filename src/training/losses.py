"""Losses and mixup/cutmix augmentation helpers.

Class-weighted cross-entropy handles the mild positive/negative imbalance.
CutMix/MixUp are provided as toggleable regularizers for experiment E3.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def compute_class_weights(labels: list[int], num_classes: int = 2,
                          device: torch.device | str = "cpu") -> torch.Tensor:
    """Balanced class weights: w_c = total / (num_classes * count_c)."""
    counts = [0] * num_classes
    for l in labels:
        counts[l] += 1
    total = sum(counts) or 1
    weights = [total / (num_classes * max(1, c)) for c in counts]
    return torch.tensor(weights, dtype=torch.float, device=device)


def get_criterion(cfg: dict, train_labels: list[int], device) -> nn.Module:
    tcfg = cfg.get("training", {})
    if tcfg.get("class_weights", False):
        w = compute_class_weights(train_labels, cfg["model"]["num_classes"], device)
    else:
        w = None
    ls = float(tcfg.get("label_smoothing", 0.0))
    return nn.CrossEntropyLoss(weight=w, label_smoothing=ls)


def cutmix_data(x, y, alpha: float = 0.2, device="cpu"):
    """Returns mixed images and the two label tensors with mixing coefficient."""
    lam = torch.distributions.Beta(alpha, alpha).sample().to(device)
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=device)
    y_a, y_b = y, y[idx]

    W, H = x.size(2), x.size(3)
    cut_rat = torch.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx = torch.randint(0, W, (1,)).item()
    cy = torch.randint(0, H, (1,)).item()
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, H)

    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1.0 - float((x2 - x1) * (y2 - y1)) / (W * H)
    return mixed, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam: float) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)
