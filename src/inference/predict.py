"""Inference: load a trained checkpoint and predict on images/folders."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from ..common import LABEL_MAP, get_device
from ..data.dataset import get_transforms
from ..models.builder import build_model


def load_model(cfg: dict, ckpt_path, device=None):
    device = device or get_device()
    model = build_model(cfg["model"]["backbone"],
                        cfg["model"]["num_classes"],
                        cfg["model"]["pretrained"])
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()
    return model


def predict_proba(model, transform, image: Image.Image, device,
                 tta: bool = False) -> float:
    if not tta:
        x = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            return float(torch.softmax(model(x), 1)[0, 1].item())
    ps = []
    for im in (image, image.transpose(Image.FLIP_LEFT_RIGHT)):
        x = transform(im).unsqueeze(0).to(device)
        with torch.no_grad():
            ps.append(torch.softmax(model(x), 1)[0, 1].item())
    return float(sum(ps) / len(ps))


def predict(model, transform, image_path, device, tta: bool = False) -> dict:
    with Image.open(image_path).convert("RGB") as im:
        p = predict_proba(model, transform, im, device, tta)
    contains = p >= 0.5
    return {
        "path": str(image_path),
        "label": LABEL_MAP[1] if contains else LABEL_MAP[0],
        "contains_saint_george": bool(contains),
        "probability": round(p, 4),
    }
