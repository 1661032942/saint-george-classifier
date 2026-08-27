"""Evaluation: metrics, confusion matrix, and misclassification export.

Unified evaluation protocol used both for per-epoch validation monitoring and
for the single final test-set evaluation required by the task.
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

from ..common import LABEL_MAP, METRICS_DIR, MISCLS_DIR

_SOFTMAX = torch.nn.Softmax(1)


def compute_metrics(y_true, y_pred, y_score) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["roc_auc"] = float("nan")
    return out


def _collect(model, loader, device, tta: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference over the loader.

    With ``tta=True`` the class probabilities are averaged over the original
    image and its horizontal flip (test-time augmentation, E4).
    """
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            prob = _SOFTMAX(model(imgs))[:, 1]
            if tta:
                prob = 0.5 * (prob + _SOFTMAX(model(torch.flip(imgs, dims=[3])))[:, 1])
            prob = prob.cpu().numpy()
            pred = (prob >= 0.5).astype(int)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(pred.tolist())
            y_score.extend(prob.tolist())
    return np.asarray(y_true), np.asarray(y_pred), np.asarray(y_score)


def quick_evaluate(model, loader, device) -> dict:
    """Metrics only (no plotting / export) — for per-epoch validation."""
    y_true, y_pred, y_score = _collect(model, loader, device)
    return compute_metrics(y_true, y_pred, y_score)


def _save_confusion(cm, path: Path, tag: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels([LABEL_MAP[0], LABEL_MAP[1]])
    ax.set_yticklabels([LABEL_MAP[0], LABEL_MAP[1]])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _export_misclassified(loader, y_true, y_pred, y_score, max_n: int, tag: str) -> list[dict]:
    """Copy misclassified images (with annotations) for failure-mode analysis."""
    MISCLS_DIR.mkdir(parents=True, exist_ok=True)
    records = loader.dataset.records
    rows = []
    copied = 0
    csv_path = MISCLS_DIR / f"misclassified_{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["exported_image", "original_path", "true_label",
                    "pred_label", "confidence"])
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                rec = records[i]
                conf = float(y_score[i] if y_pred[i] == 1 else 1.0 - y_score[i])
                fname = f"mis_{copied}_{LABEL_MAP[y_true[i]]}_as_{LABEL_MAP[y_pred[i]]}.jpg"
                dst = MISCLS_DIR / fname
                try:
                    if copied < max_n:
                        with Image.open(rec["path"]) as im:
                            im.convert("RGB").save(dst, "JPEG", quality=85)
                        copied += 1
                    w.writerow([fname, rec["path"], LABEL_MAP[y_true[i]],
                               LABEL_MAP[y_pred[i]], f"{conf:.4f}"])
                    rows.append({"file": fname, "true": LABEL_MAP[y_true[i]],
                                 "pred": LABEL_MAP[y_pred[i]], "conf": conf})
                except Exception:
                    continue
    return rows


def evaluate(model, loader, device, cfg: dict, tag: str = "test",
             export_misclassified: bool = True, tta: bool = False) -> dict:
    """Full evaluation: metrics + confusion-matrix PNG + misclassification export."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    y_true, y_pred, y_score = _collect(model, loader, device, tta=tta)
    metrics = compute_metrics(y_true, y_pred, y_score)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = cm.tolist()
    _save_confusion(cm, METRICS_DIR / f"confusion_matrix_{tag}.png", tag)

    if export_misclassified and cfg.get("eval", {}).get("export_misclassified", True):
        max_n = int(cfg.get("eval", {}).get("max_misclassified", 200))
        metrics["misclassified"] = _export_misclassified(
            loader, y_true, y_pred, y_score, max_n, tag)
        metrics["n_misclassified"] = sum(1 for a, b in zip(y_true, y_pred) if a != b)

    return metrics
