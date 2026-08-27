#!/usr/bin/env python
"""CLI: evaluate a trained checkpoint on a split (default: test).

The test set is evaluated exactly ONCE for the final report, as required by the
task (to avoid validation overfitting). Use --split val for monitoring.

Usage:
    python scripts/evaluate.py --experiment baseline_resnet18 --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import (resolve_config, ensure_dirs, set_seed, get_device,
                        DATA_SPLITS, CKPTS_DIR, METRICS_DIR)
from src.data.dataset import ClassificationDataset, get_transforms
from src.data.split import load_split
from src.eval.evaluate import evaluate
from src.inference.predict import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True,
                    help="experiment name (must match training checkpoint)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--tta", action="store_true", help="test-time augmentation")
    ap.add_argument("--no-export", action="store_true",
                    help="do not export misclassified images")
    args = ap.parse_args()

    cfg = resolve_config(args.experiment)
    set_seed(cfg["seed"])
    ensure_dirs()
    device = get_device()

    ckpt = CKPTS_DIR / f"{args.experiment}_best.pth"
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}. Train first.")

    recs = load_split(DATA_SPLITS / f"{args.split}.csv")
    ds = ClassificationDataset(recs, get_transforms(cfg, "val"))
    loader = DataLoader(ds, batch_size=int(cfg["data"]["batch_size"]),
                        shuffle=False, num_workers=0)

    model = load_model(cfg, ckpt, device)
    metrics = evaluate(model, loader, device, cfg, tag=f"{args.experiment}_{args.split}",
                       export_misclassified=not args.no_export, tta=args.tta)

    out = {k: v for k, v in metrics.items() if k not in ("misclassified", "confusion_matrix")}
    out["confusion_matrix"] = metrics.get("confusion_matrix")
    print(json.dumps(out, indent=2, ensure_ascii=False))

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / f"{args.experiment}_{args.split}_metrics.json",
              "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[evaluate] metrics saved -> "
          f"{METRICS_DIR / f'{args.experiment}_{args.split}_metrics.json'}")


if __name__ == "__main__":
    main()
