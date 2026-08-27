#!/usr/bin/env python
"""CLI: train a model for one experiment.

Usage:
    python scripts/train.py --experiment baseline_resnet18
    python scripts/train.py --experiment efficientnet_b0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import (resolve_config, ensure_dirs, set_seed, get_device,
                        setup_threads, DATA_SPLITS, PROJECT_ROOT)
from src.data.dataset import ClassificationDataset, get_transforms
from src.data.prepare import prepare_all
from src.data.split import load_split
from src.models.builder import build_model
from src.training.trainer import Trainer


def _resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (PROJECT_ROOT / p).resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=None,
                    help="experiment config (e.g. baseline_resnet18)")
    args = ap.parse_args()

    cfg = resolve_config(args.experiment)
    set_seed(cfg["seed"])
    ensure_dirs()
    device = get_device()

    if not (DATA_SPLITS / "train.csv").exists():
        print("[train] splits missing -> running prepare_data first")
        prepare_all(cfg, _resolve(cfg["paths"]["pos_zip"]),
                    _resolve(cfg["paths"]["neg_zip"]))

    train_recs = load_split(DATA_SPLITS / "train.csv")
    val_recs = load_split(DATA_SPLITS / "val.csv")
    train_labels = [r["label"] for r in train_recs]

    bs = int(cfg["data"]["batch_size"])
    # CPU training: 2 persistent loader workers overlap PIL JPEG decoding
    # (draft-mode) with tensor compute; threads sized in setup_threads().
    nw = 2
    setup_threads(leave_for_loaders=nw)
    train_ds = ClassificationDataset(train_recs, get_transforms(cfg, "train"))
    val_ds = ClassificationDataset(val_recs, get_transforms(cfg, "val"))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, drop_last=False,
                              persistent_workers=nw > 0,
                              prefetch_factor=4 if nw > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw,
                            persistent_workers=nw > 0,
                            prefetch_factor=4 if nw > 0 else None)

    model = build_model(cfg["model"]["backbone"],
                        cfg["model"]["num_classes"],
                        cfg["model"]["pretrained"])
    exp_name = args.experiment or f"base_{cfg['model']['backbone']}"

    print(f"[train] experiment={exp_name} backbone={cfg['model']['backbone']} "
          f"device={device} train={len(train_recs)} val={len(val_recs)}")
    trainer = Trainer(cfg, model, train_loader, val_loader, device,
                      exp_name, train_labels)
    trainer.fit()


if __name__ == "__main__":
    main()
