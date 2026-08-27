#!/usr/bin/env python
"""CLI: prepare the dataset (extract, validate, dedup, split, EDA).

Usage:
    python scripts/prepare_data.py [--experiment baseline_resnet18] [--force]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import (resolve_config, ensure_dirs, set_seed,
                        PROJECT_ROOT, DATA_SPLITS)
from src.data.prepare import prepare_all


def _resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (PROJECT_ROOT / p).resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=None,
                    help="experiment config name (for paths only)")
    ap.add_argument("--pos-zip", default=None)
    ap.add_argument("--neg-zip", default=None)
    ap.add_argument("--force", action="store_true", help="re-extract archives")
    args = ap.parse_args()

    cfg = resolve_config(args.experiment)
    set_seed(cfg["seed"])
    ensure_dirs()

    pos = _resolve(args.pos_zip or cfg["paths"]["pos_zip"])
    neg = _resolve(args.neg_zip or cfg["paths"]["neg_zip"])
    print(f"[prepare] positive={pos}\n          negative={neg}")
    prepare_all(cfg, pos, neg, force=args.force)
    print(f"[prepare] splits written to {DATA_SPLITS}")


if __name__ == "__main__":
    main()
