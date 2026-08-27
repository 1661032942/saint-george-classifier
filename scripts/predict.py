#!/usr/bin/env python
"""CLI: predict Saint George presence for an image or a folder.

Usage:
    python scripts/predict.py --experiment baseline_resnet18 --input path/to/img.jpg
    python scripts/predict.py --experiment baseline_resnet18 --input path/to/folder --tta
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import (resolve_config, ensure_dirs, get_device, CKPTS_DIR)
from src.data.dataset import get_transforms
from src.inference.predict import load_model, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True,
                    help="experiment name (must match the trained checkpoint)")
    ap.add_argument("--input", required=True, nargs="+",
                    help="image file(s), folder(s), or a mix")
    ap.add_argument("--tta", action="store_true", help="test-time augmentation")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    cfg = resolve_config(args.experiment)
    ensure_dirs()
    device = get_device()

    ckpt = CKPTS_DIR / f"{args.experiment}_best.pth"
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}. Train first.")

    model = load_model(cfg, ckpt, device)
    transform = get_transforms(cfg, "val")

    files = []
    for raw in args.input:
        inp = Path(raw)
        if inp.is_dir():
            files += [p for p in sorted(inp.rglob("*"))
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
        else:
            files.append(inp)

    results = []
    for f in files:
        try:
            r = predict(model, transform, f, device, tta=args.tta)
            r["threshold"] = args.threshold
            results.append(r)
            print(f"{r['path']} -> {r['label']} (p={r['probability']})")
        except Exception as e:
            print(f"{f}: ERROR {e}")

    out_path = Path("prediction_results.json")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[predict] {len(results)} result(s) -> {out_path}")


if __name__ == "__main__":
    main()
