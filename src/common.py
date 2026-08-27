"""Shared utilities: project paths, random seeding, device, config loading.

All paths are resolved relative to the project root (the directory that
contains ``src/``, ``scripts/``, ``configs/``, ``data/`` ...). Reproducibility
is enforced via :func:`set_seed`, which fixes every random source used by the
pipeline.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_SPLITS = PROJECT_ROOT / "data" / "splits"
EXPERIMENTS = PROJECT_ROOT / "experiments"
LOGS_DIR = EXPERIMENTS / "logs"
CKPTS_DIR = EXPERIMENTS / "checkpoints"
METRICS_DIR = EXPERIMENTS / "metrics"
MISCLS_DIR = EXPERIMENTS / "misclassified"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Class convention: 1 = contains Saint George (positive), 0 = does not (negative).
LABEL_MAP = {0: "negative", 1: "positive"}
POS_LABEL = 1
NEG_LABEL = 0

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"


def ensure_dirs() -> None:
    """Create all working directories if missing."""
    for d in (DATA_RAW, DATA_PROCESSED, DATA_SPLITS, LOGS_DIR,
              CKPTS_DIR, METRICS_DIR, MISCLS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    """Fix every random source for full reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU.

    Note: the reference environment for this task is CPU-only; training still
    runs correctly (only slower) on CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_threads(leave_for_loaders: int = 2) -> None:
    """Explicitly size the torch intra-op thread pool for CPU training.

    On the CPU-only reference machine (8 logical cores) the default pool can
    oversubscribe once DataLoader workers are enabled; reserving cores for the
    loader processes keeps image decoding and tensor compute overlapped.
    """
    total = os.cpu_count() or 1
    n = max(1, total - leave_for_loaders)
    torch.set_num_threads(n)
    torch.set_num_interop_threads(1)


def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    """Deep-merge ``override`` onto ``base`` (override wins)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_configs(out[k], v)
        else:
            out[k] = v
    return out


def resolve_config(experiment: str | None = None) -> dict:
    """Load base config and optionally merge an experiment override."""
    cfg = load_config(DEFAULT_CONFIG)
    if experiment:
        exp_path = PROJECT_ROOT / "configs" / "experiments" / f"{experiment}.yaml"
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {exp_path}")
        cfg = merge_configs(cfg, load_config(exp_path))
    return cfg
