"""Training loop: optimizer/scheduler, early stopping, checkpointing, logging.

Designed for CPU-first execution: no AMP, gradient clipping is optional but on
by default to keep training stable, and every random source is seeded upstream.
"""
from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD

from ..common import CKPTS_DIR, LOGS_DIR
from ..eval.evaluate import quick_evaluate
from .losses import get_criterion, cutmix_data, mixup_criterion


class Trainer:
    def __init__(self, cfg, model, train_loader, val_loader, device,
                 exp_name: str, train_labels: list[int]):
        self.cfg = cfg
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.exp_name = exp_name
        self.tcfg = cfg["training"]

        self.criterion = get_criterion(cfg, train_labels, device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.use_mix = bool(cfg.get("augment", {}).get("cutmix_mixup", False))
        self.mix_prob = float(cfg.get("augment", {}).get("cutmix_prob", 0.5))
        self.mix_alpha = float(cfg.get("augment", {}).get("mixup_alpha", 0.2))
        self.grad_clip = float(self.tcfg.get("grad_clip", 0.0))

        self.patience = int(self.tcfg.get("early_stopping_patience", 8))
        self.best_f1 = -1.0
        self.wait = 0
        self.history = []
        self.start_time = time.time()

        CKPTS_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    def _build_optimizer(self):
        lr = float(self.tcfg["lr"])
        wd = float(self.tcfg.get("weight_decay", 0.0))
        params = self.model.parameters()
        opt = self.tcfg.get("optimizer", "adamw").lower()
        if opt == "adamw":
            return AdamW(params, lr=lr, weight_decay=wd)
        if opt == "adam":
            return Adam(params, lr=lr, weight_decay=wd)
        if opt == "sgd":
            return SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        raise ValueError(f"Unknown optimizer {opt}")

    def _build_scheduler(self):
        sch = self.tcfg.get("scheduler", "cosine").lower()
        epochs = int(self.tcfg["epochs"])
        if sch == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        if sch == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=int(self.tcfg.get("step_size", 10)),
                gamma=float(self.tcfg.get("gamma", 0.1)))
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)

    def _train_epoch(self) -> float:
        self.model.train()
        running = 0.0
        n = 0
        batches = len(self.train_loader)
        t0 = time.time()
        for i, (imgs, labels) in enumerate(self.train_loader, 1):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            if self.use_mix and random.random() < self.mix_prob:
                mixed, ya, yb, lam = cutmix_data(
                    imgs, labels, alpha=self.mix_alpha, device=self.device)
                out = self.model(mixed)
                loss = mixup_criterion(self.criterion, out, ya, yb, lam)
            else:
                out = self.model(imgs)
                loss = self.criterion(out, labels)
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            running += loss.item() * imgs.size(0)
            n += imgs.size(0)
            if i % 20 == 0 or i == batches:
                rate = n / max(1e-6, time.time() - t0)
                print(f"[train]   batch {i}/{batches} loss={running / max(1, n):.4f} "
                      f"{rate:.1f} img/s", flush=True)
        self.scheduler.step()
        return running / max(1, n)

    def fit(self) -> list[dict]:
        epochs = int(self.tcfg["epochs"])
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch()
            val = quick_evaluate(self.model, self.val_loader, self.device)
            row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_accuracy": round(val["accuracy"], 4),
                "val_precision": round(val["precision"], 4),
                "val_recall": round(val["recall"], 4),
                "val_f1": round(val["f1"], 4),
                "val_f1_macro": round(val["f1_macro"], 4),
                "val_roc_auc": round(val.get("roc_auc", float("nan")), 4),
                "lr": round(self.optimizer.param_groups[0]["lr"], 7),
                "epoch_sec": round(time.time() - t0, 1),
            }
            self.history.append(row)
            improved = val["f1_macro"] > self.best_f1
            if improved:
                self.best_f1 = val["f1_macro"]
                self.wait = 0
                self._save_checkpoint()
            else:
                self.wait += 1
            print(f"[train] {self.exp_name} ep{epoch}/{epochs} "
                  f"loss={train_loss:.4f} val_f1={val['f1']:.4f} "
                  f"val_acc={val['accuracy']:.4f} best={self.best_f1:.4f} "
                  f"patience={self.wait}/{self.patience}")
            if self.wait >= self.patience:
                print(f"[train] early stopping at epoch {epoch}")
                break
        self._save_logs()
        return self.history

    def _save_checkpoint(self) -> None:
        path = CKPTS_DIR / f"{self.exp_name}_best.pth"
        torch.save({
            "exp_name": self.exp_name,
            "backbone": self.cfg["model"]["backbone"],
            "num_classes": self.cfg["model"]["num_classes"],
            "state_dict": self.model.state_dict(),
            "best_f1_macro": self.best_f1,
        }, path)

    def _save_logs(self) -> None:
        total = round(time.time() - self.start_time, 1)
        payload = {
            "exp_name": self.exp_name,
            "config": self.cfg,
            "best_val_f1_macro": self.best_f1,
            "total_sec": total,
            "history": self.history,
        }
        with open(LOGS_DIR / f"{self.exp_name}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        # human-friendly CSV
        if self.history:
            keys = list(self.history[0].keys())
            with open(LOGS_DIR / f"{self.exp_name}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(self.history)
        print(f"[train] logs saved -> {LOGS_DIR / self.exp_name}.json "
              f"(total {total}s)")
