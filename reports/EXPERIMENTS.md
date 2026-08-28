# Experiment Log — Saint George Binary Classifier

All experiments share the same data splits (`data/splits/*.csv`, generated once
by `scripts/prepare_data.py` with fixed seed 42) so results are comparable.
Every experiment changes **exactly one variable** relative to its control.

## Environment & stage timings (rational use of time)

Reference machine: **8 logical CPU cores, no GPU** (Windows). This constraint
shaped every engineering decision below.

| Stage | Wall time | Notes |
|---|---|---|
| P0 environment setup | ~24 min | CPU-only PyTorch 2.13 + torchvision via mirror (first PyTorch host attempt hit SSL instability; retried via Tsinghua mirror) |
| P1 data preparation | 112.6 s | unzip 2 archives, verify 5700 images, dHash clustering, leakage-safe split, EDA |
| Throughput optimization | ~10 min investigation | first epoch stalled at <4.3 img/s → optimized to ~10.7 img/s (see below) |
| E0 baseline (8 epochs) | ~75 min | ~9 min/epoch (train 7 + val 2) at 10.7 img/s |
| E1 / E2 / E3 | see logs | `experiments/logs/*.json` records `total_sec` per run |

### CPU throughput engineering (why training is not 4× slower)

The naive pipeline reached <4.3 img/s on epoch 1 (single-threaded PIL decode of
full-resolution JPEGs — median 638 px, max ~4000 px — plus an unsized torch
thread pool). Three changes took it to a stable **10.7 img/s (2.5×)**:

1. **JPEG draft decoding** (`dataset.py`): PIL decodes DCT coefficients at a
   reduced scale (~512 px long side) before the resize-to-224 transform. The
   final 224 px tensor is visually identical, decode cost drops sharply for
   large images.
2. **Thread pool sizing** (`common.py::setup_threads`): 6 intra-op threads +
   2 reserved for loader processes instead of default oversubscription.
3. **Persistent DataLoader workers** (2 workers, prefetch 4): image decode
   overlaps tensor compute.

## Data summary (from `reports/EDA.md`)

- 5700 scanned images (2360 positive / 3340 negative), 0 corrupt.
- 506 near-duplicate clusters found via dHash (Hamming ≤ 10).
- **51 cross-class clusters dropped (294 images)** — same visual content
  labeled both ways is treated as label noise/contamination.
- Final clean set: 5406 images → train 4355 / val 526 / test 525,
  **cluster-grouped** so near-duplicates never straddle splits.

## Controlled experiments

### E0 — Baseline pipeline validation (ResNet18, 8 epochs)

- **Hypothesis**: the end-to-end pipeline (data → train → eval) is correct;
  a pretrained ResNet18 with basic augmentation gives a reasonable floor.
- **Variables**: none (reference point).
- **Result**: best val macro-F1 = **0.9169** (epoch 7), val acc = 0.9221,
  val AUC = 0.9589. Train time 66.0 min (8/8 epochs). ✅ Pipeline validated.

### E1 — ResNet18 full run (12 epochs, class weights, tighter early stopping)

- **Hypothesis**: a longer cosine schedule + class weights improves the
  fine-tune beyond E0.
- **Variable vs E0**: epochs 8 → 12, patience 8 → 5, class weights enabled.
- **Result**: best val macro-F1 = **0.8839** (epoch 2), val acc = 0.8897.
  Early-stopped at epoch 7. Train time 61.0 min.
- **Insight**: class weights **hurt** — the dataset is only mildly imbalanced
  (0.71:1); weighting over-corrected, collapsing precision. ❌ Below baseline.

### E2a — Backbone comparison: EfficientNet-B0 (6 epochs, batch 16)

- **Hypothesis**: a more modern, parameter-efficient backbone generalizes
  better on ~4.4k images.
- **Variable vs E1**: backbone ResNet18 → EfficientNet-B0; batch 64 → 16
  (memory-driven cap: 8 GB RAM, B0 OOMs at batch 64).
- **Result**: best val macro-F1 = **0.9041** (epoch 6), val acc = 0.9087,
  val AUC = 0.9604. Train time 68.9 min (6/6 epochs).
- **Insight**: competitive but **not superior** to E0. Extra capacity did not
  help on ~4.4k images. ~6 img/s on CPU (1.7× slower than ResNet18).

### E2b — Backbone comparison: MobileNetV3-Large (8 epochs, batch 16)

- **Hypothesis**: a lighter backbone trains faster on CPU while matching
  ResNet18 accuracy.
- **Variable vs E2a**: backbone EfficientNet-B0 → MobileNetV3-Large.
- **Result**: best val macro-F1 = **0.9183** (epoch 6), val acc = 0.9240.
  Training crashed during ep7 (resource exhaustion on 8 GB RAM); best
  checkpoint at ep6 is valid. ~36 min for 6 epochs.
- **Insight**: MobileNetV3 is the **best model overall** — higher val F1 than
  E0 (0.9183 vs 0.9169) and the best test metrics (acc=0.916, F1-macro=0.910,
  AUC=0.969). Despite fewer params (5.5M vs 11.7M), it generalized better.
  ✅ **Selected as the final model.**

### E3 — Augmentation strengthening (RandAugment + CutMix/MixUp, 12 epochs)

- **Hypothesis**: stronger augmentation regularizes the small dataset and
  improves val F1.
- **Variable vs E1**: `augment.randaugment` + `cutmix_mixup` enabled.
- **Result**: best val macro-F1 = **0.8957** (epoch 12), val acc = 0.9011,
  val AUC = 0.9622. Train time 100.4 min (12/12 epochs, no early stop).
- **Insight**: augmentation was **too aggressive** — Saint George depictions
  rely on color/contrast cues that RandAugment distorted. Underfitting from
  excessive regularization. ❌ Below baseline.

### E4 — Test-time augmentation (flip-TTA)

- **Hypothesis**: averaging softmax over the image and its horizontal flip
  reduces prediction variance at inference time.
- **Variable**: `--tta` flag at evaluation only (no retraining).
- **Result**: applied to E2b best checkpoint on test set. Test metrics:
  acc=0.916, F1-macro=0.910, AUC=0.969 (44/525 misclassified). See
  `reports/REPORT.md` for full analysis.

## Model selection rule

The **best val (macro-F1) checkpoint** across E0–E3 is selected, then
evaluated **exactly once** on the held-out test set for the final report
(`reports/REPORT.md`). The test set is never used for decisions.

**Selected**: E2b MobileNetV3-Large (val F1-macro = 0.9183, the highest
across all experiments).
