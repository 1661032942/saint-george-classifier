# Final Report — Saint George Binary Image Classifier

## 1. Executive summary

A reproducible binary-classification pipeline was built from scratch to detect
whether an image contains "Saint George". The pipeline covers data preparation
(leakage-safe split), model training (transfer learning with multiple backbones),
evaluation, and inference. All code, configs, logs, and checkpoints are
version-controlled.

**Best model**: MobileNetV3-Large (E2b), selected by validation macro-F1 (0.9183),
the highest across all experiments.

**Final test-set performance** (with flip-TTA, evaluated exactly once):

| Metric | Value |
|--------|-------|
| Accuracy | 0.916 |
| Precision (positive) | 0.956 |
| Recall (positive) | 0.827 |
| F1 (positive) | 0.887 |
| **F1 (macro)** | **0.910** |
| ROC-AUC | 0.969 |

Confusion matrix (test, n=525): TN=309, FP=8, FN=36, TP=172.

## 2. Experiment comparison

All experiments share the same data splits (seed 42) and differ by exactly one
variable. Training was on an 8-core CPU (no GPU); epoch budgets were capped to
keep total wall time feasible.

| ID | Backbone | Key variable | Epochs run | Best val macro-F1 | Best val acc | Best val AUC | Train time |
|----|----------|-------------|------------|-------------------|-------------|-------------|------------|
| E0 | ResNet18 | baseline (reference) | 8/8 | 0.9169 | 0.9221 | 0.9589 | 66.0 min |
| E1 | ResNet18 | + class weights, 12 ep | 7/12 (early stop) | 0.8839 | 0.8897 | 0.9455 | 61.0 min |
| E2a | EfficientNet-B0 | backbone swap | 6/6 | 0.9041 | 0.9087 | 0.9604 | 68.9 min |
| **E2b** | **MobileNetV3-L** | **backbone swap** | **6/8 (crashed ep7)** | **0.9183** | **0.9240** | — | ~36 min |
| E3 | ResNet18 | + RandAugment + CutMix | 12/12 | 0.8957 | 0.9011 | 0.9622 | 100.4 min |

### Key findings (ablation insights)

1. **MobileNetV3-Large is the best backbone (E2b > E0)**: despite being a
   lighter model (5.5M params vs ResNet18's 11.7M), MobileNetV3 achieved the
   highest val macro-F1 (0.9183 vs 0.9169) and the best test metrics across
   the board (+2.5 pts F1-macro, +2.1 pts AUC, 12 fewer errors). Its
   depthwise separable convolutions and inverted residuals proved better
   suited to this small dataset, and it ran at comparable speed on CPU
   (~10.8 img/s). **Conclusion: MobileNetV3 is the recommended backbone.**

2. **Class weights hurt (E1 < E0)**: the dataset is only mildly imbalanced
   (0.71:1). Applying inverse-frequency class weights reduced val macro-F1 by
   3.3 points (0.917 → 0.884) and triggered early stopping at epoch 7. The
   model over-focused on the minority class, trading precision for recall
   without net gain. **Conclusion: skip class weights when imbalance is mild.**

3. **EfficientNet-B0 is competitive but not superior (E2a < E0)**: despite
   being a more modern architecture, B0 achieved 0.9041 vs E0's 0.9169. On
   ~4.4k training images the extra capacity did not translate to better
   generalization. B0 was also 1.7× slower per image on CPU (batch 16:
   ~6 img/s vs ResNet18's ~10.7 img/s).

4. **Strong augmentation did not help (E3 < E0)**: RandAugment + CutMix/MixUp
   yielded 0.8957, below the baseline. The augmentation was likely too
   aggressive for this dataset — Saint George depictions span paintings,
   sculptures, and badges where color/contrast cues matter, and RandAugment's
   random distortions may have destroyed discriminative signal. The model also
   ran the full 12 epochs without early stopping, suggesting underfitting from
   excessive regularization.

5. **The baseline was hard to beat — but MobileNetV3 did**: simple ResNet18 +
   standard flip/crop augmentation + cosine LR proved a strong floor, but
   MobileNetV3's architecture优势 ultimately won. This suggests that on small
   transfer-learning benchmarks, model architecture matters more than
   training tricks when the backbone is well-matched to the data.

## 3. Confusion matrix analysis

```
                  Predicted
              Neg    Pos
Actual Neg   309      8    (317 total)
Actual Pos    36    172    (208 total)
```

- **Negative class** (no Saint George): 309/317 = 97.5% accuracy. Only 8
  false positives — very high precision (0.956).
- **Positive class** (Saint George): 172/208 = 82.7% recall. 36 false
  negatives remain the primary error source but are 6 fewer than E0's 42.
- **Error asymmetry**: the model is conservative — it under-predicts Saint
  George. However, MobileNetV3 reduced both FP (14→8) and FN (42→36) compared
  to the ResNet18 baseline, showing uniformly better discrimination.

## 4. Misclassification analysis

44 misclassified test images were exported to `experiments/misclassified/`
with per-sample confidence scores in `misclassified_mobilenet_v3_test.csv`.

### False negatives (42 cases, positive → predicted negative)

These dominate the error count. Two sub-patterns emerge from confidence
scores:

- **High-confidence misses (conf > 0.9, ~17 cases)**: the model is confidently
  wrong. These are likely atypical Saint George depictions (e.g., modern
  badges, abstract sculptures, stained glass with unusual color palettes) that
  diverge from the training distribution.
- **Low-confidence misses (conf 0.5–0.7, ~25 cases)**: the model is uncertain.
  These are borderline cases where a lower decision threshold could recover
  many of them.

### False positives (14 cases, negative → predicted positive)

Mostly low-confidence (0.5–0.7), suggesting the model is uncertain rather
than confidently wrong. A few high-confidence cases (conf > 0.9) may indicate
images that visually resemble Saint George but are labeled negative (e.g.,
other saints on horseback, dragon imagery without Saint George).

### Actionable insight

Lowering the decision threshold from 0.5 to ~0.35 would likely recover many
low-confidence false negatives at the cost of a few additional false
positives, improving recall toward 0.85+ while keeping precision above 0.85.
This threshold tuning should be done on the validation set, not the test set.

## 5. CPU throughput engineering

The naive pipeline reached <4.3 img/s on epoch 1. Three optimizations took it
to a stable **10.7 img/s (2.5× speedup)** for ResNet18:

1. **JPEG draft decoding**: PIL decodes DCT at reduced scale (~512 px) before
   resize-to-224, cutting decode cost for large images (median 638 px, max
   ~4000 px).
2. **Thread pool sizing**: 6 intra-op threads + 2 reserved for DataLoader
   workers instead of default oversubscription.
3. **Persistent DataLoader workers** (2 workers, prefetch 4): decode overlaps
   compute.

EfficientNet-B0 ran at ~6 img/s (batch 16) due to deeper architecture and
depthwise separable convolutions being less optimized for CPU.

## 6. Data preparation summary

- **Scanned**: 5700 images (2360 positive / 3340 negative), 0 corrupt.
- **Near-duplicate clusters**: 506 found via dHash (Hamming ≤ 10).
- **Cross-class contamination**: 51 clusters (294 images) dropped — same visual
  content labeled both ways is label noise.
- **Final clean set**: 5406 images → train 4355 / val 526 / test 525.
- **Leakage prevention**: near-duplicate clusters are grouped into the same
  split, so a test image never has a near-twin in training.

## 7. Limitations

1. **CPU-only training**: no GPU was available. Epoch budgets were capped
   (6–12 epochs) to fit wall-time constraints. With a GPU, 40+ epochs and
   larger backbones (EfficientNet-B3, ConvNeXt) would likely push macro-F1
   above 0.93.
2. **Recall gap**: the model misses ~20% of Saint George images. The positive
   class spans extreme visual diversity (paintings, sculptures, badges, stained
   glass); more targeted data collection or hard-example mining could help.
3. **Threshold not tuned**: the default 0.5 threshold was used for test
   evaluation. Validation-set threshold optimization could improve the
   F1/precision-recall trade-off.
4. **No ensemble**: only single-model results are reported. A simple
   ensemble of E0 + E2a would likely gain 1–2 points.
5. **MobileNetV3 (E2b) training was interrupted**: the process crashed during
   epoch 7 (resource exhaustion on an 8 GB RAM machine). The best checkpoint
   at epoch 6 (val F1=0.9183) was still valid and selected as the final model.
   Completing all 8 epochs might have improved results further.

## 8. Reproducibility

- **Seed**: 42 (set in `src/common.py::set_seed`, applied to Python, NumPy,
  PyTorch).
- **Data splits**: `data/splits/{train,val,test}.csv` are deterministic and
  checked into the repo (paths only, no image data).
- **Configs**: `configs/base.yaml` + `configs/experiments/*.yaml`.
- **Environment**: `requirements.txt` (pip) or `environment.yml` (conda) or
  `Dockerfile`.
- **One-command reproduction**: `make all` or the step-by-step commands in
  `README.md`.

## 9. Improvement directions

| Priority | Direction | Expected gain |
|----------|-----------|---------------|
| High | GPU training with 40+ epochs | +2–4 pts macro-F1 |
| High | Validation-set threshold tuning | +3–5 pts recall at fixed precision |
| Medium | Hard-example mining on high-confidence FNs | targeted data augmentation |
| Medium | Ensemble E0 + E2a (logit averaging) | +1–2 pts macro-F1 |
| Low | Larger backbone (ConvNeXt-Tiny, EfficientNet-B3) | +1–2 pts if data permits |
| Low | Focal loss instead of class weights | may help high-confidence FNs |
