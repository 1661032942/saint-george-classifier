# Saint George Image Classifier (Bifu test task)

A reproducible binary image-classification pipeline that decides whether an
image **contains Saint George** (positive) or **does not** (negative).

> Main goal (per `测试描述.txt`): obtain the highest-quality classification
> model possible, using any legal/ethical method. This repository emphasizes
> *both* model quality **and** engineering rigor — a clear, reproducible
> pipeline with experiment tracking and reporting.

---

## 1. Task & approach at a glance

| Aspect | Choice | Why |
|--------|--------|-----|
| Problem | Binary image classification (semantic) | "Saint George" is a concept, not a pixel mask |
| Backbones | ResNet18 / EfficientNet-B0 / MobileNetV3 (ImageNet-pretrained) | Transfer learning is essential at this data scale & CPU budget |
| Anti-leakage | Perceptual hash (dHash) clustering → grouped split | Near-duplicates never cross train/val/test |
| Imbalance | Class-weighted loss | Mild (~0.71:1) positive:negative ratio |
| Reproducibility | Fixed seeds + config-driven + logged experiments | Anyone can re-run and reproduce |

## 2. Directory structure

```
saint_george_classifier/
├── README.md                 # this file
├── requirements.txt          # pinned deps (B7)
├── environment.yml           # conda env (B7)
├── Dockerfile                # CPU image (B7)
├── configs/
│   ├── base.yaml             # all defaults
│   └── experiments/          # E0 baseline, E2 backbone comparisons
├── src/
│   ├── common.py             # paths, seeding, device, config merge
│   ├── data/                 # prepare, dataset, split
│   ├── models/builder.py     # backbone factory
│   ├── training/             # trainer, losses (class-weights, CutMix)
│   ├── eval/evaluate.py       # metrics, confusion matrix, misclassified export
│   └── inference/predict.py   # single/folder prediction
├── scripts/                  # CLI entry points
├── data/                     # raw (extracted), splits (generated)
├── experiments/              # logs / checkpoints / metrics / misclassified
└── reports/                  # EDA.md, EXPERIMENTS.md, REPORT.md
```

## 3. Setup

```bash
# 1. (recommended) create & activate a venv
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
# 2. install CPU PyTorch + deps
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
#    (in mainland China, the Tsinghua mirror is more reliable:
#     -i https://pypi.tuna.tsinghua.edu.cn/simple for the non-torch deps,
#     --index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch/wheel/cpu for torch)
```

> The reference environment for this build is **CPU-only**; training still runs
> (only slower). On a GPU machine the same code uses CUDA automatically.

## 4. Run the pipeline (one command per stage)

Place the two archives next to the project (or pass `--pos-zip` / `--neg-zip`):

```bash
# Stage 1 — prepare data (extract, validate, dedup, split, EDA)
python scripts/prepare_data.py

# Stage 2 — train an experiment (checkpoint + metrics logged)
python scripts/train.py --experiment baseline_resnet18

# Stage 3 — evaluate the BEST checkpoint on the TEST set (once)
python scripts/evaluate.py --experiment baseline_resnet18 --split test

# Stage 4 — inference on new images / folders
python scripts/predict.py --experiment baseline_resnet18 --input path/to/image.jpg
python scripts/predict.py --experiment baseline_resnet18 --input img1.jpg img2.jpg --tta
python scripts/predict.py --experiment baseline_resnet18 --input folder/ --tta
```

Available experiments (configs under `configs/experiments/`):
`baseline_resnet18` (E0), `e1_resnet18_full` (E1), `efficientnet_b0` (E2a),
`mobilenet_v3` (E2b), `e3_augment` (E3). E4 (test-time augmentation) is a
flag: `--tta` on `scripts/evaluate.py` / `scripts/predict.py`.

## 5. Architecture

```
Image ─▶ transforms (resize / RandomResizedCrop / flip / jitter / normalize)
      └▶ Backbone (ImageNet weights) ─▶ 2-class head
                                        └▶ CrossEntropyLoss (+class weights / label smoothing)
                                            └▶ AdamW + CosineAnnealing / EarlyStopping
```

The training augmentations (`RandomResizedCrop` with a tight scale range) are
deliberately chosen to make the model robust to the wide range of scales at
which Saint George appears (paintings, statues, badges, stained glass).

## 6. Anti-leakage strategy

1. **Perceptual hashing** (`dHash`, 64-bit) for every image.
2. **Union-find clustering** of near-duplicates (Hamming distance ≤ threshold).
3. **Cross-class clusters are dropped** — if the same visual content appears
   under both labels it is ambiguous/contaminated.
4. **Grouped split**: a whole near-duplicate cluster is assigned to a single
   split (train/val/test = 80/10/10, stratified by class).

This guarantees a clean, honest estimate of generalization.

## 7. Experiments & results

See [`reports/EXPERIMENTS.md`](reports/EXPERIMENTS.md) for the controlled
experiments (hypotheses → variables → results → conclusions) and
[`reports/REPORT.md`](reports/REPORT.md) for the final metrics, confusion
matrix, misclassification analysis, and improvement directions.

Quick summary (filled in by the scripts; see `experiments/logs/*.json`):

| Experiment | Backbone | Best val F1-macro | Val Acc | Val AUC | Train time |
|------------|----------|-------------------|---------|---------|------------|
| **E0** | ResNet18 (8 ep, baseline) | **0.9169** | 0.9221 | 0.9589 | 66 min |
| E1 | ResNet18 (12 ep, cls weights) | 0.8839 | 0.8897 | 0.9455 | 61 min |
| E2a | EfficientNet-B0 (6 ep, bs16) | 0.9041 | 0.9087 | 0.9604 | 69 min |
| E2b | MobileNetV3-L (8 ep, bs16) | running | — | — | — |
| E3 | ResNet18 + RandAug + CutMix | 0.8957 | 0.9011 | 0.9622 | 100 min |

**Final test-set result** (E0 best checkpoint + flip-TTA, evaluated once):
Accuracy 0.893 · F1-macro 0.886 · ROC-AUC 0.948 · 56 misclassified / 525.
See `reports/REPORT.md` for full analysis.

## 8. Reproducibility & FAQ

- **Deterministic**: every script calls `set_seed(cfg.seed)` before any work.
- **Config-driven**: change `configs/*.yaml` instead of editing code.
- **Re-run everything**: `prepare_data.py` → `train.py` → `evaluate.py`.
- *"Why is training slow?"* — the reference machine has no GPU. Use a GPU host
  or reduce `data.image_size` / `training.epochs` in `configs/base.yaml`.
- *"Can I add a backbone?"* — add it to `BACKBONES` in `src/models/builder.py`
  (it must expose a replaceable classification head).

## 9. Deliverables checklist

- [x] Python code (data prep / train / inference / eval)
- [x] README (install / run / architecture)
- [x] Results / logs / checkpoints under `experiments/`
- [x] Experiment report (`reports/EXPERIMENTS.md`)
- [x] Method rationale (per-experiment notes)
- [x] Final report (`reports/REPORT.md`: metrics, misclassifications, improvements)
- [x] Environment files (`requirements.txt`, `environment.yml`, `Dockerfile`)
