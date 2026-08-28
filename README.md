# Saint George 图像分类器（Bifu 测试任务）

一个可复现的图像二分类流水线：判断一张图像**是否包含圣乔治**（正类）。
目标（见 `测试描述.txt`）：用合法/道德范围内的任意方法，获得尽可能高质量的分类模型。

> **最终结果**：E2b MobileNetV3-Large + flip-TTA，测试集（一次性评估）
> **Accuracy 0.916 · F1-macro 0.910 · ROC-AUC 0.969**，525 张中误分类 44 张。

- 任务定义 | 二分类（语义层面） | "圣乔治"是概念而非像素掩码
- 方法主线 | 迁移学习（ImageNet 预训练骨干 + 微调） | 5.4k 数据量与 CPU 预算下的最优解
- 防泄漏 | dHash 感知哈希聚簇 + 分组划分 | 近重复样本绝不跨 train/val/test
- 类不平衡 | 类别加权损失（对照实验后未启用） | 正负比约 0.71:1，属轻度
- 可复现 | 固定随机种子 + 配置驱动 + 实验日志 | 任何人可重跑复现

---

## 1. 为什么最终选择 E2b MobileNetV3-Large？

选型不是先验偏好，而是**论文依据 + 受控实验**双重验证的结果。

### 1.1 论文层面：MobileNetV3 的精度-效率优势

以《Searching for MobileNetV3》（arXiv:1905.02244）及配套讲解材料中的图表为依据：

**图 1 · 延迟-准确率 Pareto 对比图（MobileNetV2 vs V3）**
横坐标为 Google Pixel 1 上的推理延迟，纵坐标为 ImageNet Top-1 准确率。
V3 的曲线整体位于 V2 左上方——**相同延迟下准确率更高，相同准确率下延迟更低**。
论文给出量化结论：

- MobileNetV3-**Large** 较 V2：Top-1 提升 **+3.2%**，耗时减少 **20%**；
- MobileNetV3-**Small** 与同延迟 V2 相比：准确率高 **+6.6%**。

**图 2 · Table 1 网络结构表（Large 版）**
逐层给出 `Input / Operator(bneck×k×k) / exp_size / #out / SE / NL / stride`，
输入 224×224×3，骨干为 15 个倒残差块（bneck）+ 头尾卷积。关键列的含义：
`SE` = 该块是否带通道注意力；`NL` = 激活函数（HS=h-swish，RE=ReLU）；
`s` = DW 卷积步距。这正是本项目微调所用的结构。

**图 3 · SE 模块结构图**
每个 channel 经全局平均池化 → FC（降到 1/4 通道数，ReLU）→ FC（还原通道数，
h-sigmoid）→ 与原特征逐通道相乘。即：**给重要通道加权、弱通道降权**。
对"圣乔治出现在画面局部（画作/雕像/徽章/彩窗）"的任务，通道注意力
能自适应地突出判别性特征。

**图 4 · 耗时层重设计（Original vs Efficient Last Stage）**
首层卷积核 32→16（精度不变，省 2ms）；尾部 4 层卷积精简为"卷积→池化→两层卷积"
（省 7ms，约占推理全程 11%）。

三大创新总结：**① bneck 内嵌 SE 通道注意力；② h-swish/h-sigmoid 硬饱和激活**
（逼近 swish 精度但无幂运算、对量化友好）；**③ NAS（MnasNet 强化学习多目标
搜索 accuracy+latency）+ NetAdapt 逐层精调**。DW 卷积 + SE 的组合使 V3-Large
在 CPU 上的计算量远小于同精度卷积网络——这是本项目 CPU-only 预算下的决定性优势。

> 已知局限（材料第 3 节）：V3 针对 Pixel 硬件优化，在其他设备上未必达最佳
> 性价比。本项目实测（Intel CPU + PyTorch）中 V3-Large 仍是精度-速度最优，未受此影响。

### 1.2 项目实验：5 组受控对比，E2b 双重胜出

统一协议（同数据划分/训练循环/调参规则，每次只变一个变量），以**验证集
F1-macro** 为唯一选模标准，test 集只在最后评估一次：

| 实验 | 变量 | 骨干 | val F1-macro | val Acc | val AUC | 训练耗时 |
|------|------|------|--------------|---------|---------|----------|
| E0 | 基线 | ResNet18 | 0.9169 | 0.9221 | 0.9589 | 66 min |
| E1 | 类别加权损失 | ResNet18 | 0.8839 ↓ | 0.8897 | 0.9455 | 61 min |
| E2a | 骨干对比 | EfficientNet-B0 | 0.9041 | 0.9087 | 0.9604 | 69 min |
| **E2b** | **骨干对比** | **MobileNetV3-L** | **0.9183 ↑** | **0.9240** | — | **~36 min** |
| E3 | 强增强(RandAug+CutMix) | ResNet18 | 0.8957 ↓ | 0.9011 | 0.9622 | 100 min |

**结论**：E2b 精度最高 **且** 训练最快（约为其他实验的 1/2）。相关参数：
`batch_size 16`（7.8GB 内存约束）、`8 epochs`（早停 patience 4，best 为 ep6
checkpoint）、`lr 3e-4` + AdamW + cosine、ImageNet 预训练微调。
最终 test（+flip-TTA）：acc 0.916 / F1-macro 0.910 / AUC 0.969
（混淆矩阵 TN=309 FP=8 FN=36 TP=172）。

另一个隐性依据：**容量与数据规模匹配**——5406 张数据下，更重的骨干
（EfficientNet-B0 等）并未带来收益；MobileNetV3-Large 恰处精度-容量最佳平衡点。

---

## 2. 环境准备

```bash
# 1.（推荐）创建并激活虚拟环境
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. CPU 版 PyTorch（国内推荐清华镜像，官方源 SSL 不稳）
pip install torch torchvision --index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch/wheel/cpu/

# 3. 其余依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

也可用 conda（`environment.yml`）或 Docker（`Dockerfile`，CPU 镜像，一键
prepare→train→evaluate）。

## 3. CPU 运行方式及步骤

训练成果（checkpoints、数据划分）已随实验产出；**只想用模型做预测可跳过
第 1、2 步**。

```bash
# 阶段 1 — 数据准备：解压、校验、dHash 去重聚簇、划分、EDA
python scripts/prepare_data.py            # 或 --pos-zip/--neg-zip 指定压缩包位置

# 阶段 2 — 训练（CPU 上约 0.5~1.7 小时/实验）
python scripts/train.py --experiment mobilenet_v3

# 阶段 3 — 用 BEST checkpoint 在 test 集上评估（只做一次）
python scripts/evaluate.py --experiment mobilenet_v3 --split test --tta

# 阶段 4 — 对新图/目录推理
python scripts/predict.py --experiment mobilenet_v3 --input 图片.jpg --tta
python scripts/predict.py --experiment mobilenet_v3 --input 某目录/ --tta
```

可用实验名：`baseline_resnet18`(E0)、`e1_resnet18_full`(E1)、
`efficientnet_b0`(E2a)、`mobilenet_v3`(**E2b，最优**)、`e3_augment`(E3)。

**CPU 注意事项**：
- `batch_size` 请保持 ≤16（7.8GB 内存上限，batch 64 会 OOM）；
- 代码已内置 CPU 吞吐优化：JPEG draft 解码、`torch.set_num_threads`、
  persistent DataLoader，实测吞吐 4.3 → 10.7 img/s；
- 没有预先下载的 ImageNet 权重时首次运行会联网下载（离线可手动放入
  `~/.cache/torch/hub/checkpoints/`）。

## 4. GPU 运行方式及使用建议

**代码一行都不用改。** 设备选择集中在 `src/common.py`：

```python
return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

训练器/评估/预测的 device 全部由此函数自动决定，张量统一 `.to(device)`，
checkpoint 加载带 `map_location`——代码是设备无关的。只需换环境：

```bash
# 1. 卸载 CPU 版，安装 CUDA 版 PyTorch（cu121/cu124 按驱动选择）
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. 验证 GPU 可见
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 3. 照常运行即可，自动用 GPU
python scripts/train.py --experiment mobilenet_v3
```

**换 GPU 后建议调整两处配置**（非必须，把 CPU 时代的限制放开）：

| 位置 | 原值 | 建议值 | 原因 |
|------|------|--------|------|
| `configs/experiments/mobilenet_v3.yaml` → `data.batch_size` | 16 | 64–128 | 16 是内存逼出来的上限，显存撑得住，吞吐显著提升 |
| `configs/base.yaml` → `data.num_workers` | 4 | 8 | GPU 吃数据快，需更多加载进程喂饱 |

预期收益：CPU ~36 分钟的训练在主流单卡上约 **2–4 分钟**，也让你有预算尝试
更高分辨率（`data.image_size`）、更大骨干、更长训练等 CPU 上跑不起的方案。

## 5. 改动与调整指南

**第一优先：改配置、不改代码**（改完重跑 train 即生效）：

| 想调什么 | 改哪里 |
|----------|--------|
| 学习率 / epochs / 早停 | `configs/base.yaml` → `training:` |
| 输入分辨率 | `configs/base.yaml` → `data.image_size` |
| batch 大小 | 各实验 yaml → `data.batch_size` |
| 数据增强 | `configs/base.yaml` → `augment:`（RandAug / CutMix 开关） |
| 新增实验 | 复制一份 `configs/experiments/*.yaml` 改名 → `--experiment 新名字` |

**第二优先：改代码**（入口见下节文件职责表）：

- 加新骨干 → `src/models/builder.py` 的 `BACKBONES` 注册表（需可替换分类头）；
- 训练循环/早停逻辑 → `src/training/trainer.py`；
- 损失函数 → `src/training/losses.py`；
- 数据变换/增强 → `src/data/dataset.py`；
- 指标计算 → `src/eval/evaluate.py`；
- 预测输出格式 → `src/inference/predict.py`。

**改动后提交**（本地 main 已关联远端、代理与凭据已配好）：

```bash
git add -A && git commit -m "feat: 修改说明" && git push
```

**方法论提醒**：改动若影响数据划分或训练，请只用 **val 集**调参；test 集
只做最终一次性评估，避免污染测试估计（本项目实验报告遵循此原则）。

## 6. 目录结构与各文件职责

```
saint_george_classifier/
├── README.md                    # 本文件
├── requirements.txt             # pip 依赖（锁定版本）
├── environment.yml              # conda 环境
├── Dockerfile                   # CPU 一键复现镜像
├── PUSH_GUIDE.md                # GitHub 推送指引与故障排查
├── configs/
│   ├── base.yaml                # 全部默认参数（训练/数据/增强/评估）
│   └── experiments/             # 5 组实验配置，各自覆盖 base 的子集
├── src/                         # 核心库
│   ├── common.py                # 路径、随机种子、设备选择、配置深合并
│   ├── data/
│   │   ├── prepare.py           # 解压、损坏检测、dHash 聚簇去重、导出
│   │   ├── split.py             # 按聚簇分组的防泄漏划分（80/10/10）
│   │   └── dataset.py           # Dataset 定义与训练/评估 transforms
│   ├── models/builder.py        # 骨干工厂（BACKBONES 注册表，替换分类头）
│   ├── training/
│   │   ├── losses.py            # 类别权重、label smoothing、CutMix/MixUp
│   │   └── trainer.py           # 训练循环、早停、checkpoint、JSON/CSV 日志
│   ├── eval/evaluate.py         # 指标、混淆矩阵、误分类导出、flip-TTA
│   └── inference/predict.py     # 单图/多图/目录预测，含 TTA
├── scripts/                     # CLI 入口（各阶段一条命令）
│   ├── prepare_data.py          # 阶段 1：数据准备
│   ├── train.py                 # 阶段 2：训练
│   ├── evaluate.py              # 阶段 3：测试集评估
│   └── predict.py               # 阶段 4：推理
├── data/                        # raw（原图，不入库）/ splits（划分 CSV）
├── experiments/                 # logs / checkpoints / metrics / misclassified
└── reports/                     # EDA.md、EXPERIMENTS.md、REPORT.md
```

## 7. 网络架构与训练流水线

```
Image ─▶ transforms (resize / RandomResizedCrop / flip / jitter / normalize)
      └▶ Backbone (ImageNet 权重) ─▶ 2-class head
                                        └▶ CrossEntropyLoss (+class weights / label smoothing)
                                            └▶ AdamW + CosineAnnealing / EarlyStopping
```

训练增强中的 `RandomResizedCrop`（较紧的 scale 范围）是刻意选择：圣乔治在
画作、雕像、徽章、彩窗中出现的尺度跨度极大，紧裁剪增强可提升尺度鲁棒性。

## 8. 防泄漏策略

1. **感知哈希**（dHash, 64-bit）计算每张图；
2. **并查集聚簇**近重复（Hamming 距离 ≤ 阈值）；
3. **跨类簇直接剔除**——同一视觉内容出现在两个标签下视为污染；
4. **分组划分**：整个聚簇只进一个 split（train/val/test = 80/10/10，按类分层）。

原始 5700 张经此流程后为 5406 张（剔除 294 张跨类近重复），保证泛化估计诚实。

## 9. 复现性与 FAQ

- **确定性**：每个脚本在任何工作前调用 `set_seed(cfg.seed)`；
- **配置驱动**：改 `configs/*.yaml` 而不是改代码；
- **全流程重跑**：`prepare_data.py` → `train.py` → `evaluate.py`；
- *"为什么训练慢？"* —— 参考机无 GPU。换 GPU 主机或调小
  `data.image_size` / `training.epochs`；
- *"能加骨干吗？"* —— 注册到 `src/models/builder.py` 的 `BACKBONES`
  （需暴露可替换分类头）；
- *checkpoint/原始数据在哪？* —— 大文件不入库：`data/raw/`（550MB 原图）与
  `experiments/checkpoints/*.pth`（147MB 权重）被 .gitignore 排除，按 README
  流程重新生成即可复现。

## 10. 交付物清单

- [x] Python 代码（数据准备 / 训练 / 推理 / 评估）
- [x] README（选型依据 / 安装 / CPU·GPU 运行 / 架构 / 文件职责）
- [x] 结果 / 日志 / checkpoints（`experiments/`）
- [x] 实验报告（`reports/EXPERIMENTS.md`：假设→变量→结果→结论）
- [x] 方法论证（每实验配置内注释）
- [x] 最终报告（`reports/REPORT.md`：指标、混淆矩阵、误分类分析、改进方向）
- [x] 环境文件（`requirements.txt`、`environment.yml`、`Dockerfile`）

---

感谢提供的机会
