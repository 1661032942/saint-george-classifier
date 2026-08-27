# Data Exploration (EDA)

- Total images scanned: **5700**
- Valid images kept: **5700**
  - positive (Saint George): 2360 (41.4%)
  - negative (no Saint George): 3340 (58.6%)
- Corrupt / unreadable: **0** (excluded, see corrupt.txt)
- Near-duplicate clusters: **506**
- Cross-class contaminated clusters (dropped): **51** (294 images removed)
- Image width/height median: **638 px** (min 110, max 3981)

## Split sizes (leakage-safe, cluster-grouped, stratified)
- train: 4355
- val:   526
- test:  525

## Notes
- Saint George appears in many forms (paintings, sculptures, badges, stained glass); negatives may contain other saints, knights, dragons or churches — a hard semantic distinction.
- Cross-class near-duplicates are dropped because they ambiguously label the same visual content as both classes.
- Splits group near-duplicate clusters together to prevent leakage.