"""Datasets and image transforms.

The training augmentations are designed around the task specifics:
* ``RandomResizedCrop`` with a tight scale range helps the model focus on the
  Saint George figure regardless of how large/small it appears in the frame.
* ``RandAugment`` is available as a single-flag toggle for the augmentation
  strengthening experiment (E3).
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(cfg: dict, mode: str = "train"):
    size = cfg["data"]["image_size"]
    ra = cfg.get("augment", {}).get("randaugment", False) and mode == "train"

    if mode == "train":
        tf = [
            transforms.Resize(256),
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        ]
        if ra:
            tf.append(transforms.RandAugment(num_ops=2, magnitude=9))
        tf += [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    else:  # val / test
        tf = [
            transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    return transforms.Compose(tf)


class ClassificationDataset(Dataset):
    """Loads (image, label) from a list of record dicts.

    JPEG files are decoded with PIL's ``draft`` mode: the DCT coefficients are
    read at a reduced scale (roughly 512px on the long side) instead of full
    resolution (median 638px, max ~4000px) before the transform pipeline
    resizes to 224px anyway. This is lossless w.r.t. the final 224px tensor
    quality and cuts CPU decode time dramatically.
    """

    DRAFT_SIZE = 512  # long-side target fed into PIL draft decoding

    def __init__(self, records: list[dict], transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path)
        if img.format == "JPEG":
            img.draft("RGB", (self.DRAFT_SIZE, self.DRAFT_SIZE))
        return img.convert("RGB")

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = self._load_image(rec["path"])
        if self.transform:
            img = self.transform(img)
        return img, int(rec["label"])

    @staticmethod
    def from_csv(path: str | Path, transform=None):
        from .split import load_split
        return ClassificationDataset(load_split(path), transform)
