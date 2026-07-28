"""
dataset.py

Dataset loading for the AI-image classifier.

Expects a folder structure compatible with torchvision's ImageFolder:

    data_root/
        train/
            ai/
                img001.jpg
                ...
            real/
                img001.jpg
                ...
        val/
            ai/
            real/

NOTE ON CLASS ORDER: ImageFolder assigns labels alphabetically by folder
name, so "ai" = class 0, "real" = class 1. This is handled consistently
throughout this pipeline (see LABEL_NAMES below) -- don't rename the
folders without updating that too.

AUGMENTATION DESIGN NOTE -- this is deliberate, not boilerplate:
Our heuristic-based detector (the rest of this project) found that
real-world images -- resized, recompressed, and re-hosted across
platforms -- lose the fine pixel-level signals that make AI detection
work, and this broke every physics-based heuristic we tried. Rather than
let a trained model inherit that same blind spot, we deliberately
augment training data with realistic JPEG recompression and resizing,
so the model learns to detect AI generation using features that SURVIVE
typical web processing, not just pristine first-generation images.
"""

from __future__ import annotations

import io
import random

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

LABEL_NAMES = ["ai", "real"]  # index 0 = ai, index 1 = real (alphabetical, per ImageFolder)

IMAGE_SIZE = 224


class RandomJPEGCompression:
    """Randomly re-compresses the image at a random JPEG quality, simulating
    the kind of reprocessing real-world images go through (re-hosting,
    re-sharing, platform re-encoding). Applied with some probability so the
    model sees both pristine and reprocessed examples during training."""

    def __init__(self, probability: float = 0.5, quality_range: tuple[int, int] = (40, 95)):
        self.probability = probability
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return img
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


def get_train_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        RandomJPEGCompression(probability=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
    ])


def get_eval_transforms() -> transforms.Compose:
    """No augmentation for validation/test -- we want to measure real
    performance, not performance-under-augmentation."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_dataloaders(data_root: str, batch_size: int = 32, num_workers: int = 4):
    train_dataset = ImageFolder(f"{data_root}/train", transform=get_train_transforms())
    val_dataset = ImageFolder(f"{data_root}/val", transform=get_eval_transforms())

    # Sanity check: make sure folder naming produced the label order we expect.
    assert train_dataset.classes == LABEL_NAMES, (
        f"Expected class folders {LABEL_NAMES}, got {train_dataset.classes}. "
        "Check your train/ subfolder names."
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, train_dataset.classes
