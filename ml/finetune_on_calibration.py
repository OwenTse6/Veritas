"""
finetune_on_calibration.py

Short domain-adaptation fine-tuning pass on top of an already-trained
checkpoint, using the small, hand-picked calibration_set/images/ folder
(the Gunpla photos, Gundam/Gogeta art, AI illustrations, etc. -- the
exact hard cases that exposed the gap between Community Forensics
validation accuracy and real-world performance).

IMPORTANT HONEST CAVEAT: this calibration set is TINY (a few dozen
images). This script holds out a small fraction as a genuinely unseen
test, but with this few images, that held-out accuracy is a noisy
estimate, not a strong statistical result -- treat directional
improvement as a hint worth investigating further with more real data,
not a doubled-down guarantee.

Modeling decision: human_art images (hand-drawn/rendered art -- not
photos, not AI-generated) are labeled "real" (0) by default, since the
actual task here is "was this AI-generated" and hand-made art genuinely
wasn't. Use --exclude_human_art if you'd rather leave them out entirely.

Usage:
    python finetune_on_calibration.py --checkpoint checkpoints/best_model.pt \
        --calibration_dir ../calibration_set/images --epochs 5
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from dataset import get_train_transforms, get_eval_transforms
from model import build_model

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class CalibrationDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def collect_samples(calibration_dir: Path, include_human_art: bool) -> list[tuple[Path, int]]:
    samples = []

    real_dir = calibration_dir / "real"
    for f in real_dir.iterdir():
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            samples.append((f, 0))  # 0 = real

    ai_dir = calibration_dir / "ai"
    for f in ai_dir.iterdir():
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            samples.append((f, 1))  # 1 = ai

    if include_human_art:
        human_art_dir = calibration_dir / "human_art"
        if human_art_dir.exists():
            for f in human_art_dir.iterdir():
                if f.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((f, 0))  # not AI-generated -> "real" for this task

    return samples


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else 0.0


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to the CF-trained checkpoint to fine-tune")
    parser.add_argument("--calibration_dir", default="../calibration_set/images")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5,
                         help="Deliberately very low -- this is a gentle nudge on a tiny "
                              "dataset, not a full retrain. A higher LR risks destroying "
                              "what the model already learned from Community Forensics.")
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--exclude_human_art", action="store_true",
                         help="Leave out human_art images entirely instead of treating them as 'real'")
    parser.add_argument("--output", default="checkpoints/finetuned_model.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    calibration_dir = Path(args.calibration_dir)
    samples = collect_samples(calibration_dir, include_human_art=not args.exclude_human_art)
    print(f"Collected {len(samples)} calibration samples "
          f"({sum(1 for _, l in samples if l == 0)} real/non-AI, "
          f"{sum(1 for _, l in samples if l == 1)} AI)")

    if len(samples) < 10:
        print("WARNING: very few samples found -- fine-tuning results will be extremely noisy. "
              "Double-check --calibration_dir is correct.")

    random.seed(args.seed)
    random.shuffle(samples)
    n_holdout = max(1, int(len(samples) * args.holdout_fraction))
    holdout_samples = samples[:n_holdout]
    finetune_samples = samples[n_holdout:]
    print(f"Fine-tuning on {len(finetune_samples)} samples, holding out {len(holdout_samples)} for evaluation.")

    finetune_dataset = CalibrationDataset(finetune_samples, get_train_transforms())
    holdout_dataset = CalibrationDataset(holdout_samples, get_eval_transforms())

    finetune_loader = DataLoader(finetune_dataset, batch_size=4, shuffle=True, num_workers=0)
    holdout_loader = DataLoader(holdout_dataset, batch_size=4, shuffle=False, num_workers=0)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print("\nEvaluating BEFORE fine-tuning...")
    before_acc = evaluate(model, holdout_loader, device)
    print(f"Held-out accuracy before fine-tuning: {before_acc:.4f} ({n_holdout} samples)")

    for param in model.parameters():
        param.requires_grad = True  # full fine-tune, but at a very low LR

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("\nFine-tuning...")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, finetune_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} -- loss={loss:.4f}")

    print("\nEvaluating AFTER fine-tuning...")
    after_acc = evaluate(model, holdout_loader, device)
    print(f"Held-out accuracy after fine-tuning:  {after_acc:.4f} ({n_holdout} samples)")

    print(f"\nChange: {after_acc - before_acc:+.4f}")
    print(f"(Remember: only {n_holdout} held-out samples -- treat this as a directional "
          f"hint, not a statistically strong result.)")

    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": checkpoint["classes"],
        "before_holdout_acc": before_acc,
        "after_holdout_acc": after_acc,
    }, output_path)
    print(f"\nFine-tuned checkpoint saved to: {output_path}")


if __name__ == "__main__":
    main()
