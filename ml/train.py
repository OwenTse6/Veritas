"""
train.py

Main training script for the AI-image classifier.

Usage:
    python train.py --data_root /path/to/dataset --epochs 10 --batch_size 32
    python train.py --data_root /path/to/dataset --epochs 10 --resume

Expects the folder structure documented in dataset.py. Saves the best
checkpoint (by validation accuracy) to checkpoints/best_model.pt, and a
per-epoch checkpoint (model + optimizer + progress) to
checkpoints/latest.pt so an interrupted run (crash, power outage,
accidentally shutting down mid-training) only costs you the current
epoch's progress, not the whole run -- pass --resume to pick back up
from latest.pt.

IMPORTANT: use the SAME --unfreeze setting when resuming as the original
run. Freezing/unfreezing changes which parameters the optimizer tracks;
resuming with a different setting than the interrupted run will likely
error or silently behave oddly, since the saved optimizer state won't
match the newly-built optimizer's parameter groups.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import build_dataloaders
from model import build_model, get_trainable_params


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Path to dataset root (see dataset.py for structure)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze_backbone", action="store_true", default=True,
                         help="Only train the classifier head (fast, good starting point)")
    parser.add_argument("--unfreeze", action="store_true",
                         help="Fine-tune the whole backbone instead of just the head "
                              "(slower, needs more data, better ceiling -- use once "
                              "the frozen-backbone baseline is working)")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--resume", action="store_true",
                         help="Resume from checkpoints/latest.pt (model, optimizer state, "
                              "and epoch count) instead of starting from scratch. Use the "
                              "SAME --unfreeze setting as the run you're resuming.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("WARNING: no GPU detected. Training will be very slow. "
              "If you have a GPU, check your CUDA/PyTorch installation.")

    train_loader, val_loader, classes = build_dataloaders(
        args.data_root, batch_size=args.batch_size
    )
    print(f"Classes: {classes} (index 0 = {classes[0]}, index 1 = {classes[1]})")
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    model = build_model(pretrained=True, freeze_backbone=not args.unfreeze).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(get_trainable_params(model), lr=args.lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    best_val_acc = 0.0
    start_epoch = 1

    latest_path = checkpoint_dir / "latest.pt"
    if args.resume:
        if not latest_path.exists():
            print(f"ERROR: --resume was given but no checkpoint found at {latest_path}. "
                  "Starting from scratch instead.")
        else:
            print(f"Resuming from {latest_path} ...")
            checkpoint = torch.load(latest_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            best_val_acc = checkpoint["best_val_acc"]
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resumed at epoch {start_epoch}, best_val_acc so far: {best_val_acc:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) -- "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} -- "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "val_acc": val_acc,
                "epoch": epoch,
            }, checkpoint_dir / "best_model.pt")
            print(f"  -> saved new best checkpoint (val_acc={val_acc:.4f})")

        # Saved every epoch regardless of improvement -- this is what --resume
        # picks up from, so an interrupted run only loses the current epoch.
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "classes": classes,
            "best_val_acc": best_val_acc,
            "epoch": epoch,
        }, latest_path)

    print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f}")
    print(f"Checkpoint saved to: {checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
