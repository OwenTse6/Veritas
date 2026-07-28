"""
rebalance_split.py

One-off fix for datasets created by the OLD (buggy) version of
prepare_community_forensics.py, which used a single shared counter to
decide train/val split -- if the underlying data wasn't perfectly
interleaved by class, one class's validation folder could end up
completely empty (exactly what happened: val/real had 0 images).

This script moves existing files between train/ and val/ to restore a
proper per-class split, WITHOUT re-downloading anything.

Usage:
    python rebalance_split.py --dest data --val_fraction 0.1
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="data")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    dest = Path(args.dest)
    random.seed(args.seed)

    for cls in ("ai", "real"):
        train_dir = dest / "train" / cls
        val_dir = dest / "val" / cls
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        train_files = list(train_dir.iterdir())
        val_files = list(val_dir.iterdir())

        total = len(train_files) + len(val_files)
        target_val = int(total * args.val_fraction)

        print(f"\nClass '{cls}': total={total}, current train={len(train_files)}, "
              f"current val={len(val_files)}, target val={target_val}")

        if len(val_files) < target_val:
            # Move some train -> val
            n_to_move = target_val - len(val_files)
            n_to_move = min(n_to_move, len(train_files))
            to_move = random.sample(train_files, n_to_move)
            for f in to_move:
                shutil.move(str(f), str(val_dir / f.name))
            print(f"  Moved {n_to_move} files: train -> val")

        elif len(val_files) > target_val:
            # Move some val -> train
            n_to_move = len(val_files) - target_val
            n_to_move = min(n_to_move, len(val_files))
            to_move = random.sample(val_files, n_to_move)
            for f in to_move:
                shutil.move(str(f), str(train_dir / f.name))
            print(f"  Moved {n_to_move} files: val -> train")

        else:
            print("  Already balanced, no changes needed.")

    print("\nDone. New counts:")
    for split in ("train", "val"):
        for cls in ("ai", "real"):
            n = len(list((dest / split / cls).iterdir()))
            print(f"  {split}/{cls}: {n}")


if __name__ == "__main__":
    main()
