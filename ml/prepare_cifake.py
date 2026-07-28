"""
prepare_cifake.py

Reorganizes a downloaded/extracted CIFAKE dataset into the folder
structure train.py expects:

    data/train/ai/    data/train/real/
    data/val/ai/       data/val/real/

CIFAKE ships as train/REAL, train/FAKE, test/REAL, test/FAKE -- different
names and a "test" split instead of "val". This script copies (not
moves, so your original download stays intact) files into the right
place, matching folder names case-insensitively in case the exact
casing varies by download source.

Usage:
    python prepare_cifake.py --source cifake_raw --dest data
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def find_folder(root: Path, *name_candidates: str) -> Path | None:
    """Case-insensitive search for a folder matching any of the given names,
    anywhere under root (CIFAKE downloads sometimes nest an extra level)."""
    candidates_lower = {n.lower() for n in name_candidates}
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() in candidates_lower:
            return p
    return None


def copy_images(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in src_dir.iterdir():
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            shutil.copy2(f, dst_dir / f.name)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to extracted CIFAKE folder")
    parser.add_argument("--dest", default="data", help="Output folder (default: ./data)")
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)

    if not source.exists():
        print(f"ERROR: source folder '{source}' does not exist.")
        return

    # CIFAKE's train split
    train_root = find_folder(source, "train")
    if train_root is None:
        print("ERROR: could not find a 'train' folder under the source directory. "
              "Check that you extracted the CIFAKE zip correctly, and pass the "
              "correct --source path (the folder that directly contains 'train'/'test').")
        return

    train_real = find_folder(train_root, "real")
    train_fake = find_folder(train_root, "fake")

    # CIFAKE's test split -- we use this as our val split
    test_root = find_folder(source, "test")
    test_real = find_folder(test_root, "real") if test_root else None
    test_fake = find_folder(test_root, "fake") if test_root else None

    missing = [
        name for name, path in [
            ("train/REAL", train_real), ("train/FAKE", train_fake),
            ("test/REAL", test_real), ("test/FAKE", test_fake),
        ] if path is None
    ]
    if missing:
        print(f"ERROR: could not find these expected folders: {missing}")
        print("Check the extracted folder structure manually (e.g. `find cifake_raw -type d`) "
              "and adjust --source to point at the right level.")
        return

    print(f"Found train/REAL:  {train_real}")
    print(f"Found train/FAKE:  {train_fake}")
    print(f"Found test/REAL:   {test_real}")
    print(f"Found test/FAKE:   {test_fake}")
    print()

    n = copy_images(train_real, dest / "train" / "real")
    print(f"Copied {n} images -> {dest}/train/real")

    n = copy_images(train_fake, dest / "train" / "ai")
    print(f"Copied {n} images -> {dest}/train/ai")

    n = copy_images(test_real, dest / "val" / "real")
    print(f"Copied {n} images -> {dest}/val/real")

    n = copy_images(test_fake, dest / "val" / "ai")
    print(f"Copied {n} images -> {dest}/val/ai")

    print(f"\nDone. Dataset ready at: {dest}/")
    print(f"Run: python train.py --data_root {dest} --epochs 10")


if __name__ == "__main__":
    main()
