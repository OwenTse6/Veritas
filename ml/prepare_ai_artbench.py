"""
prepare_ai_artbench.py

Reorganizes a downloaded/extracted AI-ArtBench dataset into the folder
structure train.py expects:

    data/train/ai/    data/train/real/
    data/val/ai/       data/val/real/

Download first:
    pip install kaggle
    kaggle datasets download -d ravidussilva/real-ai-art
    unzip real-ai-art.zip -d ai_artbench_raw

AI-ArtBench ships as per-STYLE folders (e.g. "Post_Impressionism" for
human-drawn art, and something like "Post_Impressionism_AI_LD" /
"..._AI_SD" for the AI-generated equivalents from Latent/Standard
Diffusion) under train/ and test/ splits.

IMPORTANT UNCERTAINTY: the exact folder naming wasn't independently
verified against the live dataset (no network access to Kaggle from
where this was written) -- this script classifies folders by keyword
matching (looking for "ai", "_ld", "_sd", "diffusion" in the folder
name) rather than assuming an exact name, and PRINTS every folder it
finds along with its classification decision before copying anything,
so you can catch a misclassification before trusting the result.

Usage:
    python prepare_ai_artbench.py --source ai_artbench_raw --dest data
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

AI_KEYWORDS = ["ai", "_ld", "_sd", "diffusion", "latent", "generated", "fake"]


def classify_folder(folder_name: str) -> str:
    """Returns 'ai' or 'real' based on keyword matching in the folder name."""
    name_lower = folder_name.lower()
    for kw in AI_KEYWORDS:
        if kw in name_lower:
            return "ai"
    return "real"


def find_split_root(source: Path, split_name: str) -> Path | None:
    for p in source.rglob("*"):
        if p.is_dir() and p.name.lower() == split_name:
            return p
    return None


def copy_images(src_dir: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in src_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            # Prefix with parent folder name to avoid filename collisions
            # across multiple style folders being merged into one class.
            out_name = f"{src_dir.name}_{f.name}"
            shutil.copy2(f, dst_dir / out_name)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to extracted AI-ArtBench folder")
    parser.add_argument("--dest", default="data")
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)

    if not source.exists():
        print(f"ERROR: source folder '{source}' does not exist.")
        return

    for split_name, out_split in [("train", "train"), ("test", "val")]:
        split_root = find_split_root(source, split_name)
        if split_root is None:
            print(f"WARNING: could not find a '{split_name}' folder under {source} -- skipping.")
            continue

        subfolders = [p for p in split_root.iterdir() if p.is_dir()]
        print(f"\n=== {split_name}/ -> found {len(subfolders)} style/source folders ===")

        total_ai, total_real = 0, 0
        for folder in subfolders:
            cls = classify_folder(folder.name)
            n = copy_images(folder, dest / out_split / cls)
            print(f"  {folder.name:40s} -> classified as '{cls}'  ({n} images copied)")
            if cls == "ai":
                total_ai += n
            else:
                total_real += n

        print(f"  Subtotal for {out_split}/: ai={total_ai}, real={total_real}")

    print(f"\nDone. Dataset ready at: {dest}/")
    print("IMPORTANT: double-check the classification decisions printed above --")
    print("if any folder was misclassified, move its images manually between")
    print(f"{dest}/train/ai and {dest}/train/real (or the val/ equivalents) before training.")
    print(f"\nRun: python train.py --data_root {dest} --epochs 10")


if __name__ == "__main__":
    main()
