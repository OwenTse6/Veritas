"""
prepare_community_forensics.py

Downloads Community Forensics-Small (OwensLab/CommunityForensics-Small on
Hugging Face) ONE SHARD FILE AT A TIME, fully to disk, and processes each
shard in small in-memory batches -- rather than using the `datasets`
library's default streaming mode, which prefetches several large parquet
shards CONCURRENTLY and can exhaust RAM (some shards are 4GB+; several
being buffered in memory at once can easily blow past available memory
on a typical machine, which is exactly what happened during testing).

This approach is slower but far more memory-safe and resumable:
- Downloads exactly one shard at a time (via huggingface_hub's own
  resumable download, which writes straight to disk).
- Reads that shard in small row-group batches via pyarrow, never loading
  the whole file into memory.
- Deletes the shard after processing (unless --keep_cache) to save disk
  space, since shards can be several GB each.
- Tracks fully-processed shards in a manifest file, so re-running after
  a crash/interruption skips shards you've already finished rather than
  re-downloading them.

LICENSE NOTE: Community Forensics-Small is cc-by-nc-sa-4.0 --
non-commercial use only.

Usage:
    python prepare_community_forensics.py --n_samples 15000 --dest data
    # Re-running later adds MORE samples on top of existing ones, and
    # skips shards already fully processed:
    python prepare_community_forensics.py --n_samples 35000 --dest data
"""

from __future__ import annotations

import argparse
import io
import os
import random
from pathlib import Path

from PIL import Image

REPO_ID = "OwensLab/CommunityForensics-Small"
BATCH_SIZE = 200  # rows read into memory at once per shard -- small and safe


def count_existing(dest: Path) -> tuple[int, int]:
    total = 0
    max_index = -1
    for split in ("train", "val"):
        for cls in ("ai", "real"):
            folder = dest / split / cls
            if not folder.exists():
                continue
            for f in folder.glob("img_*.jpg"):
                total += 1
                try:
                    idx = int(f.stem.split("_")[1])
                    max_index = max(max_index, idx)
                except (IndexError, ValueError):
                    pass
    return total, max_index + 1


def load_manifest(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    return set(manifest_path.read_text().splitlines())


def append_manifest(manifest_path: Path, shard_name: str):
    with open(manifest_path, "a") as f:
        f.write(shard_name + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=15000,
                         help="Number of NEW images to add on top of what's already in --dest")
    parser.add_argument("--dest", default="data")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--keep_cache", action="store_true",
                         help="Keep downloaded shard files instead of deleting them after "
                              "processing (uses much more disk space -- shards can be "
                              "several GB each)")
    args = parser.parse_args()

    try:
        from huggingface_hub import list_repo_files, hf_hub_download
        import pyarrow.parquet as pq
    except ImportError:
        print("ERROR: missing dependencies. Run:")
        print("  pip install huggingface_hub pyarrow")
        return

    dest = Path(args.dest)
    for split in ("train", "val"):
        for cls in ("ai", "real"):
            (dest / split / cls).mkdir(parents=True, exist_ok=True)

    manifest_path = dest / ".cf_processed_shards.txt"
    processed_shards = load_manifest(manifest_path)

    print("Listing available shard files (no download yet)...")
    all_files = list_repo_files(REPO_ID, repo_type="dataset")
    shard_files = sorted(f for f in all_files if f.startswith("data/") and f.endswith(".parquet"))
    print(f"Found {len(shard_files)} total shard files in the dataset.")

    remaining_shards = [f for f in shard_files if f not in processed_shards]
    print(f"{len(processed_shards)} shards already fully processed (from a previous run). "
          f"{len(remaining_shards)} remaining to choose from.")

    random.seed(args.seed)
    random.shuffle(remaining_shards)

    existing_total, next_index = count_existing(dest)
    print(f"Existing images in {dest}/: {existing_total}. Adding up to {args.n_samples} more.\n")

    n_val_target_per_class = int(args.n_samples * args.val_fraction / 2)
    counts = {"train": {"ai": 0, "real": 0}, "val": {"ai": 0, "real": 0}}
    total_saved = 0
    current_index = next_index

    for shard_name in remaining_shards:
        if total_saved >= args.n_samples:
            break

        print(f"Downloading shard: {shard_name} ...")
        try:
            local_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=shard_name)
        except Exception as e:
            print(f"  Failed to download {shard_name}, skipping: {e}")
            continue

        try:
            parquet_file = pq.ParquetFile(local_path)
            shard_fully_processed = True

            for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=["image_data", "label"]):
                if total_saved >= args.n_samples:
                    shard_fully_processed = False  # stopped partway -- don't mark as done
                    break

                image_data_col = batch.column("image_data").to_pylist()
                label_col = batch.column("label").to_pylist()

                for image_bytes, label in zip(image_data_col, label_col):
                    if total_saved >= args.n_samples:
                        shard_fully_processed = False
                        break

                    cls = "ai" if label == 1 else "real"
                    # Per-class split -- this is the fix. The old version used one
                    # shared counter for ALL classes, so if a run of same-label
                    # examples came through first (shards aren't perfectly
                    # interleaved), the val quota could fill up entirely with one
                    # class, leaving the other with zero validation examples --
                    # exactly what happened (val/real ended up empty).
                    split = "val" if counts["val"][cls] < n_val_target_per_class else "train"

                    try:
                        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    except Exception:
                        continue  # skip unreadable image, doesn't count toward total

                    out_path = dest / split / cls / f"img_{current_index:06d}.jpg"
                    img.save(out_path, format="JPEG", quality=95)

                    counts[split][cls] += 1
                    total_saved += 1
                    current_index += 1

                    if total_saved % 500 == 0:
                        print(f"  Saved {total_saved}/{args.n_samples}...")

            if shard_fully_processed:
                append_manifest(manifest_path, shard_name)

        finally:
            if not args.keep_cache:
                try:
                    os.remove(local_path)
                except OSError:
                    pass

    print("\nDone.")
    print(f"Added this run -- train/ai: {counts['train']['ai']}, train/real: {counts['train']['real']}, "
          f"val/ai: {counts['val']['ai']}, val/real: {counts['val']['real']}")

    new_total, _ = count_existing(dest)
    print(f"Total images now in {dest}/: {new_total}")
    print(f"\nRun: python train.py --data_root {dest} --epochs 10")


if __name__ == "__main__":
    main()
