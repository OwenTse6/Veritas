"""
infer_batch.py

Run the trained classifier on every image in a folder at once, and print
a sorted summary table -- mirrors the style of the main project's
`python cli.py batch` command.

Usage:
    python infer_batch.py --checkpoint checkpoints/best_model.pt --folder some_folder/
    python infer_batch.py --checkpoint checkpoints/best_model.pt --folder some_folder/ --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from infer import load_model, predict

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of a table")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(args.checkpoint, device)

    folder = Path(args.folder)
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)

    rows = []
    for f in files:
        try:
            result = predict(model, classes, str(f), device)
            rows.append((f.name, result["ai_probability"], result["verdict"]))
        except Exception as e:
            rows.append((f.name, 0.0, f"ERROR: {e}"))

    rows.sort(key=lambda r: -r[1])  # highest AI-probability first

    if args.json:
        payload = [
            {"file": name, "ai_probability": prob, "verdict": verdict}
            for name, prob, verdict in rows
        ]
        print(json.dumps(payload, indent=2))
        return

    print(f"\n{'AI %':>7}  {'VERDICT':20}  FILE")
    print("-" * 70)
    for name, prob, verdict in rows:
        print(f"{prob:7.1f}  {verdict:20}  {name}")
    print(f"\n{len(rows)} file(s) scanned.\n")


if __name__ == "__main__":
    main()
