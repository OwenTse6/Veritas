"""
cli.py

Command-line interface for slopcheck.

Usage:
    python cli.py text myfile.txt
    python cli.py text myfile.txt --explain
    python cli.py image myphoto.png --explain
    python cli.py batch some_folder/
    python cli.py batch some_folder/ --sort score
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np

from core import score_text, score_image

TEXT_EXTENSIONS = {".txt", ".md"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


@click.group()
def main():
    """slopcheck -- heuristic, explainable AI-content detector."""
    pass


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--explain", is_flag=True, help="Print human-readable explanation.")
@click.option("--json", "as_json", is_flag=True, help="Print raw JSON output.")
def text(path, explain, as_json):
    """Analyze a text file for AI-generated content signals."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    result = score_text(content)
    _print_result(result, explain, as_json)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--explain", is_flag=True, help="Print human-readable explanation.")
@click.option("--json", "as_json", is_flag=True, help="Print raw JSON output.")
def image(path, explain, as_json):
    """Analyze an image file for AI-generated content signals."""
    result = score_image(path)
    _print_result(result, explain, as_json)


@main.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--sort", "sort_by", type=click.Choice(["score", "name"]), default="score",
    help="Sort results by score (highest suspicion first) or filename.",
)
@click.option("--json", "as_json", is_flag=True, help="Print raw JSON output instead of a table.")
def batch(folder, sort_by, as_json):
    """Scan every text/image file in a folder and print one summary table.

    Auto-detects file type by extension -- no need to sort files into
    separate folders first, and no need to run one file at a time.
    """
    folder_path = Path(folder)
    files = sorted(p for p in folder_path.iterdir() if p.is_file())

    rows = []
    skipped = []

    for f in files:
        ext = f.suffix.lower()
        try:
            if ext in TEXT_EXTENSIONS:
                content = f.read_text(encoding="utf-8")
                result = score_text(content)
                rows.append((f.name, "text", result["overall_score"], result["verdict"]))
            elif ext in IMAGE_EXTENSIONS:
                result = score_image(str(f))
                rows.append((f.name, "image", result["overall_score"], result["verdict"]))
            else:
                skipped.append(f.name)
        except Exception as e:
            rows.append((f.name, "ERROR", 0.0, str(e)))

    if sort_by == "score":
        rows.sort(key=lambda r: -r[2])
    else:
        rows.sort(key=lambda r: r[0].lower())

    if as_json:
        payload = [
            {"file": name, "type": kind, "score": score, "verdict": verdict}
            for name, kind, score, verdict in rows
        ]
        click.echo(json.dumps(payload, indent=2, default=_json_default))
        return

    click.echo(f"\n{'SCORE':>7}  {'TYPE':6}  {'VERDICT':28}  FILE")
    click.echo("-" * 80)
    for name, kind, score, verdict in rows:
        click.echo(f"{score:7.1f}  {kind:6}  {verdict:28}  {name}")

    click.echo(f"\n{len(rows)} file(s) scanned.")
    if skipped:
        click.echo(f"{len(skipped)} file(s) skipped (unrecognized extension): {', '.join(skipped)}")
    click.echo()


def _print_result(result: dict, explain: bool, as_json: bool):
    if as_json:
        click.echo(json.dumps(result, indent=2, default=_json_default))
        return

    click.echo(f"\nOverall score: {result['overall_score']} / 100")
    click.echo(f"Verdict: {result['verdict']}\n")

    click.echo("Signal breakdown:")
    for name, data in result["signals"].items():
        click.echo(f"  {name}: {data['score']}")

    if explain:
        click.echo("\nExplanation:")
        for line in result["explanation"]:
            click.echo(f"  - {line}")

    click.echo()


if __name__ == "__main__":
    main()
