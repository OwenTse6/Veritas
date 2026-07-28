"""
calibrate.py

Batch-test slopcheck against a labeled folder of samples and report how
well the current signals/weights separate the two groups. This is the
tool for calibration work: drop files into calibration_set/text/human,
calibration_set/text/ai, calibration_set/images/real, calibration_set/images/ai,
then run this script.

Usage:
    python calibrate.py text
    python calibrate.py image
    python calibrate.py both
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from core import score_text, score_image

BASE_DIR = Path(__file__).parent / "calibration_set"

TEXT_EXTENSIONS = {".txt", ".md"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _list_files(folder: Path, extensions: set[str]) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


def _summarize(scores: list[float]) -> dict:
    if not scores:
        return {"count": 0}
    n = len(scores)
    mean = sum(scores) / n
    sorted_scores = sorted(scores)
    median = sorted_scores[n // 2] if n % 2 else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
    variance = sum((s - mean) ** 2 for s in scores) / n
    return {
        "count": n,
        "mean": round(mean, 1),
        "median": round(median, 1),
        "min": round(min(scores), 1),
        "max": round(max(scores), 1),
        "stdev": round(variance ** 0.5, 1),
    }


def _print_group_table(label: str, results: list[tuple[str, float, str]]):
    click.echo(f"\n{label}:")
    for name, score, verdict in results:
        click.echo(f"  {score:5.1f}  {verdict:28s}  {name}")


def _print_summary(label: str, stats: dict):
    if stats["count"] == 0:
        click.echo(f"{label}: no files found")
        return
    click.echo(
        f"{label}: n={stats['count']}  mean={stats['mean']}  median={stats['median']}  "
        f"min={stats['min']}  max={stats['max']}  stdev={stats['stdev']}"
    )


def run_text_calibration():
    click.echo("=" * 70)
    click.echo("TEXT CALIBRATION")
    click.echo("=" * 70)

    human_files = _list_files(BASE_DIR / "text" / "human", TEXT_EXTENSIONS)
    ai_files = _list_files(BASE_DIR / "text" / "ai", TEXT_EXTENSIONS)

    human_results = []
    for f in human_files:
        result = score_text(f.read_text(encoding="utf-8"))
        human_results.append((f.name, result["overall_score"], result["verdict"]))

    ai_results = []
    for f in ai_files:
        result = score_text(f.read_text(encoding="utf-8"))
        ai_results.append((f.name, result["overall_score"], result["verdict"]))

    _print_group_table("HUMAN samples", human_results)
    _print_group_table("AI samples", ai_results)

    click.echo()
    human_stats = _summarize([s for _, s, _ in human_results])
    ai_stats = _summarize([s for _, s, _ in ai_results])
    _print_summary("Human", human_stats)
    _print_summary("AI   ", ai_stats)

    _print_separation(human_stats, ai_stats)


def run_image_calibration():
    click.echo("=" * 70)
    click.echo("IMAGE CALIBRATION")
    click.echo("=" * 70)

    real_files = _list_files(BASE_DIR / "images" / "real", IMAGE_EXTENSIONS)
    ai_files = _list_files(BASE_DIR / "images" / "ai", IMAGE_EXTENSIONS)
    human_art_files = _list_files(BASE_DIR / "images" / "human_art", IMAGE_EXTENSIONS)

    real_results = []
    for f in real_files:
        result = score_image(str(f))
        real_results.append((f.name, result["overall_score"], result["verdict"]))

    ai_results = []
    for f in ai_files:
        result = score_image(str(f))
        ai_results.append((f.name, result["overall_score"], result["verdict"]))

    human_art_results = []
    for f in human_art_files:
        result = score_image(str(f))
        human_art_results.append((f.name, result["overall_score"], result["verdict"]))

    _print_group_table("REAL (camera photo) samples", real_results)
    _print_group_table("AI-GENERATED samples", ai_results)
    if human_art_results:
        _print_group_table("HUMAN ART / RENDER (non-camera, non-AI) samples -- reference only", human_art_results)

    click.echo()
    real_stats = _summarize([s for _, s, _ in real_results])
    ai_stats = _summarize([s for _, s, _ in ai_results])
    human_art_stats = _summarize([s for _, s, _ in human_art_results])
    _print_summary("Real      ", real_stats)
    _print_summary("AI        ", ai_stats)
    if human_art_results:
        _print_summary("Human art ", human_art_stats)
        click.echo(
            "\nNote: human_art samples are shown for reference, NOT included in the "
            "real-vs-AI separation check below. Your image signals currently test "
            "'was a camera involved', not 'was this AI vs. human-made' -- so "
            "human-made digital art/renders are EXPECTED to score similarly to AI "
            "images here. That's a real scope boundary, not a bug: see project notes."
        )

    _print_separation(real_stats, ai_stats)


def _print_separation(low_group_stats: dict, high_group_stats: dict):
    """Prints a simple, honest read on how well the two groups separate --
    not a formal AUC, just a sanity check on whether the gap is real or
    the groups overlap enough that the current thresholds are unreliable."""
    if low_group_stats["count"] == 0 or high_group_stats["count"] == 0:
        click.echo("\nNot enough samples in both groups to assess separation.")
        return

    gap = high_group_stats["mean"] - low_group_stats["mean"]
    overlap_risk = (
        low_group_stats["max"] > high_group_stats["min"]
        if low_group_stats["max"] is not None and high_group_stats["min"] is not None
        else True
    )

    click.echo(f"\nMean gap (AI/fake - real/human): {round(gap, 1)}")
    if overlap_risk:
        click.echo(
            "WARNING: score ranges overlap between groups (one group's max "
            "exceeds the other's min). Verdict thresholds may misclassify "
            "borderline samples -- add more calibration samples before trusting "
            "the current 40/70 thresholds."
        )
    else:
        click.echo("Groups are cleanly separated (no overlap in min/max range) at current sample size.")


@click.command()
@click.argument("mode", type=click.Choice(["text", "image", "both"]))
def main(mode):
    """Run batch calibration against calibration_set/ folders."""
    if mode in ("text", "both"):
        run_text_calibration()
    if mode in ("image", "both"):
        run_image_calibration()


if __name__ == "__main__":
    main()
