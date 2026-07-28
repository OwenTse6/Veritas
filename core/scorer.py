"""
scorer.py

Combines individual signal scores into one overall verdict, with a
transparent explanation of which signals fired and why. This is the
whole point of the project: instead of a black-box "94% AI" number,
you get a breakdown you can actually reason about.

COMBINATION METHOD -- log-odds addition, not weighted averaging.

Each signal returns a 0-100 "how suspicious is this" score, with 50 as
the neutral midpoint. We convert that into a log-odds contribution and
SUM contributions across signals, then squash the total back into a
0-100 score with a logistic function. This is the same principle behind
naive Bayes: independent pieces of evidence combine multiplicatively in
probability space, which is additive in log-odds space.

Why this replaces the old weighted-average-with-override: a plain
weighted average dilutes a strongly-firing signal with several merely
neutral ones (a real bug we hit in practice -- a 90+ signal on a fully
synthetic image got dragged down to 30-something by three signals that
just didn't apply to that image). We patched that earlier with a
hard-coded "if any signal >= 80, override the average" rule, which
worked but was ad hoc and had a hard edge at exactly 80. Log-odds
addition fixes this properly: evidence ADDS instead of averaging, so one
very confident signal naturally dominates the sum without needing a
special-cased threshold, and several weak-but-aligned signals can also
add up to a confident verdict on their own.

Per-signal CONFIDENCE values (replacing the old WEIGHTS) control how much
each signal's evidence counts -- a higher confidence means that signal's
extreme readings can push the final verdict further on their own. These
still need real calibration data to set precisely; treat the current
values as informed starting points, not final answers.
"""

from __future__ import annotations

import math

from .text_signals import analyze_text
from .image_signals import analyze_image

# How much each signal's evidence counts, in log-odds units. Roughly:
# confidence 2-3 = a supporting signal; 4-6 = a signal that can single-
# handedly push the verdict to "likely AI" on its own if it maxes out.
TEXT_CONFIDENCE = {
    "burstiness": 2.5,
    "vocab_fingerprint": 3.0,
    "structural_repetition": 2.0,
    "punctuation_signature": 1.8,
    "evaluative_language": 2.5,
}


IMAGE_CONFIDENCE = {
    "metadata_check": 3.0,       # near-certain when it actually has evidence
    "camera_exif_check": 6.0,    # full EXIF is hard to fake
    "error_level_analysis": 2.0,
    "fft_artifact_check": 1.5,
    "noise_consistency_check": 2.0,
    # SEVERELY DOWNGRADED (was 4.0): tested against 15 confirmed-real web
    # images and found TOTAL distributional overlap with known AI samples
    # (real noise floors ranged 0.0-19.6, AI ranged 1.07-8.07 -- no
    # threshold separates them). Real-world images that have been resized/
    # recompressed for web hosting can legitimately show near-zero noise,
    # sometimes more extreme than actual AI output. This assumption only
    # holds for relatively unprocessed images -- see LIMITATIONS.md.
    "noise_floor_check": 1.0,
    # SEVERELY DOWNGRADED (was 3.5): same overlap problem -- real photos'
    # brightness-noise correlation ranged -0.522 to 0.604, AI ranged
    # -0.243 to 0.394, again fully overlapping.
    "signal_dependent_noise_check": 1.0,
    "chromatic_aberration_check": 2.5,  # promising but not yet well-calibrated
}


def _combine_log_odds(signals: dict, confidence: dict, low_score_damping: float = 0.2) -> float:
    """Combines signal scores via log-odds addition instead of averaging.

    Each score in [0, 100] is centered at 50 (neutral). Scores ABOVE 50
    contribute positive log-odds (evidence toward AI) at full confidence.
    Scores BELOW 50 are damped by `low_score_damping` before contributing
    negative log-odds (evidence toward real/human).

    Why asymmetric: most of these signals are narrow tests for one
    specific artifact (an upsampling pattern, a compression mismatch, a
    physics mismatch in noise). A HIGH score means "found the specific
    thing this test looks for" -- strong, direct evidence. A LOW score
    mostly means "this particular test didn't find its thing here" --
    which is much weaker evidence of authenticity, since the image could
    still be fake via a mechanism this signal doesn't cover. Treating low
    and high scores as equally strong (symmetric) evidence was a real bug
    we hit in practice: three signals that had simply found nothing
    outvoted two signals that had found strong, specific evidence, even
    though "found nothing" and "found evidence of authenticity" are not
    the same thing.
    """
    active = {
        name: k
        for name, k in confidence.items()
        if not signals[name].get("abstain", False)
    }

    if not active:
        return 0.0

    total_log_odds = 0.0
    for name, k in active.items():
        score = signals[name]["score"]
        normalized = (score - 50.0) / 50.0  # in [-1, 1]
        if normalized < 0:
            normalized *= low_score_damping
        total_log_odds += normalized * k

    probability = 1.0 / (1.0 + math.exp(-total_log_odds))
    return round(probability * 100.0, 1)


def score_text(text: str) -> dict:
    signals = analyze_text(text)

    overall = _combine_log_odds(signals, TEXT_CONFIDENCE)
    verdict = _text_verdict_label(overall)

    return {
        "overall_score": overall,
        "verdict": verdict,
        "signals": signals,
        "explanation": _explain(signals),
    }


def _text_verdict_label(score: float) -> str:
    if score >= 70:
        return "likely AI-generated"
    if score >= 40:
        return "mixed / uncertain"
    return "likely human-written"


def _image_verdict_label(score: float) -> str:
    if score >= 70:
        return "likely AI-generated"
    if score >= 40:
        return "mixed / uncertain"
    return "likely a real, unedited photo"


def _explain(signals: dict) -> list[str]:
    """Turn raw signal detail into short, human-readable explanation lines."""
    lines = []

    b = signals["burstiness"]["detail"]
    if "coefficient_of_variation" in b:
        lines.append(
            f"Sentence length variance (CV={b['coefficient_of_variation']}): "
            + ("unusually uniform, a common AI trait." if signals["burstiness"]["score"] > 50
               else "shows natural human-like variation.")
        )

    v = signals["vocab_fingerprint"]["detail"]
    top_hits = v.get("top_hits", [])
    if top_hits:
        phrases = ", ".join(f"'{h['phrase']}'" for h in top_hits[:5])
        lines.append(f"Found common AI-associated phrases: {phrases}.")
    else:
        lines.append("No common AI-associated phrases detected.")

    s = signals["structural_repetition"]["detail"]
    if s.get("template_hits"):
        templates = ", ".join(s["template_hits"].keys())
        lines.append(f"Repeated sentence templates detected: {templates}.")
    if s.get("repeated_fourgram_ratio", 0) > 0.05:
        lines.append("Noticeable repeated phrase structures across the text.")

    p = signals["punctuation_signature"]["detail"]
    if p.get("em_dash_per_1000_words", 0) > 3:
        lines.append(
            f"Elevated em-dash usage ({p['em_dash_per_1000_words']} per 1000 words)."
        )
    if p.get("rule_of_three_per_1000_words", 0) > 3:
        lines.append("Frequent 'rule of three' list constructions detected.")

    e = signals["evaluative_language"]["detail"]
    if e.get("superlative_count", 0) > 0 and signals["evaluative_language"]["score"] > 30:
        praises = ", ".join(list(e.get("superlatives_found", {}).keys())[:5])
        lines.append(
            f"Heavy unqualified praise detected ({praises}) with little hedging "
            f"(praise-to-hedge ratio={e['praise_to_hedge_ratio']})."
        )

    return lines


def score_image(image_path: str) -> dict:
    signals = analyze_image(image_path)

    overall = _combine_log_odds(signals, IMAGE_CONFIDENCE)
    verdict = _image_verdict_label(overall)

    return {
        "overall_score": overall,
        "verdict": verdict,
        "signals": signals,
        "explanation": _explain_image(signals),
    }


def _explain_image(signals: dict) -> list[str]:
    lines = []

    m = signals["metadata_check"]["detail"]
    if m.get("ai_software_tags_found"):
        tags = ", ".join(m["ai_software_tags_found"])
        lines.append(f"Metadata references known AI tools: {tags}.")
    if m.get("c2pa_detected"):
        lines.append("C2PA content credentials detected.")
    if not m.get("ai_software_tags_found") and not m.get("c2pa_detected"):
        lines.append("No AI-related metadata found (not conclusive -- easily stripped).")

    f = signals["fft_artifact_check"]["detail"]
    if signals["fft_artifact_check"]["score"] > 50:
        lines.append(
            f"Frequency spectrum shows periodic spikes (spike ratio={f.get('spike_ratio')}), "
            "consistent with upsampling artifacts."
        )

    e = signals["error_level_analysis"]["detail"]
    if signals["error_level_analysis"]["score"] > 50:
        count = len(e.get("top_outlier_patches", []))
        lines.append(
            f"Found {count} region(s) with a compression 'error level' that doesn't "
            f"match the rest of the image (outlier ratio={e.get('outlier_patch_ratio')}), "
            "suggesting content spliced in from elsewhere."
        )

    n = signals["noise_consistency_check"]["detail"]
    if signals["noise_consistency_check"]["score"] > 50:
        patch_count = len(n.get("top_anomalous_patches", []))
        lines.append(
            f"Found {patch_count} region(s) whose noise level sharply differs from "
            f"a wider ring of surrounding patches (anomalous patch ratio="
            f"{n.get('anomalous_patch_ratio')}), consistent with a spliced-in element."
        )

    nf = signals["noise_floor_check"]["detail"]
    if signals["noise_floor_check"]["score"] > 50:
        lines.append(
            f"Flattest regions of the image show unusually low noise "
            f"(std={nf.get('mean_flat_region_noise_std')}), more consistent with a "
            "rendered/painted image than a real camera photo."
        )

    sdn = signals["signal_dependent_noise_check"]["detail"]
    if signals["signal_dependent_noise_check"]["score"] > 50:
        lines.append(
            f"Noise level doesn't rise with brightness the way real camera shot-noise "
            f"physics would (correlation={sdn.get('brightness_noise_correlation')}), "
            "suggesting the noise/grain wasn't produced by an actual sensor."
        )

    ca = signals["chromatic_aberration_check"]["detail"]
    if signals["chromatic_aberration_check"]["score"] > 50:
        lines.append(
            f"Color-channel edge misalignment doesn't increase toward the image "
            f"edges the way real lens dispersion would (radial correlation="
            f"{ca.get('radial_shift_correlation')}), suggesting no real lens was involved."
        )

    return lines


# --------------------------------------------------------------------------
# Calibration notes
# --------------------------------------------------------------------------
# 1. Collect a real labeled calibration set (see calibrate.py) -- at least
#    15-20 samples per group, covering multiple styles/genres/generators.
# 2. Run calibrate.py to see score distributions per group and whether
#    they overlap.
# 3. Tune TEXT_CONFIDENCE / IMAGE_CONFIDENCE (not raw weights anymore --
#    these are log-odds strengths). Raise a signal's confidence if it's
#    reliably separating the two groups on its own; lower it if it's noisy
#    or frequently wrong. A signal that's often flatly uninformative should
#    stay low-confidence even if it's occasionally decisive.
# 4. Re-run calibrate.py after every change to catch regressions -- a fix
#    for one case can quietly change scores elsewhere (we found this
#    happening with the text/image combination logic sharing code).
