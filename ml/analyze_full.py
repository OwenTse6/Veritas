"""
analyze_full.py

Combines the trained ML classifier's AI-generation score with:
  - heuristic editing/tampering signals from the heuristic project
    (Error Level Analysis, regional noise consistency, chromatic
    aberration -- see core/image_signals.py), plus a new Grad-CAM
    spatial-concentration signal
  - camera EXIF / AI-generation-metadata evidence, which the CNN has
    NO way to see on its own (it only ever looks at pixels)

...to report scores the binary classifier alone couldn't express:

    1. AI-GENERATION likelihood -- was this whole image synthesized?
       (CNN score, adjusted by hard EXIF/metadata evidence when present)
    2. TAMPERING/EDITING likelihood -- does this show signs of a
       localized edit/splice, regardless of whether it's AI or real?

Why the EXIF adjustment matters: we found a real photo (dsc_2049.jpg --
a genuine DSLR photo with full camera EXIF: GPS, aperture, shutter
speed, ISO) that the CNN confidently misclassified as AI-generated,
almost certainly because professional studio product photography has
unnaturally low sensor noise -- the exact same blind spot that broke
the ORIGINAL heuristic noise_floor_check hours earlier in this project,
before camera_exif_check was built to counter it. The CNN has no
equivalent defense, since it never sees metadata, only pixels. Camera
EXIF is hard to fake convincingly (an internally-consistent GPS +
aperture + shutter speed + ISO cluster), so when it's present, we let
it pull the final AI-generation verdict toward "real" -- similarly,
if AI generation parameters are found in metadata (Stable Diffusion/
ComfyUI-style embedded parameters), that pulls the verdict toward "AI".
Absence of either kind of metadata changes nothing -- it's common and
proves nothing either way, so the CNN's raw score stands unadjusted.

Why the tampering axis matters: a real photo with something spliced
into it should ideally score LOW on #1 (it's mostly real) but HIGH on
#2 (something was edited). The binary classifier alone can't express
that distinction -- it was forced to call it one or the other.

GRAD-CAM CONCENTRATION SIGNAL: if the model's attention (from Grad-CAM)
is concentrated in one small region rather than spread across the whole
image, that's suggestive of a localized edit rather than either a clean
photo or whole-image AI generation -- a genuinely new, nearly-free signal
using a model you've already trained.

Usage:
    python analyze_full.py --checkpoint checkpoints/best_model.pt --image somefile.jpg
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))  # so we can import core/

from core.image_signals import (
    error_level_analysis, noise_consistency_check, chromatic_aberration_check,
    metadata_check, camera_exif_check,
)
from core.scorer import _combine_log_odds

from dataset import get_eval_transforms
from gradcam import GradCAM
from infer import load_model

TAMPERING_CONFIDENCE = {
    "error_level_analysis": 2.0,
    "noise_consistency_check": 2.0,
    "chromatic_aberration_check": 2.5,
    "heatmap_concentration": 2.0,
}

# Combines the CNN's raw AI-generation score with hard metadata evidence
# the CNN structurally cannot see (it only ever looks at pixels).
CAMERA_EXIF_CONFIDENCE = 6.0   # hard to fake a consistent GPS+aperture+shutter+ISO cluster
AI_METADATA_CONFIDENCE = 5.0   # hard to fake embedded AI generation parameters


def combine_final_verdict(cnn_ai_probability: float, camera_exif_result: dict, metadata_result: dict) -> float:
    """
    Combines the CNN's own probability with metadata evidence correctly.

    BUG THIS FIXES: an earlier version ran the CNN's probability through
    the same log-odds-plus-damping transform built for heuristic 0-100
    "suspicion scores" (like error_level_analysis or noise_consistency).
    That regressed extreme CNN scores toward 50 even with ZERO metadata
    evidence present -- a raw CNN score of 0.0% came out as 35.4% after
    "combining" with two signals that had both abstained. That's wrong:
    an abstaining signal should never change the result. The heuristic
    signals' scores aren't literal probabilities (a 100 there means
    "strongest evidence this heuristic can give", not "100% certain"),
    but the CNN's softmax output genuinely IS a calibrated probability,
    so it needs the standard logit transform (log(p/(1-p))), not the
    damped heuristic-score transform used for everything else.

    With this fix: if both metadata signals abstain, the result is
    mathematically identical to the CNN's own raw probability (sigmoid
    exactly undoes logit) -- metadata only shifts the verdict when it
    actually fires.
    """
    p = max(0.001, min(0.999, cnn_ai_probability / 100.0))
    total_log_odds = math.log(p / (1 - p))  # CNN's own calibrated contribution, undamped

    for result, confidence in [(camera_exif_result, CAMERA_EXIF_CONFIDENCE),
                                (metadata_result, AI_METADATA_CONFIDENCE)]:
        if result.get("abstain", False):
            continue
        score = result["score"]
        normalized = (score - 50.0) / 50.0
        if normalized < 0:
            normalized *= 0.2  # same damping convention used elsewhere in the project
        total_log_odds += normalized * confidence

    probability = 1.0 / (1.0 + math.exp(-total_log_odds))
    return round(probability * 100.0, 1)


def heatmap_concentration_signal(heatmap: np.ndarray) -> dict:
    """
    Measures how concentrated the Grad-CAM heatmap's activation is.
    A localized edit/splice should draw the model's attention to a small
    region; a clean real photo or whole-image AI generation should spread
    attention more evenly across the frame.

    Metric: what fraction of total activation falls in the "hottest" 10%
    of pixels. A perfectly uniform heatmap has this at exactly 0.10 by
    definition; a highly concentrated one approaches 1.0.
    """
    flat = heatmap.flatten()
    total = flat.sum()

    if total < 1e-6:
        return {
            "score": 0.0,
            "abstain": True,
            "detail": {"reason": "no significant model activation to measure"},
        }

    sorted_vals = np.sort(flat)[::-1]
    top_10pct_count = max(1, int(len(sorted_vals) * 0.1))
    top_10pct_sum = sorted_vals[:top_10pct_count].sum()
    concentration_ratio = float(top_10pct_sum / total)

    baseline = 0.10  # a perfectly uniform heatmap's top-10%-of-pixels holds exactly 10% of total heat
    score = max(0.0, min(100.0, (concentration_ratio - baseline) / (1 - baseline) * 100))

    return {
        "score": round(score, 1),
        "detail": {
            "concentration_ratio": round(concentration_ratio, 3),
            "note": "High score = model's attention is concentrated in a small region "
                    "(suggestive of a localized edit) rather than spread across the image.",
        },
    }


def analyze(checkpoint_path: str, image_path: str, device: torch.device):
    model, classes = load_model(checkpoint_path, device)

    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    transform = get_eval_transforms()
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]
    ai_probability = probs[classes.index("ai")].item() * 100

    # Grad-CAM heatmap, used both for the concentration signal and available
    # for visual inspection if the caller wants to save it.
    cam = GradCAM(model, target_layer=model.layer4[-1])
    heatmap, _ = cam.generate(input_tensor)

    tampering_signals = {
        "error_level_analysis": error_level_analysis(image_path),
        "noise_consistency_check": noise_consistency_check(image_path),
        "chromatic_aberration_check": chromatic_aberration_check(image_path),
        "heatmap_concentration": heatmap_concentration_signal(heatmap),
    }
    tampering_score = _combine_log_odds(tampering_signals, TAMPERING_CONFIDENCE)

    # Adjust the CNN's raw AI-generation score with hard metadata evidence
    # it has no way to see on its own.
    camera_exif_result = camera_exif_check(image_path)
    metadata_result = metadata_check(image_path)
    final_ai_score = combine_final_verdict(ai_probability, camera_exif_result, metadata_result)

    return {
        "cnn_raw_score": round(ai_probability, 1),
        "ai_generation_score": final_ai_score,
        "ai_generation_verdict": "likely AI-generated" if final_ai_score > 50 else "likely not AI-generated",
        "metadata_signals": {
            "camera_exif_check": camera_exif_result,
            "metadata_check": metadata_result,
        },
        "tampering_score": tampering_score,
        "tampering_verdict": "shows signs of localized editing/tampering" if tampering_score > 50
                              else "no strong signs of localized editing detected",
        "tampering_signals": tampering_signals,
        "heatmap": heatmap,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = analyze(args.checkpoint, args.image, device)

    print(f"\nAI-generation:  {result['ai_generation_score']:.1f}%  -- {result['ai_generation_verdict']}")
    if abs(result['ai_generation_score'] - result['cnn_raw_score']) > 1.0:
        print(f"  (CNN alone said {result['cnn_raw_score']:.1f}% -- adjusted by camera EXIF / "
              f"AI-metadata evidence the CNN can't see on its own)")
    print(f"Tampering:      {result['tampering_score']:.1f}%  -- {result['tampering_verdict']}")

    print("\nMetadata evidence:")
    for name, sig in result["metadata_signals"].items():
        flag = " (abstained -- no evidence either way)" if sig.get("abstain") else " (fired -- real evidence found)"
        print(f"  {name}: {sig['score']}{flag}")

    print("\nTampering signal breakdown:")
    for name, sig in result["tampering_signals"].items():
        flag = " (abstained)" if sig.get("abstain") else ""
        print(f"  {name}: {sig['score']}{flag}")
    print()


if __name__ == "__main__":
    main()
