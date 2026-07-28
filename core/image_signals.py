"""
image_signals.py

Heuristic signals for detecting likely AI-generated images.
Skeleton for Sunday -- metadata check is fully implemented since it's
the cheapest, highest-confidence signal. FFT and noise-consistency
checks are stubbed with a working starting point to build on.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ExifTags


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Signal 1: Metadata / C2PA content credentials
# --------------------------------------------------------------------------
#
# Many AI image tools now embed content credentials (C2PA) or at least
# generator info in EXIF/XMP metadata. Cheapest, most reliable signal
# when present -- but absence proves nothing (metadata is easy to strip).
#
# Beyond just naming the software, local Stable Diffusion / ComfyUI /
# Automatic1111 setups commonly embed the ENTIRE generation recipe as a
# raw PNG text chunk unless the user deliberately strips it -- things like
# "Steps: 30, Sampler: Euler a, CFG scale: 7, Seed: 12345". This is an
# extremely specific, high-confidence tell: no human illustration tool
# writes "CFG scale" into a file, so finding this pattern is close to
# definitive, with essentially no false-positive risk on real art.

_KNOWN_AI_SOFTWARE_TAGS = [
    "midjourney", "dall-e", "dalle", "stable diffusion", "stability ai",
    "adobe firefly", "runway", "leonardo.ai", "ideogram", "comfyui",
    "automatic1111", "novelai", "invokeai", "sdxl", "flux.1", "flux1",
]

# Generation-parameter field names commonly embedded by SD/ComfyUI/A1111
# tooling. Finding several of these together is a very strong tell.
_GENERATION_PARAM_MARKERS = [
    "cfg scale", "sampler:", "steps:", "model hash", "denoising strength",
    "clip skip", "seed:", "negative prompt:", "hires upscale", "lora:",
    "checkpoint:", "vae:",
]


def metadata_check(image_path: str | Path) -> dict:
    img = Image.open(image_path)
    info = img.info or {}
    exif = getattr(img, "_getexif", lambda: None)()

    found_tags = []
    found_param_markers = []

    # Check PNG text chunks / info dict (where many AI tools stash metadata).
    combined_info_text = ""
    for key, value in info.items():
        text = f"{key}={value}".lower()
        combined_info_text += text + " "
        for tag in _KNOWN_AI_SOFTWARE_TAGS:
            if tag in text:
                found_tags.append(tag)

    for marker in _GENERATION_PARAM_MARKERS:
        if marker in combined_info_text:
            found_param_markers.append(marker)

    # Check EXIF software field if present.
    if exif:
        software = str(exif.get(305, "")).lower()  # 305 = Software tag
        for tag in _KNOWN_AI_SOFTWARE_TAGS:
            if tag in software:
                found_tags.append(tag)

    has_c2pa = "c2pa" in str(info).lower() or "content credentials" in str(info).lower()

    # Two or more generation-parameter markers together is very hard to
    # explain any other way -- treat it as strong evidence on its own,
    # even with no recognized software name attached.
    has_param_signature = len(found_param_markers) >= 2

    has_evidence = bool(found_tags) or has_c2pa or has_param_signature
    score = 95.0 if has_evidence else 0.0

    return {
        "score": score,
        "abstain": not has_evidence,
        "detail": {
            "ai_software_tags_found": list(set(found_tags)),
            "c2pa_detected": has_c2pa,
            "generation_param_markers_found": found_param_markers,
            "note": "Absence of metadata does NOT imply human-made -- metadata is trivially stripped.",
        },
    }


# --------------------------------------------------------------------------
# Signal 2: Frequency-domain artifacts (FFT)
# --------------------------------------------------------------------------
#
# GAN/diffusion upsampling steps can leave periodic artifacts visible as
# peaks in the frequency spectrum. This is a STARTING POINT -- expect to
# tune the peak-detection threshold against real samples.

def fft_artifact_check(image_path: str | Path) -> dict:
    img = Image.open(image_path).convert("L")  # grayscale
    arr = np.asarray(img, dtype=np.float32)

    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    # Look at the high-frequency ring (outer portion of spectrum) for
    # unusually sharp, periodic peaks vs. the smooth falloff typical of
    # natural camera images.
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    outer_radius = min(h, w) // 3

    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    outer_mask = dist > outer_radius

    outer_values = magnitude[outer_mask]
    mean_outer = float(outer_values.mean())
    std_outer = float(outer_values.std())

    # Spikes far above the local mean suggest periodic artifacts.
    threshold = mean_outer + 3 * std_outer
    spike_ratio = float((outer_values > threshold).mean())

    # Calibration knob.
    score = min(spike_ratio * 5000, 100.0)

    return {
        "score": round(score, 1),
        "detail": {
            "mean_outer_magnitude": round(mean_outer, 3),
            "std_outer_magnitude": round(std_outer, 3),
            "spike_ratio": round(spike_ratio, 5),
            "note": "Heuristic starting point -- calibrate threshold against known samples.",
        },
    }


# --------------------------------------------------------------------------
# Signal 3: Error Level Analysis (ELA) -- compression-history mismatch
# --------------------------------------------------------------------------
#
# JPEG compresses in 8x8 blocks and is lossy. Re-saving an already-compressed
# JPEG at a known quality barely changes it (it's near a local minimum), so
# the pixel-level difference ("error level") is low almost everywhere.
#
# A region that was pasted in from elsewhere -- a render, a screenshot from
# a different source, an object composited in fresh -- usually has a
# DIFFERENT compression history than the surrounding photo. When the whole
# thing is re-saved, that region reacts differently under recompression and
# stands out as a patch of unusually high (or unusually low) error relative
# to the rest of the image.
#
# We use a robust median-based outlier test (rather than mean/stdev) because
# it isn't thrown off by a photo that legitimately has some naturally
# high-detail, high-contrast areas (which also nudges error level up a bit).

def error_level_analysis(image_path: str | Path, quality: int = 90, patch_size: int = 32) -> dict:
    img = Image.open(image_path).convert("RGB")

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer).convert("RGB")

    orig_arr = np.asarray(img, dtype=np.float32)
    resaved_arr = np.asarray(resaved, dtype=np.float32)

    # Average absolute difference across RGB channels = the ELA map.
    ela_map = np.abs(orig_arr - resaved_arr).mean(axis=2)

    h, w = ela_map.shape
    rows = (h - patch_size) // patch_size
    cols = (w - patch_size) // patch_size

    if rows < 3 or cols < 3:
        return {
            "score": 0.0,
            "abstain": True,
            "detail": {"reason": "image too small for ELA patch analysis"},
        }

    patch_means = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            y, x = r * patch_size, c * patch_size
            patch_means[r, c] = ela_map[y:y + patch_size, x:x + patch_size].mean()

    flat = patch_means.flatten()
    median = float(np.median(flat))
    mad = float(np.median(np.abs(flat - median))) + 1e-6
    # 1.4826 makes MAD comparable to a standard deviation for normal data.
    robust_z = (flat - median) / (1.4826 * mad)

    outlier_mask = np.abs(robust_z) > 3.5
    outlier_ratio = float(outlier_mask.mean())

    outlier_patches = []
    for idx in np.where(outlier_mask)[0]:
        r, c = divmod(idx, cols)
        outlier_patches.append({
            "row": int(r), "col": int(c),
            "ela_value": round(float(patch_means[r, c]), 3),
            "robust_z": round(float(robust_z[idx]), 2),
        })
    outlier_patches.sort(key=lambda p: -abs(p["robust_z"]))

    score = min(outlier_ratio * 400, 100.0)

    return {
        "score": round(score, 1),
        "detail": {
            "grid_size": f"{rows}x{cols}",
            "median_ela": round(median, 3),
            "outlier_patch_ratio": round(outlier_ratio, 4),
            "top_outlier_patches": outlier_patches[:8],
            "note": "Flags patches whose compression 'error level' is a robust "
                    "outlier vs. the rest of the image -- a signature of content "
                    "with a different compression history spliced into a photo.",
        },
    }


# --------------------------------------------------------------------------
# Signal 4: Noise consistency (sensor noise patchwork)
# --------------------------------------------------------------------------

def noise_consistency_check(image_path: str | Path, patch_size: int = 32, ring_radius: int = 3) -> dict:
    """
    Compares each patch's noise level to a RING of patches at `ring_radius`
    distance (not immediate neighbors, and not the whole-image average).
    A real photo can legitimately have very different noise levels across
    the frame (shadow vs. bright light, sharp foreground vs. blurry
    background) -- that gradual, global variation should NOT be flagged.
    What's suspicious is a patch that disagrees with a wider ring around it,
    reaching far enough out to land back on genuine, untouched background
    even for a sizeable spliced-in region.
    """
    img = Image.open(image_path).convert("L")
    arr = np.asarray(img, dtype=np.float32)

    h, w = arr.shape
    rows = (h - patch_size) // patch_size
    cols = (w - patch_size) // patch_size

    min_dim = 2 * ring_radius + 1
    if rows < min_dim or cols < min_dim:
        return {
            "score": 0.0,
            "abstain": True,
            "detail": {"reason": "image too small for regional patch analysis at this ring radius"},
        }

    grid = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            y, x = r * patch_size, c * patch_size
            patch = arr[y:y + patch_size, x:x + patch_size]
            noise = patch - patch.mean()
            grid[r, c] = noise.std()

    deviations = []
    anomalous_patches = []
    for r in range(rows):
        for c in range(cols):
            ring_vals = []
            for dr in range(-ring_radius, ring_radius + 1):
                for dc in range(-ring_radius, ring_radius + 1):
                    if max(abs(dr), abs(dc)) != ring_radius:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        ring_vals.append(grid[nr, nc])

            if not ring_vals:
                continue

            ring_mean = float(np.mean(ring_vals))
            deviation = abs(grid[r, c] - ring_mean) / (ring_mean + 1e-6)
            deviations.append(deviation)
            if deviation > 1.0:
                anomalous_patches.append({"row": r, "col": c, "deviation": round(deviation, 2)})

    deviations = np.array(deviations)
    mean_deviation = float(deviations.mean())
    max_deviation = float(deviations.max())
    anomalous_ratio = len(anomalous_patches) / len(deviations)

    score = min(anomalous_ratio * 300, 100.0)
    anomalous_patches.sort(key=lambda p: -p["deviation"])

    return {
        "score": round(score, 1),
        "detail": {
            "grid_size": f"{rows}x{cols}",
            "ring_radius_patches": ring_radius,
            "mean_local_deviation": round(mean_deviation, 3),
            "max_local_deviation": round(max_deviation, 3),
            "anomalous_patch_ratio": round(anomalous_ratio, 3),
            "top_anomalous_patches": anomalous_patches[:8],
            "note": "Flags patches whose noise differs sharply from a wider ring "
                    "of patches around them -- tolerates legitimate lighting/focus "
                    "variation across a scene while still reaching past sizeable edits.",
        },
    }


# --------------------------------------------------------------------------
# Signal 5: Noise floor (absolute smoothness check)
# --------------------------------------------------------------------------
#
# The other image signals (FFT, ELA, regional noise consistency) all target
# "was a real photo altered/spliced". They're blind to a different case:
# an image that's ENTIRELY synthetic -- a digital painting or diffusion-model
# illustration -- with no real camera involved anywhere in it. Since nothing
# is spliced in, there's no internal mismatch for those signals to catch;
# everything is uniformly "clean", which those signals misread as evidence
# of a real photo.
#
# The tell here is different: real camera sensors always leave a baseline
# noise floor, even in the flattest, most evenly-lit part of a photo (open
# sky, a plain wall). Digital paintings/renders often have unnaturally
# smooth flat regions -- there's no sensor noise to be added.
#
# To avoid false-flagging real photos that are simply well-exposed and
# low-ISO, we specifically look at the FLATTEST patches (lowest local
# gradient -- i.e. areas with little detail/edges, like open sky), since
# that's where any real sensor noise floor would be easiest to see and
# hardest for a smooth illustration to fake.

def noise_floor_check(image_path: str | Path, patch_size: int = 32, flat_percentile: float = 25.0) -> dict:
    img = Image.open(image_path).convert("L")
    arr = np.asarray(img, dtype=np.float32)

    h, w = arr.shape
    rows = (h - patch_size) // patch_size
    cols = (w - patch_size) // patch_size

    if rows < 2 or cols < 2:
        return {
            "score": 0.0,
            "abstain": True,
            "detail": {"reason": "image too small for noise-floor patch analysis"},
        }

    grad_grid = np.zeros((rows, cols), dtype=np.float32)
    noise_grid = np.zeros((rows, cols), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            y, x = r * patch_size, c * patch_size
            patch = arr[y:y + patch_size, x:x + patch_size]

            gy, gx = np.gradient(patch)
            grad_grid[r, c] = float(np.sqrt(gx ** 2 + gy ** 2).mean())

            noise = patch - patch.mean()
            noise_grid[r, c] = float(noise.std())

    flat_threshold = float(np.percentile(grad_grid.flatten(), flat_percentile))
    flat_mask = grad_grid <= flat_threshold
    flat_noise_vals = noise_grid[flat_mask]

    if flat_noise_vals.size == 0:
        return {
            "score": 0.0,
            "abstain": True,
            "detail": {"reason": "no sufficiently flat regions found to evaluate"},
        }

    mean_flat_noise = float(flat_noise_vals.mean())

    # Calibration knob, now grounded in real measurements: two real photos
    # from this project measured 10.8 and 12.8 in their flattest regions,
    # while a fully-synthetic illustration measured 1.07 -- an order of
    # magnitude apart. A baseline of 8.0 gives real photos comfortable
    # headroom to stay at 0 while letting genuinely synthetic-smooth images
    # score close to 100 instead of being capped by an overly cautious curve.
    baseline = 8.0
    score = max(0.0, min(100.0, (baseline - mean_flat_noise) / baseline * 100))

    return {
        "score": round(score, 1),
        "detail": {
            "flat_patch_count": int(flat_mask.sum()),
            "mean_flat_region_noise_std": round(mean_flat_noise, 3),
            "note": "Looks at the flattest (lowest-detail) patches specifically -- "
                    "real sensor noise floor should still show up there even in a "
                    "clean, well-lit photo. Unnaturally low noise in flat regions "
                    "suggests a rendered/painted image rather than a photograph.",
        },
    }


# --------------------------------------------------------------------------
# Signal 6: Signal-dependent noise correlation (shot-noise physics check)
# --------------------------------------------------------------------------
#
# noise_floor_check needs a genuinely flat region to test -- it has nothing
# to say about a densely textured image (heavy smoke, fabric, foliage) where
# every patch has real visual detail. This signal doesn't need flat patches
# at all.
#
# Real camera sensors exhibit "shot noise": noise variance increases with
# local brightness (more photons hitting the sensor -> more photon-counting
# variance), roughly following signal level up until the sensor saturates.
# This is a physical property of the SENSOR, not the scene -- it holds in a
# plain sky patch and in a heavily textured smoke patch alike.
#
# AI-generated grain, even when deliberately added for realism, is typically
# applied at a roughly uniform strength regardless of local brightness --
# generators aren't simulating photon-counting physics, just adding a
# texture pattern. So we check: across ALL patches (any content), does
# noise level actually correlate with local brightness the way a real
# sensor would, or does it stay flat/uncorrelated?

def signal_dependent_noise_check(image_path: str | Path, patch_size: int = 24) -> dict:
    img = Image.open(image_path).convert("L")
    arr = np.asarray(img, dtype=np.float32)

    h, w = arr.shape
    rows = (h - patch_size) // patch_size
    cols = (w - patch_size) // patch_size

    if rows < 4 or cols < 4:
        return {
            "score": 0.0,
            "abstain": True,
            "detail": {"reason": "image too small for signal-dependent noise analysis"},
        }

    brightness_vals = []
    noise_vals = []

    for r in range(rows):
        for c in range(cols):
            y, x = r * patch_size, c * patch_size
            patch = arr[y:y + patch_size, x:x + patch_size]

            # Skip near-saturated/near-black patches -- shot noise behavior
            # breaks down at the extremes (clipping), which would just add
            # noise to the correlation measurement itself.
            mean_val = float(patch.mean())
            if mean_val < 10 or mean_val > 245:
                continue

            residual = patch - mean_val
            noise_std = float(residual.std())

            brightness_vals.append(mean_val)
            noise_vals.append(noise_std)

    if len(brightness_vals) < 10:
        return {
            "score": 0.0,
            "abstain": True,
            "detail": {"reason": "not enough valid (non-clipped) patches to measure correlation"},
        }

    brightness_arr = np.array(brightness_vals)
    noise_arr = np.array(noise_vals)

    correlation = float(np.corrcoef(brightness_arr, noise_arr)[0, 1])
    if np.isnan(correlation):
        correlation = 0.0

    # Real sensor noise: expect a clear positive correlation (shot noise
    # rising with brightness). Weak/near-zero/negative correlation is the
    # suspicious case -- suggests noise wasn't generated by real sensor
    # physics, regardless of how much texture or grain is actually present.
    #
    # Calibration knob: real photos often land around 0.3-0.6+; synthetic
    # grain tends to sit closer to 0.
    if correlation >= 0.35:
        score = 5.0
    elif correlation <= 0.0:
        score = 90.0
    else:
        score = 90.0 - (correlation / 0.35) * 85.0

    return {
        "score": round(score, 1),
        "detail": {
            "patch_count_used": len(brightness_vals),
            "brightness_noise_correlation": round(correlation, 3),
            "note": "Checks whether noise increases with brightness the way real "
                    "camera shot-noise physics would -- works even on densely "
                    "textured images where no flat region exists to test.",
        },
    }


# --------------------------------------------------------------------------
# Signal 7: Chromatic aberration (lens optics check)
# --------------------------------------------------------------------------
#
# Real camera lenses bend different wavelengths of light by slightly
# different amounts (dispersion), causing the red/blue channels to be very
# subtly offset from green at high-contrast edges -- classic color fringing.
# Critically, this isn't random: it's a lens artifact, so it gets STRONGER
# radially, further from the image center, in a physically consistent way.
#
# This is a genuinely different physical property from noise/grain -- it's
# about optics, not sensor statistics -- so a generator faking realistic
# noise (as we found earlier) has no particular reason to also be faking a
# radially-increasing channel misalignment pattern.
#
# We estimate the R-vs-G and B-vs-G sub-pixel shift in patches across the
# image using phase correlation (a standard FFT-based shift-estimation
# technique), then check whether that shift grows with distance from the
# image center the way real lens dispersion would.

def _phase_correlation_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Estimate the (y, x) shift between two same-sized patches via FFT
    phase correlation, refined to SUB-PIXEL precision using parabolic
    interpolation around the correlation peak -- a standard technique from
    optical flow / image registration.

    A whole-pixel-only version of this (taking the raw argmax) turned out
    to be non-discriminative here: real chromatic aberration is almost
    always a fraction of a pixel, so every image -- real or synthetic --
    just reported an integer shift of ~0. The parabolic refinement below
    fits a small quadratic through the peak and its immediate neighbors to
    recover the true sub-pixel offset, which is where the actual signal lives.
    """
    h, w = a.shape

    # A Hanning window tapers the patch edges to reduce spectral leakage,
    # which otherwise distorts the correlation peak and hurts sub-pixel
    # accuracy -- standard practice for phase correlation.
    window = np.outer(np.hanning(h), np.hanning(w))
    window = np.where(window == 0, 1e-6, window)  # avoid a fully-zero window on tiny patches

    fa = np.fft.fft2((a - a.mean()) * window)
    fb = np.fft.fft2((b - b.mean()) * window)
    cross_power = fa * np.conj(fb)
    denom = np.abs(cross_power) + 1e-6
    r = np.abs(np.fft.ifft2(cross_power / denom))

    py, px = np.unravel_index(np.argmax(r), r.shape)

    def parabolic_offset(f_minus: float, f_center: float, f_plus: float) -> float:
        denom = f_minus - 2.0 * f_center + f_plus
        if abs(denom) < 1e-9:
            return 0.0
        return 0.5 * (f_minus - f_plus) / denom

    dy = parabolic_offset(r[(py - 1) % h, px], r[py, px], r[(py + 1) % h, px])
    dx = parabolic_offset(r[py, (px - 1) % w], r[py, px], r[py, (px + 1) % w])

    shift_y = py + dy
    shift_x = px + dx
    if shift_y > h / 2:
        shift_y -= h
    if shift_x > w / 2:
        shift_x -= w

    return float(shift_y), float(shift_x)


def chromatic_aberration_check(image_path: str | Path, patch_size: int = 64) -> dict:
    """
    Estimates R-vs-G and B-vs-G sub-pixel shift in edge-rich patches across
    the image via phase correlation, then checks whether that shift grows
    with distance from the image center the way real lens dispersion would.

    Earlier attempt used whole-pixel-only phase correlation and was
    non-discriminative (see _phase_correlation_shift docstring) -- this
    version adds parabolic sub-pixel refinement, which is where real
    chromatic aberration actually shows up.
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    h, w, _ = arr.shape

    def grad_mag(channel: np.ndarray) -> np.ndarray:
        gy, gx = np.gradient(channel)
        return np.sqrt(gx ** 2 + gy ** 2)

    r_edges = grad_mag(arr[:, :, 0])
    g_edges = grad_mag(arr[:, :, 1])
    b_edges = grad_mag(arr[:, :, 2])

    rows = h // patch_size
    cols = w // patch_size

    if rows < 3 or cols < 3:
        return {
            "score": 0.0,
            "abstain": True,
            "detail": {"reason": "image too small for chromatic aberration analysis"},
        }

    cy, cx = h / 2.0, w / 2.0
    max_dist = float(np.hypot(cy, cx))

    # Only bother with patches that actually contain meaningful edge content --
    # a flat patch has no edges to measure fringing on.
    edge_threshold = float(np.percentile(g_edges, 80))

    shift_mags = []
    norm_dists = []

    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * patch_size, c * patch_size
            g_patch = g_edges[y0:y0 + patch_size, x0:x0 + patch_size]

            if g_patch.max() < edge_threshold:
                continue

            r_patch = r_edges[y0:y0 + patch_size, x0:x0 + patch_size]
            b_patch = b_edges[y0:y0 + patch_size, x0:x0 + patch_size]

            shift_rg = _phase_correlation_shift(r_patch, g_patch)
            shift_bg = _phase_correlation_shift(b_patch, g_patch)
            shift_mag = (np.hypot(*shift_rg) + np.hypot(*shift_bg)) / 2.0

            patch_cy = y0 + patch_size / 2.0
            patch_cx = x0 + patch_size / 2.0
            dist = float(np.hypot(patch_cy - cy, patch_cx - cx)) / max_dist

            shift_mags.append(shift_mag)
            norm_dists.append(dist)

    if len(shift_mags) < 8:
        return {
            "score": 0.0,
            "abstain": True,
            "detail": {"reason": "not enough high-edge-content patches to measure aberration pattern"},
        }

    shift_arr = np.array(shift_mags)
    dist_arr = np.array(norm_dists)

    correlation = float(np.corrcoef(dist_arr, shift_arr)[0, 1])
    if np.isnan(correlation):
        correlation = 0.0

    # Real lenses: fringing should increase with distance from center, so
    # expect a positive correlation. Weak/no/negative correlation suggests
    # no real lens was involved -- calibration knob, tune against samples.
    if correlation >= 0.3:
        score = 5.0
    elif correlation <= -0.1:
        score = 85.0
    else:
        score = 85.0 - ((correlation + 0.1) / 0.4) * 80.0

    return {
        "score": round(score, 1),
        "detail": {
            "patches_used": len(shift_mags),
            "radial_shift_correlation": round(correlation, 3),
            "note": "Checks whether color-channel edge misalignment increases toward "
                    "the image edges/corners the way real lens dispersion would.",
        },
    }


# --------------------------------------------------------------------------
# EXPERIMENTAL -- line-art / flat-shading check (v2, gradient-bimodality)
# --------------------------------------------------------------------------
#
# NOT wired into analyze_image() yet -- call this directly to test it.
#
# v1 (counting distinct quantized colors) FAILED empirically: real flat/
# cel-shaded art (tested against actual Gundam/Gogeta samples) scored just
# as "smooth" as genuine AI images, because both have low pixel-level
# noise. Total color count doesn't capture the real distinguishing
# feature.
#
# What actually differs: traditional flat/cel-shaded art has a
# BIMODAL gradient-magnitude distribution -- large flat interior regions
# (near-zero gradient) plus sharp, deliberate line-art edges (very high
# gradient), with almost nothing in between. Continuous-tone content
# (photos AND AI-generated images alike) has a smoother, more evenly
# spread distribution of gradient magnitudes, without that gap in the
# middle.
#
# Same caveat as v1: this cannot distinguish AI-generated from real
# photos (both are continuous-tone). It can only speak to "does this
# look like traditional flat/line-art shading technique" -- useful mainly
# for correctly recognizing traditional illustration so it isn't
# misread as AI-generated by the noise-based signals.

def flat_shading_check(image_path: str | Path, bins: int = 50) -> dict:
    img = Image.open(image_path).convert("L")
    arr = np.asarray(img, dtype=np.float32)

    gy, gx = np.gradient(arr)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    max_grad = float(grad_mag.max())
    if max_grad < 1e-6:
        return {
            "score": 0.0,
            "abstain": True,
            "detail": {"reason": "image has no gradient variation to analyze"},
        }

    normalized = grad_mag / max_grad  # now in [0, 1]

    flat_fraction = float((normalized < 0.05).mean())
    edge_fraction = float((normalized > 0.4).mean())
    mid_fraction = float(((normalized >= 0.05) & (normalized <= 0.4)).mean())

    # High mid_fraction relative to the flat+edge extremes means a smooth,
    # continuously-varying gradient distribution (continuous-tone content).
    # Low mid_fraction with most pixels at the extremes (flat OR sharp edge)
    # means traditional flat/line-art shading.
    bimodality_ratio = mid_fraction / (flat_fraction + edge_fraction + 1e-6)

    # Calibration knob, tested against real samples: traditional flat/
    # cel-shaded art showed bimodality_ratio well under 1; continuous-tone
    # content showed it well over 1.
    score = min(bimodality_ratio * 40.0, 100.0)

    return {
        "score": round(score, 1),
        "detail": {
            "flat_fraction": round(flat_fraction, 3),
            "mid_fraction": round(mid_fraction, 3),
            "edge_fraction": round(edge_fraction, 3),
            "bimodality_ratio": round(bimodality_ratio, 3),
            "note": "EXPERIMENTAL. High score = smooth/continuous-tone gradient "
                    "distribution. Low score = flat-region-plus-sharp-edges "
                    "distribution typical of traditional line/cel-shaded art. "
                    "Does NOT distinguish AI art from real photos.",
        },
    }


# --------------------------------------------------------------------------
# Signal 8: Camera EXIF completeness (Pillow ExifTags)
# --------------------------------------------------------------------------
#
# Real cameras (phones, DSLRs, mirrorless) write a rich set of EXIF fields
# that no renderer or AI generator has any reason to produce: lens model,
# f-number, exposure time, ISO/photographic sensitivity, focal length,
# and often GPS. Finding several of these together is strong, specific
# evidence a real camera was involved -- much stronger than just "EXIF
# exists" (a generator could in principle write SOME metadata).
#
# IMPORTANT: absence is common and proves very little. EXIF is routinely
# stripped by messaging apps, social platforms, and browsers, so most
# real photos you'll encounter online have already lost this data. This
# signal should fire (and fire strongly) when present, but ABSTAIN
# entirely when absent, rather than treating "no EXIF" as evidence of
# anything.
#
# Same honest caveat as metadata_check: this does NOT distinguish AI art
# from human-made digital art. Neither involves a camera, so neither will
# ever have these fields -- this only strengthens the camera-vs-not
# distinction where it already works.

_CAMERA_EXIF_TAGS = [
    "Make", "Model", "LensModel", "FNumber", "ExposureTime",
    "ISOSpeedRatings", "PhotographicSensitivity", "FocalLength",
    "DateTimeOriginal", "GPSInfo", "ShutterSpeedValue", "ApertureValue",
]


def camera_exif_check(image_path: str | Path) -> dict:
    img = Image.open(image_path)
    exif = getattr(img, "_getexif", lambda: None)()

    if not exif:
        return {
            "score": 0.0,
            "abstain": True,
            "detail": {"reason": "no EXIF data present -- common even for real photos "
                                  "(routinely stripped by messaging apps/social platforms), "
                                  "so this proves nothing either way"},
        }

    found_tags = []
    for tag_id, value in exif.items():
        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
        if tag_name in _CAMERA_EXIF_TAGS and value:
            found_tags.append(tag_name)

    # Require several distinct camera-specific fields together -- a single
    # stray tag is weaker evidence than a genuine cluster of them.
    has_evidence = len(found_tags) >= 3
    # BUG FIX: finding strong camera evidence means this is MORE likely a
    # real photo, so it should push the suspicion score DOWN, not up. This
    # was previously inverted (scored 90 when evidence was found), which
    # meant every real photo with intact EXIF was being scored as if it
    # were strong evidence of AI generation -- exactly backwards.
    score = 3.0 if has_evidence else 0.0

    return {
        "score": score,
        "abstain": not has_evidence,
        "detail": {
            "camera_exif_tags_found": found_tags,
            "note": "Requires 3+ distinct camera-specific EXIF fields together as "
                    "genuine evidence. Absence abstains rather than counting against "
                    "authenticity, since EXIF stripping is extremely common.",
        },
    }


def analyze_image(image_path: str | Path) -> dict:
    return {
        "metadata_check": metadata_check(image_path),
        "camera_exif_check": camera_exif_check(image_path),
        "fft_artifact_check": fft_artifact_check(image_path),
        "error_level_analysis": error_level_analysis(image_path),
        "noise_consistency_check": noise_consistency_check(image_path),
        "noise_floor_check": noise_floor_check(image_path),
        "signal_dependent_noise_check": signal_dependent_noise_check(image_path),
        "chromatic_aberration_check": chromatic_aberration_check(image_path),
    }
