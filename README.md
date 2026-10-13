# Veritas

A heuristic, **explainable** AI-generated content detector for text and images —
plus a separate, complementary trained-model detector for cases the
heuristics can't reach (see below).

**Two genuinely different approaches live in this repo, and it matters
which one you're using:**

- **`core/` — no black-box classifiers.** Every score comes from named,
  inspectable statistical signals (noise floor, burstiness, chromatic
  aberration, etc.) — you can always see exactly *why* something was
  flagged and exactly how confident you should be. This is the original
  engine and the project's main differentiator from typical AI detectors.
- **`ml/` — a trained neural network,** built deliberately once we proved
  (see `LIMITATIONS.md`) that some problems -- distinguishing AI-generated
  art from human-made art, in particular -- are fundamentally unsolvable
  by pixel statistics alone, no matter how many heuristics you throw at
  them. This is explicitly a black-box classifier (with partial
  Grad-CAM-based explainability layered on top, not the same kind of
  transparency as `core/`). See `ml/README.md` for the full rationale.

If you want the "no black box" guarantee, use `core/` -- `cli.py`'s
`text`/`image`/`batch` commands only ever call into it. If you want the
higher accuracy the trained models achieved (particularly for AI-art
detection, where they succeeded and the heuristics didn't), use `ml/`,
understanding you're trading transparency for performance.

## Why this exists

Most AI-content detectors are black boxes: you get a confidence percentage
and no way to check its reasoning. `core/` takes the opposite approach —
transparent, individually-tunable signals, combined with a documented,
principled method, and an honest account of what it can and can't do.
`ml/` exists alongside it as an honest acknowledgment that this approach
has real, documented limits (see `LIMITATIONS.md`) -- rather than
overclaiming that heuristics solved everything, the project's own testing
is what led to bringing in a different kind of tool for the parts they
couldn't reach.

## What it actually detects (validated)

- **AI-generated text** — burstiness, vocabulary fingerprinting, structural
  repetition, punctuation patterns, and unqualified-praise detection.
  Validated across 19 diverse human/AI samples with clean separation
  (human mean 19.6, AI mean 97.2, zero overlap).
- **Whether an image involved a real camera** — sensor noise floor,
  brightness-dependent shot-noise physics, and (experimentally) lens
  chromatic aberration. This is the strongest, best-tested part of the
  image side.
- **AI generation metadata**, when present — including raw Stable
  Diffusion/ComfyUI/Automatic1111 generation parameters, not just software
  name tags. Real, but only catches files where metadata hasn't been
  stripped (common on social platforms and messaging apps).
- **Real camera EXIF fields** (lens, ISO, aperture, GPS, etc.) — strong
  positive evidence when present; absence proves nothing, since EXIF is
  routinely stripped.
- **Some localized photo splicing** — via Error Level Analysis and regional
  noise-consistency checks. Works for some compositing workflows, not all
  (see Limitations).

## What `core/` (the heuristic engine) does NOT detect, and why

**Distinguishing AI-generated art from human-made digital art/illustration.**
We tried this seriously with pure heuristics — twice, with real test
samples, not just theory — and both attempts failed the moment a stylized
AI illustration was involved, because modern generators are trained on
exactly the art styles they're being compared against and reproduce their
statistical signature convincingly. Every heuristic signal in this tool
measures physical/statistical properties (camera noise, lens optics,
compression history) — none of them can tell *who* made a piece of art,
only *how* it was rendered.

**This conclusion about heuristics turned out to be correct, and also not
the end of the story.** A model trained specifically on this task (in
`ml/`, using the AI-ArtBench dataset) succeeded where the heuristics
couldn't — correctly classifying 7 of 8 known-hard art samples that had
defeated every prior attempt. See `LIMITATIONS.md` section 4b for the
full result. The honest lesson: "this can't be solved" and "this can't
be solved by *this kind* of tool" are different claims, and it's worth
knowing which one you've actually proven before concluding a problem is
unsolvable.

Full writeup of this and other findings in [`LIMITATIONS.md`](LIMITATIONS.md).

## Installation

```bash
git clone <this repo>
cd slopcheck
pip install -r requirements.txt
```

## Usage

**Check a text file:**
```bash
python cli.py text somefile.txt --explain
```

**Check an image:**
```bash
python cli.py image somephoto.jpg --explain
```

**Batch-scan a whole folder** (auto-detects text vs. image per file):
```bash
python cli.py batch some_folder/
```

**Run as an API:**
```bash
uvicorn api:app --reload
# POST /check/text   {"text": "..."}
# POST /check/image  multipart file upload
```

Add `--json` to any command for raw JSON output instead of a formatted table.

## How scoring works

Each signal returns a 0-100 "how suspicious is this" score plus a `detail`
dict explaining its reasoning. Signals are combined via **log-odds
addition** (not a weighted average) — evidence adds rather than dilutes,
so one strongly-firing signal isn't drowned out by several signals that
simply don't apply to a given file. Low scores are damped relative to high
scores, since most signals are narrow tests where "found nothing" is much
weaker evidence than "found something specific." See the top of
`core/scorer.py` for the full reasoning and history of how this was
arrived at (including a symmetric-combination bug we found and fixed).

## Calibration

`calibration_set/` holds labeled text and image samples for testing.

```bash
python calibrate.py text    # runs calibration_set/text/{human,ai}
python calibrate.py image   # runs calibration_set/images/{real,ai,human_art}
python calibrate.py both
```

This reports score distributions per group and flags overlap risk, so you
can see directly whether current thresholds hold up as you add more
samples. Contributions of labeled samples (especially real photos and
varied AI-generated images) are welcome and directly improve calibration.

## Project structure

```
slopcheck/
├── core/
│   ├── text_signals.py      # burstiness, vocab fingerprint, structural repetition, punctuation, evaluative language
│   ├── image_signals.py     # metadata, EXIF, FFT, ELA, noise consistency, noise floor, shot-noise, chromatic aberration
│   └── scorer.py            # log-odds combination + explanations
├── cli.py                    # command-line interface
├── api.py                    # FastAPI service
├── calibrate.py               # batch calibration/testing tool
├── calibration_set/           # labeled test samples (images excluded from git -- see calibration_set/images/README.md)
├── ml/                        # trained ML classifiers -- a separate, complementary approach (see ml/README.md)
├── LIMITATIONS.md              # documented findings, failures, and scope boundaries
└── data/ai_tells.txt           # curated AI phrase list (weighted)
```

## License

MIT for the code in this repository (see `LICENSE`). This does NOT cover
datasets downloaded via the scripts in `ml/` (several are non-commercial-
use only) or any images you place in `calibration_set/images/` locally --
see the note at the bottom of `LICENSE` and `ml/README.md`.

## Contributing

More calibration samples are the highest-value contribution right now —
especially real photos with intact EXIF, and AI-generated images from a
variety of generators/styles. See `LIMITATIONS.md` for open problems if
you want to attempt a fix for something documented there.
