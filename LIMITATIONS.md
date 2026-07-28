# slopcheck -- Known Limitations & Findings

This documents real limitations discovered through testing, not assumed
ones. Each was found by running actual samples through the tool, not
theorized in advance.

## 1. Image detection tests "was a camera involved", not "AI vs. human"

**Finding:** running a folder of human-made digital art (video game key art,
character renders, fan art -- no camera involved, no AI involved either)
through the image detector scored it almost identically to genuine
AI-generated images (scores in the 75-99 range for both groups).

**Why:** the strongest image signals (`noise_floor_check`,
`signal_dependent_noise_check`, and to a lesser extent chromatic
aberration) all test for the physical signature of a real camera sensor
and lens -- shot noise, sensor grain, lens dispersion. A skilled human
digital illustrator's work has exactly as little of this as an AI
generator's output does, because neither involved a camera.

**Practical implication:** slopcheck's image side should be described as
detecting "was this born from a camera or a generator/renderer", not
"was this made by AI specifically". Distinguishing AI-generated content
from human-made digital art/renders by pixel statistics alone was found
to be unsolvable (see section 7) -- resolved instead with a purpose-
trained model (section 4).

**What we did about it:** calibration_set/images/ supports a third
`human_art/` category, scored for reference but excluded from the
real-vs-AI separation statistics, so this confound doesn't quietly
corrupt threshold tuning.

## 2. Localized photo compositing is only weakly detectable

**Finding:** a real photo with a large, cleanly-rendered character
composited in and flattened into a single fresh JPEG defeated both
Error Level Analysis and regional noise-consistency checks. Chromatic
aberration showed a promising signal on this specific case but is not
yet validated across enough samples to trust on its own.

**Why:** ELA needs a compression-history mismatch (double JPEG
compression) to detect anything -- a single-pass flatten-and-save
workflow never creates one. Regional noise-consistency needs the edited
region to be small enough that its neighbors are genuine background --
a large edit's interior only borders more of itself.

**Status:** partially resolved -- see section 4's tampering-axis finding,
which combines these signals with a Grad-CAM concentration signal in the
ML pipeline and successfully separated a known composite (23.6% AI /
72.2% tampering) from a clean photo (20.7% tampering).

## 3. Sophisticated AI generators can partially defeat noise-based signals

**Finding:** at least one tested AI-generated photorealistic image showed
a believable positive correlation between brightness and noise
(mimicking real shot-noise physics), partially defeating
`signal_dependent_noise_check`, which was otherwise a strong performer.

**Implication:** heuristic detection is not a solved, static problem --
generators can and do simulate the exact physical properties a detector
checks for. This is worth treating as an ongoing arms race in any
writeup, not a one-time fix.

## 4. Trained ML classifier (ml/): a genuinely different tool, with its own findings

A separate, complementary approach lives in `ml/` -- trained ResNet50
classifiers (transfer learning), rather than hand-built heuristics. This
is explicitly NOT in keeping with the rest of the project's "no
black-box ML" philosophy -- it was built deliberately as a second,
different kind of tool once heuristics hit genuine, well-evidenced walls
(sections 1-3, 7). See `ml/README.md` for the full rationale and
tradeoffs. Two separate models were trained for two separate tasks --
mixing them in one checkpoint folder caused a real accidental overwrite
during development (recovered by retraining from the still-intact
source dataset; lesson: always use separate `--checkpoint_dir` per task).

### 4a. Photo detector (Community Forensics-trained)

**Headline result:** ~81-97% val_acc depending on run, but only ~69-84%
on the hand-picked, adversarial calibration set -- a real, expected gap
between in-distribution and out-of-distribution performance, not a bug.

**Finding: forcing a binary real/AI verdict hides a real third category.**
Several recurring misclassifications turned out to be real photos that
were heavily EDITED (not AI-generated) -- e.g. professional product
photography with background cleanup and retouching. The binary
classifier had no way to express "this is real, but tampered with" --
it just picked whichever side was statistically closer. Fixed by adding
a SECOND, separate score (`ml/analyze_full.py`) that combines the
existing tampering-focused heuristics (ELA, regional noise consistency,
chromatic aberration) with a new Grad-CAM-based signal (whether the
model's attention is spatially concentrated -- suggestive of a localized
edit -- or spread across the whole image). Confirmed on a known composite
image: AI-generation dropped to 23.6% (correctly "not AI") while
tampering correctly read 72.2%.

**Finding: the CNN has no access to metadata, which sometimes matters
more than pixels.** A genuine DSLR photo (full EXIF: GPS, aperture,
shutter speed, ISO) was confidently misclassified as AI-generated by
the CNN -- almost certainly the same "unnaturally low sensor noise from
studio product photography" blind spot that broke the ORIGINAL heuristic
`noise_floor_check` (section 1), except the CNN has no equivalent
defense, since it only ever sees pixels, never metadata. Fixed by
adjusting the CNN's score with `camera_exif_check` and `metadata_check`
(hard-to-fake evidence: a consistent GPS+aperture+shutter+ISO cluster,
or embedded AI-generation parameters) when either fires. Confirmed:
87.3% (CNN alone, wrong) -> 69.0% (adjusted) on the known-real DSLR
photo -- a genuine, meaningful correction, though not enough to fully
flip the verdict given how confidently wrong the CNN's own read was.
That's an honest partial result, not a forced win.

**Bug caught and fixed along the way:** an early version of the metadata
adjustment ran the CNN's already-calibrated probability through the same
log-odds-plus-damping transform built for heuristic 0-100 "suspicion
scores". That's a category error -- a heuristic's 100 means "strongest
evidence this test can give", not "100% certain", but the CNN's softmax
output genuinely is a calibrated probability. Running it through the
heuristic transform regressed extreme scores toward 50 even when NO
metadata evidence was present (a raw 0.0% became 35.4% after "combining"
with two abstaining signals) -- caught by testing the no-evidence case
explicitly before trusting the fix. Corrected with the standard logit
transform, verified to be an exact no-op when metadata is absent.

**Still unexplained:** one recurring misclassification (a stock photo,
metadata stripped, no tampering signature, no localized Grad-CAM
concentration) remains wrong for reasons neither the tampering axis nor
the metadata axis explains. Documented as open rather than papered over.

### 4b. Art detector (AI-ArtBench-trained) -- resolves section 7

**This is the resolution to section 7's open question.** Pure heuristics
(tried twice, both failed -- see section 7) cannot distinguish AI-
generated art from human-made art, because a good AI illustration
generator reproduces the same low-level pixel statistics as hand-drawn
art. That conclusion still holds for pixel-statistics-only methods.

But a model trained specifically on the RIGHT task -- AI-ArtBench (human
art from ArtBench-10 vs. AI art generated via Latent/Standard Diffusion,
using prompts based on the same human artworks so style/subject are
controlled for) -- succeeded where both heuristic attempts and even the
photo-trained CNN failed. Tested against the project's own human_art
calibration images (the exact Gundam/Gogeta samples that broke every
prior attempt, including the photo-detection CNN, which called them
81-99% "AI-generated", wrong): the ArtBench-trained model correctly
classified 7 of 8 (87.5%) as "likely real", including the official
Gundam render (0.1%), both Gogeta images (16.9%, 0.0%), and the game key
art (0.1%).

**The honest lesson:** "this problem can't be solved" and "this problem
can't be solved by THIS KIND of tool" are different claims. The
heuristic investigation correctly proved the former was false and the
latter was true -- pixel statistics alone were never going to crack
style/content-based classification, since that's fundamentally a
semantic recognition task. Recognizing that boundary precisely, rather
than either giving up or force-fitting a heuristic that doesn't work,
is what made it possible to reach for the right tool (a trained
classifier on a purpose-built, style-controlled dataset) once the
scope had shifted from "pure heuristics only" to "heuristics plus a
separate ML tool for tasks heuristics structurally can't do".

**One remaining miss, not yet investigated:** a meme-style repost (casual
filename suggesting social-media reposting/compression/possible caption
overlay) scored 88.8% AI, incorrectly. Plausibly a distribution-shift
case (heavily reprocessed/captioned image, different from ArtBench's
relatively clean training distribution) rather than a fundamental flaw --
worth checking specifically rather than generalizing from n=1.

## 5. Combination logic: log-odds addition, with an asymmetric fix

Signal scores are combined via log-odds addition (not a weighted
average), because averaging let a single strongly-firing signal get
diluted by several signals that simply didn't apply to a given image.
An early symmetric version of this had its own bug: treating a low
score (no evidence found) as equally strong evidence of authenticity as
a high score is evidence of fakery let several "found nothing" signals
outvote one "found something" signal. Fixed by damping low-score
(real-leaning) contributions relative to high-score (AI-leaning) ones,
since most signals are narrow tests where absence-of-evidence is weaker
than presence-of-evidence.

## 6. Text detection: validated across 19 samples, clean separation

As of the last calibration run: 10 human samples (mean 19.6, range
9.0-38.8) vs. 9 AI samples (mean 97.2, range 89.5-99.6) -- zero overlap.
This is the best-validated part of the heuristic project. Worth
periodically re-running `calibrate.py text` as more samples are added.

## 7. Content/style-based AI-art detection with PURE HEURISTICS: tried twice, failed both times

**Attempt 1 (color quantization):** hypothesized that traditional flat/
cel-shaded illustration uses fewer distinct colors than AI-generated
content. Tested against real samples (official Gundam art, Dragon Ball
Gogeta art) vs. AI images -- both groups scored in the same low range.
Turned out to be redundant with noise_floor_check (both measure the same
underlying smoothness), not a new axis of information.

**Attempt 2 (gradient-magnitude bimodality):** hypothesized that flat/
cel-shaded art has a bimodal gradient distribution (flat interiors + sharp
line-art edges) while continuous-tone content doesn't. This DID correctly
separate the two human-art samples from real photos -- but an AI-generated
stylized illustration (dramatic clouds with hard-edged lighting contrast)
scored just as "bimodal" as the human art, because modern AI illustration
generators are trained on exactly this kind of art and reproduce its
low-level statistical signature convincingly.

**Conclusion (still valid for pure heuristics):** distinguishing AI-
generated art from human-made art by content/style is a fundamentally
different problem from "was a camera involved", and pure statistical
heuristics don't survive contact with a generator that was trained on
the same style it's being compared against. This would need semantic/
stylistic pattern recognition -- i.e. a trained model -- which is out of
scope by design for the heuristic-only core project.

**UPDATE -- resolved with the right tool, see section 4b.** Once the
project's scope explicitly expanded to include a separate ML component
(itself a deliberate, documented departure from "no black-box ML"), this
stopped being an open problem: a model trained on AI-ArtBench correctly
classified 7/8 of the exact human_art calibration images that defeated
every heuristic attempt. The heuristic conclusion above wasn't wrong --
it correctly identified that THIS KIND of tool couldn't solve THIS KIND
of problem. It just wasn't the end of the story once a different kind
of tool was deliberately brought in.

**What still works for AI-art specifically, heuristically:** the
metadata expansion (see image_signals.py metadata_check) catches AI art
specifically when generation parameters haven't been stripped from the
file (common with local Stable Diffusion/ComfyUI setups) -- a real, if
partial, heuristic win for this exact problem, independent of the ML
resolution above.
