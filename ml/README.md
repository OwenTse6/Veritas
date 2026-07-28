# slopcheck-ml

A trained neural network classifier for AI-generated image detection --
a separate, complementary approach from the heuristic signals in the main
`slopcheck` project.

## Why this is separate from the main project

The rest of slopcheck is built around transparent, explainable statistical
heuristics -- no black-box ML, by design. This folder is the opposite: a
trained ResNet50 classifier, the same general approach commercial tools
like TruthScan use. We built it because heuristics genuinely hit a wall
(see the main project's `LIMITATIONS.md`) that a trained model can likely
do better on -- but it comes with different tradeoffs: it needs training
data, GPU time, and its explanations (via Grad-CAM heatmaps) tell you
*where* it looked, not *why* that area seemed suspicious, unlike the
heuristic signals which always state a specific reason.

## Requirements

- A GPU is strongly recommended. CPU training will work but is very slow.
- ~10-50GB disk space depending on which dataset you use.

```bash
pip install -r requirements.txt
```

## Getting a dataset

You'll need real and AI-generated images sorted into this structure:

```
data/
    train/
        ai/
            img001.jpg
            ...
        real/
            img001.jpg
            ...
    val/
        ai/
        real/
```

**Note the folder names matter**: `ai` and `real`, lowercase, exactly as
shown -- the code checks for this and will error clearly if it doesn't
match.

Recommended datasets to start with (roughly easiest to most comprehensive):

- **CIFAKE** -- small, low-res (32x32), good for a first correctness
  check of the whole pipeline before committing to a bigger download.
- **Artifact** (Rahman et al.) -- ~2.5M images across 25 generators.
- **SynthBuster** + **RAISE-1k** -- ~9k synthetic images across 9
  diffusion models, paired with a real-photo dataset.
- **Community Forensics** (Park & Owens) -- 2.4M real + 2.4M AI images
  across 4,803 different generative models -- the most comprehensive,
  best choice if you want a model that generalizes well.

Search for these by name on Kaggle or Hugging Face Datasets -- exact
download steps vary by dataset and platform account setup, so follow
their own instructions for pulling the files down, then reorganize into
the folder structure above (a short reorganization script per-dataset is
normal here, since none of them ship in exactly this layout).

## Training

Start with the frozen-backbone baseline (fast, good sanity check):

```bash
python train.py --data_root ./data --epochs 10 --batch_size 32
```

Once that's working and you want to push accuracy further, fine-tune the
whole network instead of just the classifier head:

```bash
python train.py --data_root ./data --epochs 10 --batch_size 16 --unfreeze
```

(Smaller batch size recommended when unfreezing -- fine-tuning the full
network uses much more GPU memory.)

## Running inference

```bash
python infer.py --checkpoint checkpoints/best_model.pt --image somefile.jpg
python infer.py --checkpoint checkpoints/best_model.pt --image somefile.jpg --heatmap out.png
```

The `--heatmap` flag saves a Grad-CAM overlay showing which regions of the
image most influenced the verdict.

## Honest expectations

Published baselines for this exact task (ResNet50, similar-scale data)
land around 80% accuracy on binary real-vs-AI classification -- not the
99%+ some commercial tools advertise. Expect a real, useful improvement
over the heuristic approach, especially on the reprocessed/recompressed
images that broke the heuristics -- but not a perfect solution. This
remains a genuine arms race against improving generators, for trained
models as much as for heuristics.
