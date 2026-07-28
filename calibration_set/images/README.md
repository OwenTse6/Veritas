# Calibration images

This folder is intentionally excluded from git (see `.gitignore`) --
real-world calibration data tends to include stock photography, product
photos, and fan/official art whose licenses don't permit redistribution,
and potentially photos of identifiable people who never consented to
being part of a public dataset. Auditing every image's provenance
individually isn't practical, so the safer default is: keep these local,
don't push them.

## Expected structure

To rebuild a working calibration set locally, populate these folders
with your own images:

```
calibration_set/images/
    real/         -- genuine, unedited camera photographs
    ai/           -- known AI-generated images (any generator/style)
    human_art/    -- human-made digital art/illustration/renders --
                     NOT photographs, NOT AI-generated. Kept as a
                     separate category because it shares statistical
                     properties with AI images on some heuristic
                     signals (see LIMITATIONS.md section 1) -- mixing
                     it into real/ would corrupt threshold calibration.
```

Then run `python calibrate.py image` from the project root to see score
distributions and separation quality across your own set.
