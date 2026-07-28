"""
model.py

Model definition for the trained AI-image classifier.

Uses transfer learning: a pretrained ResNet50 backbone (trained on ImageNet)
with its final classification layer replaced for our binary task
(real vs. AI-generated). This is the standard, well-documented approach for
building an image classifier without needing millions of your own images or
massive compute -- the backbone already knows how to extract useful visual
features; we're just teaching the last layer to use those features for a
new question.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_model(pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    """
    Builds a ResNet50-based binary classifier.

    Args:
        pretrained: whether to load ImageNet-pretrained weights. Set False
            only for architecture testing without internet access -- for
            real training you want this True.
        freeze_backbone: if True, only the final classifier layer is
            trained (fast, needs less data, good starting point). Set
            False later to fine-tune the whole network for better accuracy
            once you have a working baseline and want to push performance.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully-connected layer for binary classification
    # (2 classes: real, ai). Newly-created layers require grad by default,
    # so this trains even when the rest of the backbone is frozen.
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),
    )

    return model


def get_trainable_params(model: nn.Module):
    """Returns only the parameters that require gradients -- pass this to
    your optimizer so frozen backbone weights aren't wastefully tracked."""
    return [p for p in model.parameters() if p.requires_grad]


if __name__ == "__main__":
    # Quick architecture sanity check -- run with pretrained=False here
    # since this environment can't reach download.pytorch.org. Run with
    # pretrained=True (the default) wherever you actually train.
    m = build_model(pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = m(dummy_input)
    print("Output shape:", output.shape)  # expect (2, 2)
    print("Trainable params:", sum(p.numel() for p in get_trainable_params(m)))
