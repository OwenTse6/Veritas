"""
gradcam.py

Grad-CAM (Gradient-weighted Class Activation Mapping) for the trained
classifier -- this is how we keep some explainability even though this is
a trained neural network rather than a hand-built heuristic. It produces
a heatmap showing which regions of the image most influenced the model's
decision, similar to what commercial tools like TruthScan show their
users ("activation maps" / heatmap overlays).

This does NOT tell you *why* a region looks suspicious (unlike our
heuristic signals, which always have a stated reason) -- it only tells
you *where* the model was looking. That's a meaningfully weaker kind of
explainability, and worth being upfront about with anyone using this.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None):
        """
        Args:
            input_tensor: preprocessed image tensor, shape (1, 3, H, W)
            class_idx: which class to explain (default: the model's own
                top prediction)
        Returns:
            heatmap as a (H, W) numpy array, normalized to [0, 1]
        """
        self.model.eval()

        # IMPORTANT: with a frozen backbone (requires_grad=False on backbone
        # params), autograd won't build a gradient path through intermediate
        # layers unless SOMETHING upstream requires grad -- otherwise it
        # correctly determines there's nothing to compute and skips it
        # entirely, so our backward hook never fires (gradients stay None).
        # Explicitly requiring grad on the input forces that path to exist.
        input_tensor = input_tensor.clone().requires_grad_(True)

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Grad-CAM hooks did not fire -- check that target_layer is "
                "actually part of the model's forward pass."
            )

        gradients = self.gradients[0]        # (C, h, w)
        activations = self.activations[0]     # (C, h, w)

        weights = gradients.mean(dim=(1, 2))  # (C,)
        # IMPORTANT: must create this on the SAME device as activations
        # (cuda if the model is on GPU) -- this defaulted to CPU before,
        # which only broke on an actual CUDA setup, not in CPU-only testing.
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), class_idx


def overlay_heatmap(original_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Resizes the heatmap to match the original image and blends them
    into a single visual (red/warm = high suspicion contribution)."""
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        original_image.size, resample=Image.BILINEAR
    )
    heatmap_arr = np.array(heatmap_img)

    # Simple red-heat colormap without needing matplotlib as a dependency.
    colored = np.zeros((*heatmap_arr.shape, 3), dtype=np.uint8)
    colored[..., 0] = 255                    # red channel full
    colored[..., 1] = 255 - heatmap_arr       # green fades as suspicion rises
    colored[..., 2] = 0

    original_arr = np.array(original_image.convert("RGB"))
    blended = (original_arr * (1 - alpha) + colored * alpha).astype(np.uint8)
    return Image.fromarray(blended)
