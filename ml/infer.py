"""
infer.py

Run the trained classifier on a single image.

Usage:
    python infer.py --checkpoint checkpoints/best_model.pt --image path/to/image.jpg
    python infer.py --checkpoint checkpoints/best_model.pt --image path/to/image.jpg --heatmap out.png
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from PIL import Image

from dataset import get_eval_transforms
from gradcam import GradCAM, overlay_heatmap
from model import build_model


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(pretrained=False)  # weights come from the checkpoint, not ImageNet
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint["classes"]


def predict(model, classes, image_path: str, device: torch.device):
    img = Image.open(image_path).convert("RGB")
    transform = get_eval_transforms()
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]

    ai_prob = probs[classes.index("ai")].item()
    real_prob = probs[classes.index("real")].item()

    return {
        "ai_probability": round(ai_prob * 100, 1),
        "real_probability": round(real_prob * 100, 1),
        "verdict": "likely AI-generated" if ai_prob > 0.5 else "likely real",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--heatmap", default=None, help="Optional path to save a Grad-CAM heatmap overlay")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(args.checkpoint, device)

    result = predict(model, classes, args.image, device)
    print(f"\nAI probability:   {result['ai_probability']}%")
    print(f"Real probability:  {result['real_probability']}%")
    print(f"Verdict: {result['verdict']}\n")

    if args.heatmap:
        img = Image.open(args.image).convert("RGB")
        transform = get_eval_transforms()
        input_tensor = transform(img).unsqueeze(0).to(device)

        # target_layer: last conv block of ResNet50 -- standard choice for Grad-CAM
        cam = GradCAM(model, target_layer=model.layer4[-1])
        heatmap, class_idx = cam.generate(input_tensor)
        overlay = overlay_heatmap(img, heatmap)
        overlay.save(args.heatmap)
        print(f"Heatmap saved to {args.heatmap} (highlighting regions that most "
              f"influenced the '{classes[class_idx]}' prediction)")


if __name__ == "__main__":
    main()
