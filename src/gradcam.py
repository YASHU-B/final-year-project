"""Grad-CAM implementation and visualization for skin lesion interpretability.

Supports standard EfficientNet-B0 and EfficientNet-B0 + CBAM architectures.
Produces publication-quality visual heatmaps overlaying dermoscopic images.

Example:
    python -m src.gradcam \
      --config configs/baseline_efficientnet_b0.yaml \
      --checkpoint checkpoints/baseline_efficientnet_b0_cross_entropy/best_model.pt \
      --num-samples 7 \
      --output-dir figures/gradcam
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .constants import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD
from .model import build_efficientnet_b0_cbam_classifier, build_efficientnet_b0_classifier
from .preprocessing import build_evaluation_transform
from .utils import resolve_device


class GradCAM:
    """Computes Grad-CAM heatmaps for a specified convolutional target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self.hooks: list[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(module: nn.Module, grad_input: Any, grad_output: tuple[torch.Tensor, ...]) -> None:
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """Compute Grad-CAM heatmap for a single image tensor of shape (1, 3, H, W)."""
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = int(logits.argmax(dim=1).item())
        confidence = float(probabilities[0, predicted_class].item())

        if target_class is None:
            target_class = predicted_class

        target_score = logits[0, target_class]
        target_score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture activations or gradients for Grad-CAM.")

        # Global average pool the gradients along spatial dimensions (C, H, W) -> (C, 1, 1)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        # Apply ReLU to focus on features that have a positive influence
        cam = F.relu(cam)

        # Upsample to match input tensor resolution
        cam = F.interpolate(
            cam,
            size=(input_tensor.size(2), input_tensor.size(3)),
            mode="bilinear",
            align_corners=False,
        )

        cam_numpy = cam.squeeze().cpu().numpy()
        # Normalize to [0, 1]
        cam_min, cam_max = cam_numpy.min(), cam_numpy.max()
        if cam_max - cam_min > 1e-8:
            cam_numpy = (cam_numpy - cam_min) / (cam_max - cam_min)
        else:
            cam_numpy = np.zeros_like(cam_numpy)

        return cam_numpy, predicted_class, confidence

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()


def overlay_cam_on_image(
    original_image: Image.Image,
    cam_mask: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet",
) -> np.ndarray:
    """Overlay Grad-CAM heatmap on top of a PIL dermoscopic image."""
    rgb_image = original_image.convert("RGB")
    width, height = rgb_image.size

    # Resize CAM mask to original image size using PIL bilinear interpolation
    cam_pil = Image.fromarray((cam_mask * 255).astype(np.uint8)).resize((width, height), resample=Image.BILINEAR)
    cam_norm = np.array(cam_pil, dtype=np.float32) / 255.0

    colormap = plt.colormaps[colormap_name]
    colored_heatmap = colormap(cam_norm)[:, :, :3]  # (H, W, 3) in float [0, 1]
    colored_heatmap_uint8 = (colored_heatmap * 255).astype(np.uint8)

    img_array = np.array(rgb_image, dtype=np.float32)
    blended = (alpha * colored_heatmap_uint8 + (1.0 - alpha) * img_array).astype(np.uint8)
    return blended


def get_target_layer(model: nn.Module) -> nn.Module:
    """Identify the optimal final feature layer for Grad-CAM."""
    if hasattr(model, "features"):
        return model.features[-1]
    if hasattr(model, "cbam"):
        return model.cbam
    raise ValueError(f"Could not automatically resolve target conv layer for {type(model)}")


def load_model_from_checkpoint(config_path: Path, checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Instantiate and load model weights from a config and checkpoint."""
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    use_cbam = bool(config.get("model", {}).get("use_cbam", False))
    reduction_ratio = int(config.get("model", {}).get("cbam_reduction_ratio", 16))

    if use_cbam:
        model = build_efficientnet_b0_cbam_classifier(
            num_classes=7,
            pretrained=False,
            reduction_ratio=reduction_ratio,
        )
    else:
        model = build_efficientnet_b0_classifier(num_classes=7, pretrained=False)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def generate_gradcam_gallery(
    baseline_model: nn.Module | None,
    proposed_model: nn.Module | None,
    test_df: pd.DataFrame,
    raw_data_dir: Path,
    output_dir: Path,
    device: torch.device,
    samples_per_class: int = 1,
) -> None:
    """Generate side-by-side explainability figures for sample test lesions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = build_evaluation_transform(image_size=224)

    # Pick samples representing each class
    selected_rows = []
    for class_name in CLASS_NAMES:
        class_subset = test_df[test_df["dx"] == class_name]
        if not class_subset.empty:
            samples = class_subset.head(samples_per_class)
            selected_rows.append(samples)

    if not selected_rows:
        return

    gallery_df = pd.concat(selected_rows, ignore_index=True)

    baseline_cam = GradCAM(baseline_model, get_target_layer(baseline_model)) if baseline_model else None
    proposed_cam = GradCAM(proposed_model, get_target_layer(proposed_model)) if proposed_model else None

    cols = 1 + (1 if baseline_cam else 0) + (1 if proposed_cam else 0)
    figure, axes = plt.subplots(len(gallery_df), cols, figsize=(cols * 4, len(gallery_df) * 3.5), squeeze=False)

    for row_idx, (_, row) in enumerate(gallery_df.iterrows()):
        image_id = row["image_id"]
        true_dx = row["dx"]

        # Search for image file in raw_data_dir subfolders
        image_path = None
        for candidate in raw_data_dir.rglob(f"{image_id}.jpg"):
            image_path = candidate
            break

        if image_path is None or not image_path.exists():
            continue

        raw_pil = Image.open(image_path).convert("RGB")
        input_tensor = transform(raw_pil).unsqueeze(0).to(device)

        col_curr = 0
        # 1. Original Image
        axes[row_idx, col_curr].imshow(raw_pil)
        axes[row_idx, col_curr].set_title(f"Image: {image_id}\nGround Truth: {true_dx.upper()}", fontsize=10, weight="bold")
        axes[row_idx, col_curr].axis("off")
        col_curr += 1

        # 2. Baseline Grad-CAM
        if baseline_cam:
            mask, pred_idx, conf = baseline_cam.generate_cam(input_tensor)
            overlay = overlay_cam_on_image(raw_pil, mask)
            pred_name = CLASS_NAMES[pred_idx].upper()
            status_color = "green" if pred_name.lower() == true_dx.lower() else "red"
            axes[row_idx, col_curr].imshow(overlay)
            axes[row_idx, col_curr].set_title(
                f"Baseline EfficientNet-B0\nPred: {pred_name} ({conf:.1%})",
                fontsize=10,
                color=status_color,
            )
            axes[row_idx, col_curr].axis("off")
            col_curr += 1

        # 3. Proposed/CBAM Grad-CAM
        if proposed_cam:
            mask, pred_idx, conf = proposed_cam.generate_cam(input_tensor)
            overlay = overlay_cam_on_image(raw_pil, mask)
            pred_name = CLASS_NAMES[pred_idx].upper()
            status_color = "green" if pred_name.lower() == true_dx.lower() else "red"
            axes[row_idx, col_curr].imshow(overlay)
            axes[row_idx, col_curr].set_title(
                f"Proposed CBAM Model\nPred: {pred_name} ({conf:.1%})",
                fontsize=10,
                color=status_color,
            )
            axes[row_idx, col_curr].axis("off")

    plt.tight_layout()
    gallery_path = output_dir / "gradcam_comparison_gallery.png"
    plt.savefig(gallery_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Grad-CAM explainability comparison to: {gallery_path}")

    if baseline_cam:
        baseline_cam.remove_hooks()
    if proposed_cam:
        proposed_cam.remove_hooks()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explainability heatmaps.")
    parser.add_argument("--config-baseline", type=Path, default=Path("configs/baseline_efficientnet_b0.yaml"))
    parser.add_argument(
        "--checkpoint-baseline",
        type=Path,
        default=Path("checkpoints/baseline_efficientnet_b0_cross_entropy/best_model.pt"),
    )
    parser.add_argument("--config-cbam", type=Path, default=None)
    parser.add_argument("--checkpoint-cbam", type=Path, default=None)
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/gradcam"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/splits/test.csv"))
    parser.add_argument("--raw-data-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    device = resolve_device()
    print(f"Using device for Grad-CAM: {device}")

    baseline_model = None
    if args.checkpoint_baseline and args.checkpoint_baseline.exists():
        print(f"Loading baseline model from {args.checkpoint_baseline}...")
        baseline_model = load_model_from_checkpoint(args.config_baseline, args.checkpoint_baseline, device)

    cbam_model = None
    if args.checkpoint_cbam and args.checkpoint_cbam.exists():
        print(f"Loading CBAM model from {args.checkpoint_cbam}...")
        cbam_model = load_model_from_checkpoint(args.config_cbam, args.checkpoint_cbam, device)

    test_df = pd.read_csv(args.test_csv)
    generate_gradcam_gallery(
        baseline_model=baseline_model,
        proposed_model=cbam_model,
        test_df=test_df,
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        device=device,
        samples_per_class=args.samples_per_class,
    )


if __name__ == "__main__":
    main()
