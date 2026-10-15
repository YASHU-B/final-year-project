"""Evaluate one saved model checkpoint on the untouched HAM10000 test split.

Example:
    python -m src.evaluate \
      --config configs/baseline_efficientnet_b0.yaml \
      --checkpoint checkpoints/baseline_efficientnet_b0_cross_entropy/best_model.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constants import CLASS_NAMES
from .dataset import HAM10000Dataset
from .model import build_efficientnet_b0_cbam_classifier, build_efficientnet_b0_classifier
from .utils import resolve_device, save_json


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def compute_specificity(matrix: np.ndarray) -> np.ndarray:
    """Compute one-vs-rest true-negative rate for every class."""
    total = matrix.sum()
    values = []
    for class_index in range(len(matrix)):
        true_negative = total - matrix[class_index, :].sum() - matrix[:, class_index].sum() + matrix[class_index, class_index]
        false_positive = matrix[:, class_index].sum() - matrix[class_index, class_index]
        denominator = true_negative + false_positive
        values.append(float(true_negative / denominator) if denominator else float("nan"))
    return np.asarray(values)


def save_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axis,
    )
    axis.set(xlabel="Predicted class", ylabel="True class", title="Test-set confusion matrix")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_roc_curves(labels: np.ndarray, probabilities: np.ndarray, output_path: Path) -> dict[str, float]:
    """Save one-vs-rest ROC curves and return class-wise/macro AUC values."""
    one_hot_labels = np.eye(len(CLASS_NAMES))[labels]
    auc_scores: dict[str, float] = {}
    figure, axis = plt.subplots(figsize=(8, 6))
    for class_index, class_name in enumerate(CLASS_NAMES):
        if one_hot_labels[:, class_index].min() == one_hot_labels[:, class_index].max():
            auc_scores[class_name] = float("nan")
            continue
        false_positive_rate, true_positive_rate, _ = roc_curve(
            one_hot_labels[:, class_index], probabilities[:, class_index]
        )
        auc_value = roc_auc_score(one_hot_labels[:, class_index], probabilities[:, class_index])
        auc_scores[class_name] = float(auc_value)
        axis.plot(false_positive_rate, true_positive_rate, label=f"{class_name} (AUC={auc_value:.3f})")
    axis.plot([0, 1], [0, 1], "k--", linewidth=1)
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title="One-vs-rest ROC curves")
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    auc_scores["macro"] = float(roc_auc_score(one_hot_labels, probabilities, multi_class="ovr", average="macro"))
    return auc_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device()
    data_config = config["data"]
    test_dataset = HAM10000Dataset(
        Path(data_config["split_dir"]) / "test.csv",
        data_config["raw_data_dir"],
        training=False,
        image_size=data_config["image_size"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    if config["model"].get("use_cbam", False):
        model = build_efficientnet_b0_cbam_classifier(
            num_classes=config["model"]["num_classes"],
            pretrained=False,
            reduction_ratio=config["model"].get("cbam_reduction_ratio", 16),
        ).to(device)
    else:
        model = build_efficientnet_b0_classifier(
            num_classes=config["model"]["num_classes"], pretrained=False
        ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[np.ndarray] = []
    all_image_ids: list[str] = []
    with torch.no_grad():
        for images, labels, image_ids in tqdm(test_loader, desc="Evaluate"):
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)
            all_labels.extend(labels.tolist())
            all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
            all_probabilities.append(probabilities.cpu().numpy())
            all_image_ids.extend(image_ids)

    labels = np.asarray(all_labels)
    predictions = np.asarray(all_predictions)
    probabilities = np.concatenate(all_probabilities)
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(CLASS_NAMES))))
    precision, recall, f1_score, support = precision_recall_fscore_support(
        labels, predictions, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )
    specificity = compute_specificity(matrix)
    report = classification_report(labels, predictions, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    experiment_dir = Path("results") / config["experiment_name"]
    experiment_dir.mkdir(parents=True, exist_ok=True)
    class_metrics = pd.DataFrame(
        {
            "class": CLASS_NAMES,
            "support": support,
            "precision": precision,
            "recall_sensitivity": recall,
            "specificity": specificity,
            "f1_score": f1_score,
        }
    )
    class_metrics.to_csv(experiment_dir / "test_class_metrics.csv", index=False)
    save_confusion_matrix(matrix, experiment_dir / "test_confusion_matrix.png")
    auc_scores = save_roc_curves(labels, probabilities, experiment_dir / "test_roc_curves.png")
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall_sensitivity": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "macro_roc_auc_ovr": auc_scores["macro"],
        "per_class_roc_auc_ovr": {name: value for name, value in auc_scores.items() if name != "macro"},
    }
    save_json(summary, experiment_dir / "test_summary.json")
    pd.DataFrame(
        {
            "image_id": all_image_ids,
            "true_class": [CLASS_NAMES[index] for index in labels],
            "predicted_class": [CLASS_NAMES[index] for index in predictions],
            "confidence": probabilities.max(axis=1),
        }
    ).to_csv(experiment_dir / "test_predictions.csv", index=False)
    print(json.dumps(summary, indent=2))
    print(f"Saved test metrics and figures to: {experiment_dir}")


if __name__ == "__main__":
    main()
