"""Train a reproducible EfficientNet-B0 baseline from a YAML configuration.

Example:
    python -m src.train --config configs/baseline_efficientnet_b0.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
import pandas as pd
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import HAM10000Dataset
from .losses import FocalLoss, effective_number_class_weights
from .model import build_efficientnet_b0_cbam_classifier, build_efficientnet_b0_classifier
from .utils import plot_training_history, resolve_device, save_json, set_seed


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def make_loader(dataset: HAM10000Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def training_class_counts(split_csv: Path) -> list[int]:
    """Return training-only sample counts ordered by the project class indices."""
    class_names = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
    diagnoses = pd.read_csv(split_csv)["dx"]
    return [int((diagnoses == class_name).sum()) for class_name in class_names]


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
) -> tuple[float, float]:
    """Run one training or validation epoch and return mean loss and macro-F1."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []
    iterator = tqdm(loader, desc="Train" if is_training else "Validate", leave=False)

    for images, labels, _ in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = loss_function(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())

    mean_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
    return mean_loss, macro_f1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint path. Resumes at the epoch after its saved epoch.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])
    device = resolve_device()
    print(f"Using device: {device}")

    data_config = config["data"]
    split_dir = Path(data_config["split_dir"])
    dataset_arguments = {
        "raw_data_dir": data_config["raw_data_dir"],
        "image_size": data_config["image_size"],
        "targeted_augmentation": data_config["targeted_augmentation"],
        "minority_classes": data_config["targeted_classes"],
    }
    training_dataset = HAM10000Dataset(
        split_dir / "train.csv", training=True, **dataset_arguments
    )
    validation_dataset = HAM10000Dataset(
        split_dir / "validation.csv", training=False, **dataset_arguments
    )
    train_loader = make_loader(
        training_dataset, data_config["batch_size"], data_config["num_workers"], shuffle=True
    )
    validation_loader = make_loader(
        validation_dataset, data_config["batch_size"], data_config["num_workers"], shuffle=False
    )

    model_config = config["model"]
    if model_config.get("use_cbam", False):
        model = build_efficientnet_b0_cbam_classifier(
            num_classes=model_config["num_classes"],
            pretrained=model_config["pretrained"],
            reduction_ratio=model_config.get("cbam_reduction_ratio", 16),
        ).to(device)
    else:
        model = build_efficientnet_b0_classifier(
            num_classes=model_config["num_classes"], pretrained=model_config["pretrained"]
        ).to(device)

    loss_config = config["loss"]
    if loss_config["name"] == "cross_entropy":
        loss_function = nn.CrossEntropyLoss()
    elif loss_config["name"] == "focal":
        if loss_config.get("alpha_strategy") != "effective_number":
            raise ValueError("Focal Loss currently requires alpha_strategy: effective_number.")
        counts = training_class_counts(split_dir / "train.csv")
        alpha = effective_number_class_weights(
            counts, beta=loss_config.get("effective_number_beta", 0.9999)
        ).to(device)
        print(f"Focal Loss alpha weights: {[round(value, 4) for value in alpha.cpu().tolist()]}")
        loss_function = FocalLoss(gamma=loss_config.get("gamma", 2.0), alpha=alpha)
    else:
        raise ValueError(f"Unsupported loss: {loss_config['name']}")
    training_config = config["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    experiment_name = config["experiment_name"]
    checkpoint_dir = Path("checkpoints") / experiment_name
    result_dir = Path("results") / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, result_dir / "config.json")

    best_validation_f1 = float("-inf")
    patience_counter = 0
    history: list[dict[str, float]] = []
    start_epoch = 1
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_validation_f1 = float(checkpoint["validation_macro_f1"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history_path = result_dir / "history.json"
        if history_path.is_file():
            with history_path.open("r", encoding="utf-8") as history_file:
                history = json.load(history_file)
            # A best checkpoint can predate later non-improving epochs. Remove
            # those records because the resumed run will reproduce them.
            history = [record for record in history if record["epoch"] < start_epoch]
        print(
            f"Resumed from {args.resume}: epoch {checkpoint['epoch']} complete; "
            f"best validation macro-F1={best_validation_f1:.4f}"
        )

    for epoch in range(start_epoch, training_config["epochs"] + 1):
        train_loss, train_f1 = run_epoch(model, train_loader, loss_function, device, optimizer)
        validation_loss, validation_f1 = run_epoch(model, validation_loader, loss_function, device)
        scheduler.step(validation_f1)
        learning_rate = optimizer.param_groups[0]["lr"]
        record = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_macro_f1": train_f1,
            "validation_loss": validation_loss,
            "validation_macro_f1": validation_f1,
            "learning_rate": learning_rate,
        }
        history.append(record)
        save_json(history, result_dir / "history.json")
        plot_training_history(history, result_dir / "training_curves.png")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_macro_f1": validation_f1,
                "config": config,
            },
            checkpoint_dir / "last_model.pt",
        )
        print(
            f"Epoch {epoch:02d}/{training_config['epochs']} | "
            f"train loss={train_loss:.4f}, macro-F1={train_f1:.4f} | "
            f"val loss={validation_loss:.4f}, macro-F1={validation_f1:.4f}"
        )

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_macro_f1": validation_f1,
                    "config": config,
                },
                checkpoint_dir / "best_model.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= training_config["early_stopping_patience"]:
                print("Early stopping: validation macro-F1 did not improve.")
                break

    save_json(history, result_dir / "history.json")
    plot_training_history(history, result_dir / "training_curves.png")
    print(f"Best validation macro-F1: {best_validation_f1:.4f}")
    print(f"Checkpoint: {checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
