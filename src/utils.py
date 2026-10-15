"""Reproducibility, device selection, and training-output helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for repeatable experiments where platform permits."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms can reject some MPS operations, so use stable
    # seeds and deterministic CuDNN settings without forcing all operations.
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device() -> torch.device:
    """Prefer CUDA, then Apple Metal (MPS), and fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(data: dict[str, Any] | list[dict[str, Any]], path: Path) -> None:
    """Write readable UTF-8 JSON, creating parent folders as necessary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)


def plot_training_history(history: list[dict[str, float]], output_path: Path) -> None:
    """Plot loss and macro-F1 curves from epoch-level history."""
    if not history:
        return
    epochs = [record["epoch"] for record in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [record["train_loss"] for record in history], label="Train")
    axes[0].plot(epochs, [record["validation_loss"] for record in history], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [record["train_macro_f1"] for record in history], label="Train")
    axes[1].plot(epochs, [record["validation_macro_f1"] for record in history], label="Validation")
    axes[1].set(title="Macro F1-score", xlabel="Epoch", ylabel="Macro F1")
    axes[1].legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
