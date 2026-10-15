"""PyTorch dataset for the leakage-checked HAM10000 split CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .constants import CLASS_TO_INDEX, DEFAULT_TARGETED_AUGMENTATION_CLASSES
from .preprocessing import (
    build_base_training_transform,
    build_evaluation_transform,
    build_targeted_training_transform,
)


class HAM10000Dataset(Dataset[tuple[Tensor, int, str]]):
    """Load an existing split without changing class labels or data membership."""

    def __init__(
        self,
        split_csv: str | Path,
        raw_data_dir: str | Path,
        *,
        training: bool = False,
        image_size: int = 224,
        targeted_augmentation: bool = False,
        minority_classes: Sequence[str] = DEFAULT_TARGETED_AUGMENTATION_CLASSES,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.raw_data_dir = Path(raw_data_dir)
        self.frame = pd.read_csv(self.split_csv)
        required_columns = {"image_id", "dx", "image_path"}
        missing = required_columns.difference(self.frame.columns)
        if missing:
            raise ValueError(f"{self.split_csv} is missing required column(s): {sorted(missing)}")

        unknown_classes = set(self.frame["dx"]).difference(CLASS_TO_INDEX)
        if unknown_classes:
            raise ValueError(f"Unexpected diagnosis label(s): {sorted(unknown_classes)}")

        self.training = training
        self.targeted_augmentation = targeted_augmentation
        self.minority_classes = set(minority_classes)
        self.base_transform: Callable[[Image.Image], Tensor]
        self.targeted_transform: Callable[[Image.Image], Tensor]
        if training:
            self.base_transform = build_base_training_transform(image_size)
            self.targeted_transform = build_targeted_training_transform(image_size)
        else:
            evaluation_transform = build_evaluation_transform(image_size)
            self.base_transform = evaluation_transform
            self.targeted_transform = evaluation_transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Tensor, int, str]:
        row = self.frame.iloc[index]
        image_path = self.raw_data_dir / row["image_path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Image listed in split is missing: {image_path}")

        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")

        use_targeted_transform = (
            self.training
            and self.targeted_augmentation
            and row["dx"] in self.minority_classes
        )
        image_tensor = (
            self.targeted_transform(image) if use_targeted_transform else self.base_transform(image)
        )
        label = CLASS_TO_INDEX[row["dx"]]
        return image_tensor, label, str(row["image_id"])
