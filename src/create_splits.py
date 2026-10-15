"""Create reproducible, lesion-grouped HAM10000 train/validation/test splits.

Run from the project root:
    python -m src.create_splits --data-dir data/raw --output-dir data/splits

The two-stage StratifiedGroupKFold procedure creates an approximately
71.4% / 14.3% / 14.3% split while keeping each lesion_id in one split only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


METADATA_FILENAMES = ("HAM10000_metadata.csv", "HAM10000_metadata.tab")
NUM_OUTER_FOLDS = 7
NUM_INNER_FOLDS = 6


def load_metadata(data_dir: Path) -> pd.DataFrame:
    """Load the supplied HAM10000 metadata file without modifying it."""
    metadata_path = next(
        (data_dir / filename for filename in METADATA_FILENAMES if (data_dir / filename).is_file()),
        None,
    )
    if metadata_path is None:
        raise FileNotFoundError(
            "No HAM10000 metadata file found. Expected one of: "
            f"{', '.join(METADATA_FILENAMES)} in {data_dir}"
        )
    return pd.read_csv(metadata_path, sep="\t" if metadata_path.suffix == ".tab" else ",")


def assert_no_group_overlap(splits: dict[str, pd.DataFrame]) -> None:
    """Stop immediately if a lesion appears in more than one data split."""
    names = tuple(splits)
    for index, first_name in enumerate(names):
        first_groups = set(splits[first_name]["lesion_id"])
        for second_name in names[index + 1 :]:
            overlap = first_groups.intersection(splits[second_name]["lesion_id"])
            if overlap:
                raise AssertionError(
                    f"Leakage detected: {len(overlap)} lesion_id values overlap "
                    f"between {first_name} and {second_name}."
                )


def add_relative_image_path(frame: pd.DataFrame) -> pd.DataFrame:
    """Add portable image paths relative to the data/raw directory."""
    result = frame.copy()
    result["image_path"] = result["image_id"].map(lambda image_id: f"images/{image_id}.jpg")
    return result


def build_summary(splits: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Return JSON-serializable split sizes and per-class counts."""
    return {
        name: {
            "images": int(len(frame)),
            "unique_lesions": int(frame["lesion_id"].nunique()),
            "class_counts": {label: int(count) for label, count in frame["dx"].value_counts().sort_index().items()},
        }
        for name, frame in splits.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    metadata = load_metadata(data_dir)
    required_columns = {"image_id", "lesion_id", "dx"}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing_columns)}")
    if metadata["lesion_id"].isna().any():
        raise ValueError("lesion_id contains missing values; grouped splitting cannot proceed safely.")

    outer_splitter = StratifiedGroupKFold(
        n_splits=NUM_OUTER_FOLDS, shuffle=True, random_state=args.seed
    )
    outer_splits = list(
        outer_splitter.split(metadata, y=metadata["dx"], groups=metadata["lesion_id"])
    )
    train_validation_indices, test_indices = outer_splits[0]

    train_validation = metadata.iloc[train_validation_indices].reset_index(drop=True)
    inner_splitter = StratifiedGroupKFold(
        n_splits=NUM_INNER_FOLDS, shuffle=True, random_state=args.seed + 1
    )
    inner_splits = list(
        inner_splitter.split(
            train_validation,
            y=train_validation["dx"],
            groups=train_validation["lesion_id"],
        )
    )
    train_indices, validation_indices = inner_splits[0]

    splits = {
        "train": add_relative_image_path(train_validation.iloc[train_indices]),
        "validation": add_relative_image_path(train_validation.iloc[validation_indices]),
        "test": add_relative_image_path(metadata.iloc[test_indices]),
    }
    assert_no_group_overlap(splits)
    for split_name, frame in splits.items():
        missing_classes = set(metadata["dx"].unique()).difference(frame["dx"].unique())
        if missing_classes:
            raise AssertionError(f"{split_name} is missing class(es): {sorted(missing_classes)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, frame in splits.items():
        frame.sort_values("image_id").to_csv(output_dir / f"{split_name}.csv", index=False)

    summary = {
        "seed": args.seed,
        "method": "two-stage StratifiedGroupKFold grouped by lesion_id",
        "target_proportions": {"train": "~71.4%", "validation": "~14.3%", "test": "~14.3%"},
        "splits": build_summary(splits),
    }
    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved leakage-checked splits to: {output_dir}")


if __name__ == "__main__":
    main()
