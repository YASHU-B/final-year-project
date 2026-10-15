"""Audit HAM10000 metadata before creating any train/validation/test split.

Run from the project root:
    python -m src.audit_ham10000 --data-dir data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
METADATA_FILENAMES = ("HAM10000_metadata.csv", "HAM10000_metadata.tab")


def find_image_files(data_dir: Path) -> list[Path]:
    """Return all JPEG images in the supplied HAM10000 directory."""
    return sorted(
        path
        for pattern in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
        for path in data_dir.rglob(pattern)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing HAM10000_metadata.csv and image folders.",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    metadata_path = next(
        (data_dir / filename for filename in METADATA_FILENAMES if (data_dir / filename).is_file()),
        None,
    )

    if metadata_path is None:
        raise FileNotFoundError(
            f"Metadata file not found in: {data_dir}\n"
            "Download HAM10000 from the official source and place either "
            "HAM10000_metadata.csv or HAM10000_metadata.tab in data/raw/."
        )

    separator = "\t" if metadata_path.suffix == ".tab" else ","
    metadata = pd.read_csv(metadata_path, sep=separator)
    required_columns = {"image_id", "dx"}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        raise ValueError(
            f"Metadata is missing required column(s): {sorted(missing_columns)}"
        )

    image_paths = find_image_files(data_dir)
    image_stems = {path.stem for path in image_paths}
    metadata_ids = set(metadata["image_id"].astype(str))
    missing_images = metadata_ids.difference(image_stems)
    unreferenced_images = image_stems.difference(metadata_ids)

    print("HAM10000 metadata audit")
    print("=" * 24)
    print(f"Metadata rows: {len(metadata):,}")
    print(f"Unique image IDs: {metadata['image_id'].nunique():,}")
    print(f"Image files found: {len(image_paths):,}")
    print(f"Duplicate image IDs: {metadata['image_id'].duplicated().sum():,}")
    print(f"Metadata IDs without image files: {len(missing_images):,}")
    print(f"Image files without metadata rows: {len(unreferenced_images):,}")

    print("\nClass distribution")
    distribution = metadata["dx"].value_counts().reindex(EXPECTED_CLASSES, fill_value=0)
    summary = pd.DataFrame(
        {
            "count": distribution,
            "percentage": (distribution / len(metadata) * 100).round(2),
        }
    )
    print(summary.to_string())

    unknown_classes = sorted(set(metadata["dx"].dropna()) - set(EXPECTED_CLASSES))
    if unknown_classes:
        print(f"\nWarning: unexpected diagnosis labels: {unknown_classes}")

    candidate_group_columns = [
        column
        for column in ("patient_id", "lesion_id")
        if column in metadata.columns
    ]
    print("\nLeakage-relevant grouping fields")
    if not candidate_group_columns:
        print("No patient_id or lesion_id column was found. Do not claim a patient-wise split.")
    else:
        for column in candidate_group_columns:
            groups = metadata[column].dropna()
            repeated = groups.duplicated(keep=False).sum()
            print(
                f"{column}: {groups.nunique():,} unique non-null values; "
                f"{repeated:,} rows belong to repeated groups"
            )

    print("\nAudit complete. Create splits only after reviewing this output.")


if __name__ == "__main__":
    main()
