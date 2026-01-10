"""Data loading and preprocessing utilities for the IMU project."""

from __future__ import annotations

from pathlib import Path
import csv
from collections import Counter

import numpy as np

from utils import setup_logger


def get_project_root() -> Path:
    """Return the project root directory (where this file lives)."""
    return Path(__file__).resolve().parent


def get_preprocessed_root(root: str | Path | None = None) -> Path:
    """Return the preprocessed_data folder path (auto-resolves base vs. direct path)."""
    if root is None:
        base = get_project_root()
    else:
        base = Path(root)

    # If the user already passed the preprocessed_data directory, keep it.
    if base.name == "preprocessed_data":
        preprocessed_root = base
    else:
        preprocessed_root = base / "preprocessed_data"

    if not preprocessed_root.exists():
        raise FileNotFoundError(
            "preprocessed_data folder not found. Please place the folder in the project root."
        )
    return preprocessed_root


def load_metadata(csv_path: Path) -> list[dict]:
    """Load metadata.csv into a list of dictionaries."""
    rows = []
    if not csv_path.exists():
        return rows
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def summarize_labels(labels: np.ndarray) -> dict:
    """Return a count per class label."""
    counts = Counter(labels.tolist())
    return dict(sorted(counts.items(), key=lambda x: x[0]))


def load_split(
    split: str = "train",
    use_scaled: bool = True,
    preprocessed_root: str | Path | None = None,
    logger=None,
) -> dict:
    """Load one split (train/val/test) from preprocessed_data."""
    if logger is None:
        logger = setup_logger("imu-data")

    preprocessed_root = get_preprocessed_root(preprocessed_root)
    split_dir = preprocessed_root / split
    data_path = split_dir / "data.npz"
    meta_path = split_dir / "metadata.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing data.npz for split: {split}")

    with np.load(data_path, allow_pickle=True) as data:
        X = data["X_scaled"] if use_scaled else data["X_raw"]
        y = data["y"]
        feature_mean = data.get("feature_mean")
        feature_std = data.get("feature_std")

    metadata = load_metadata(meta_path)

    logger.info(
        f"Loaded {split} split: X={X.shape}, y={y.shape}, use_scaled={use_scaled}"
    )
    logger.info(f"Metadata rows: {len(metadata)}")
    logger.info(f"Label counts: {summarize_labels(y)}")

    return {
        "X": X,
        "y": y,
        "metadata": metadata,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }


def load_splits(
    use_scaled: bool = True,
    preprocessed_root: str | Path | None = None,
    logger=None,
) -> dict:
    """Load train, val, and test splits."""
    if logger is None:
        logger = setup_logger("imu-data")

    preprocessed_root = get_preprocessed_root(preprocessed_root)
    logger.info(f"Using preprocessed_data at: {preprocessed_root}")

    splits = {}
    for split in ["train", "val", "test"]:
        splits[split] = load_split(
            split=split,
            use_scaled=use_scaled,
            preprocessed_root=preprocessed_root,
            logger=logger,
        )

    return splits
