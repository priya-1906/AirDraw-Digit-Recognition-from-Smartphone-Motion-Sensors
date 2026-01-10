"""Utility helpers for logging, metrics, plotting, and reproducibility."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import logging
import os
import random
import sys
import time

import numpy as np
import matplotlib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# Backend selection:
# - On Linux headless (no DISPLAY/WAYLAND), force Agg.
# - Otherwise, prefer QtAgg if PyQt5 is installed, else use default backend.
is_linux = sys.platform.startswith("linux")
is_mac = sys.platform == "darwin"
has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
HEADLESS = is_linux and not has_display

if HEADLESS:
    matplotlib.use("Agg")
else:
    try:
        import PyQt5  # noqa: F401

        matplotlib.use("QtAgg")
    except Exception:
        pass

import matplotlib.pyplot as plt


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logger(name: str = "imu", level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """Create or reuse a logger with a consistent format."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return the Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


@contextmanager
def timer(label: str, logger: logging.Logger | None = None):
    """Context manager to time a block of code."""
    start = time.time()
    yield
    elapsed = time.time() - start
    msg = f"{label} took {elapsed:.2f}s"
    if logger:
        logger.info(msg)
    else:
        print(msg)


def save_json(path: str | Path, data: dict) -> None:
    """Save a dictionary as JSON."""
    path = Path(path)
    path.write_text(json.dumps(data, indent=2))


def load_json(path: str | Path) -> dict:
    """Load a JSON file into a dictionary."""
    path = Path(path)
    return json.loads(path.read_text())


def compute_accuracy(y_true, y_pred) -> float:
    """Compute classification accuracy."""
    return float(accuracy_score(y_true, y_pred))


def compute_f1(y_true, y_pred, average: str = "macro") -> float:
    """Compute F1-score (macro by default)."""
    return float(f1_score(y_true, y_pred, average=average))


def compute_confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    """Compute confusion matrix for classification."""
    return confusion_matrix(y_true, y_pred, labels=labels)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
    normalize: bool = False,
    save_path: str | Path | None = None,
):
    """Plot a confusion matrix with optional normalization."""
    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)
    plt.colorbar()
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)

    if HEADLESS:
        plt.close()
    else:
        plt.show()


def plot_training_history(history, save_path: str | Path | None = None):
    """Plot loss and accuracy curves from a Keras History object."""
    if hasattr(history, "history"):
        history = history.history

    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(loss, label="train")
    if val_loss:
        axes[0].plot(val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(acc, label="train")
    if val_acc:
        axes[1].plot(val_acc, label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)

    if HEADLESS:
        plt.close()
    else:
        plt.show()
