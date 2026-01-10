"""Training and evaluation routines for the IMU project."""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json

import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report, precision_score, recall_score

import data
import model
from utils import (
    setup_logger,
    ensure_dir,
    set_seed,
    compute_accuracy,
    compute_f1,
    compute_confusion_matrix,
    plot_confusion_matrix,
    plot_training_history,
    timer,
)


def _subset_training_data(X, y, fraction, seed, logger):
    """Return a stratified subset of the training data."""
    if fraction >= 1.0:
        logger.info("Using 100% of training data (no sub-sampling).")
        return X, y
    if fraction <= 0.0:
        raise ValueError("train_fraction must be > 0")

    rng = np.random.default_rng(seed)
    subset_idx = []

    for label in np.unique(y):
        idx = np.where(y == label)[0]
        n = len(idx)
        if n == 0:
            continue
        n_sub = max(1, int(np.floor(n * fraction)))
        rng.shuffle(idx)
        subset_idx.extend(idx[:n_sub])

    subset_idx = np.array(subset_idx)
    rng.shuffle(subset_idx)

    logger.info(
        f"Sub-sampling training data: fraction={fraction:.2f}, "
        f"from {len(y)} to {len(subset_idx)} samples"
    )
    return X[subset_idx], y[subset_idx]


def _time_scale_sample(sample, scale):
    """Apply a small temporal scaling to one sample and return the same length."""
    t_len, num_feat = sample.shape
    new_len = max(2, int(round(t_len * scale)))

    t_old = np.linspace(0.0, 1.0, t_len)
    t_new = np.linspace(0.0, 1.0, new_len)

    # First resample to scaled length
    scaled = np.vstack([
        np.interp(t_new, t_old, sample[:, i]) for i in range(num_feat)
    ]).T

    # Then resample back to original length
    t_back = np.linspace(0.0, 1.0, t_len)
    restored = np.vstack([
        np.interp(t_back, t_new, scaled[:, i]) for i in range(num_feat)
    ]).T

    return restored


def _apply_time_scaling(X, scale_min, scale_max, rng):
    """Apply random time scaling to each sample in X."""
    X_scaled = np.empty_like(X)
    for i in range(X.shape[0]):
        scale = rng.uniform(scale_min, scale_max)
        X_scaled[i] = _time_scale_sample(X[i], scale)
    return X_scaled


def augment_dataset(X, y, copies, noise_std, scale_min, scale_max, seed, logger):
    """Create augmented copies of training data using noise + time scaling."""
    if copies <= 0:
        logger.info("Augmentation disabled (copies <= 0).")
        return X, y

    rng = np.random.default_rng(seed)
    augmented = []

    for _ in range(copies):
        X_aug = X.copy()

        # Add small Gaussian noise
        if noise_std > 0:
            X_aug = X_aug + rng.normal(0.0, noise_std, size=X_aug.shape)

        # Apply time scaling
        if scale_min != 1.0 or scale_max != 1.0:
            X_aug = _apply_time_scaling(X_aug, scale_min, scale_max, rng)

        augmented.append(X_aug)

    X_out = np.concatenate([X] + augmented, axis=0)
    y_out = np.concatenate([y] * (copies + 1), axis=0)

    logger.info(
        "Augmentation applied: "
        f"copies={copies}, noise_std={noise_std}, time_scale=[{scale_min}, {scale_max}]"
    )
    logger.info(f"Training samples after augmentation: {len(y_out)}")

    return X_out, y_out


def train_and_evaluate(args):
    logger = setup_logger("imu-train")

    set_seed(args.seed)

    with timer("Load data", logger):
        splits = data.load_splits(use_scaled=not args.use_raw, preprocessed_root=args.preprocessed_root, logger=logger)

    X_train = splits["train"]["X"]
    y_train = splits["train"]["y"]
    X_val = splits["val"]["X"]
    y_val = splits["val"]["y"]
    X_test = splits["test"]["X"]
    y_test = splits["test"]["y"]

    # Use only a fraction of training data if requested
    X_train, y_train = _subset_training_data(
        X_train, y_train, args.train_fraction, args.seed, logger
    )

    # Always apply data augmentation (noise + slight time scaling)
    X_train, y_train = augment_dataset(
        X_train,
        y_train,
        copies=1,
        noise_std=0.01,
        scale_min=0.9,
        scale_max=1.1,
        seed=args.seed,
        logger=logger,
    )

    input_shape = X_train.shape[1:]
    num_classes = int(len(np.unique(y_train)))

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Number of classes: {num_classes}")

    model_dir = ensure_dir(Path(__file__).resolve().parent / "model")
    model_prefix = args.model_type

    # Fixed training hyperparameters (simple + consistent)
    learning_rate = 1e-3
    batch_size = 32
    logger.info(f"Training config: lr={learning_rate}, batch_size={batch_size}")

    # Select model builder without if/elif chains
    model_builders = {
        "1D_CNN_Feature_Extractor": model.build_1d_cnn_feature_extractor,
        "GRU_temporal_modeling": model.build_gru_temporal_modeling,
    }
    builder = model_builders.get(args.model_type)
    if builder is None:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    net = builder(input_shape, num_classes, learning_rate)

    logger.info("Model summary:")
    net.summary(print_fn=logger.info)

    callbacks = []
    if args.early_stop:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )
        )

    with timer("Training", logger):
        history = net.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

    # Save model in project root/model
    model_path = model_dir / f"{model_prefix}_model.keras"
    net.save(model_path)
    logger.info(f"Saved model: {model_path}")

    # Evaluate on validation and test
    logger.info("Evaluating on validation set...")
    val_pred = np.argmax(net.predict(X_val, verbose=0), axis=1)
    val_acc = compute_accuracy(y_val, val_pred)
    val_f1 = compute_f1(y_val, val_pred)
    val_precision = float(precision_score(y_val, val_pred, average="macro", zero_division=0))
    val_recall = float(recall_score(y_val, val_pred, average="macro", zero_division=0))
    logger.info(f"Val accuracy: {val_acc:.4f} | Val F1 (macro): {val_f1:.4f}")
    logger.info(f"Val precision (macro): {val_precision:.4f} | Val recall (macro): {val_recall:.4f}")

    logger.info("Evaluating on test set...")
    test_pred = np.argmax(net.predict(X_test, verbose=0), axis=1)
    test_acc = compute_accuracy(y_test, test_pred)
    test_f1 = compute_f1(y_test, test_pred)
    test_precision = float(precision_score(y_test, test_pred, average="macro", zero_division=0))
    test_recall = float(recall_score(y_test, test_pred, average="macro", zero_division=0))
    logger.info(f"Test accuracy: {test_acc:.4f} | Test F1 (macro): {test_f1:.4f}")
    logger.info(f"Test precision (macro): {test_precision:.4f} | Test recall (macro): {test_recall:.4f}")

    # Per-digit metrics (classification report)
    report = classification_report(
        y_test,
        test_pred,
        output_dict=True,
        digits=4,
        zero_division=0,
    )

    # Save report as JSON
    report_json_path = model_dir / f"{model_prefix}_test_classification_report.json"
    report_json_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Saved per-digit metrics (JSON): {report_json_path}")

    # Save report as CSV (per-class rows + summary rows)
    report_csv_path = model_dir / f"{model_prefix}_test_classification_report.csv"
    with report_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "precision", "recall", "f1_score", "support"])
        for label, stats in report.items():
            if isinstance(stats, dict):
                writer.writerow(
                    [
                        label,
                        stats.get("precision", ""),
                        stats.get("recall", ""),
                        stats.get("f1-score", ""),
                        stats.get("support", ""),
                    ]
                )
        # Add accuracy row explicitly for clarity
        writer.writerow(["accuracy", test_acc, "", "", ""])
    logger.info(f"Saved per-digit metrics (CSV): {report_csv_path}")

    # Save a compact metrics summary
    summary = {
        "val": {
            "accuracy": val_acc,
            "precision_macro": val_precision,
            "recall_macro": val_recall,
            "f1_macro": val_f1,
        },
        "test": {
            "accuracy": test_acc,
            "precision_macro": test_precision,
            "recall_macro": test_recall,
            "f1_macro": test_f1,
        },
    }
    summary_path = model_dir / f"{model_prefix}_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Saved metrics summary: {summary_path}")

    # Confusion matrix plot
    class_names = [str(i) for i in range(num_classes)]
    cm = compute_confusion_matrix(y_test, test_pred, labels=list(range(num_classes)))
    plot_confusion_matrix(
        cm,
        class_names=class_names,
        title="Test Confusion Matrix",
        normalize=True,
        save_path=model_dir / f"{model_prefix}_confusion_matrix.png",
    )

    # Training curves
    plot_training_history(history, save_path=model_dir / f"{model_prefix}_training_curves.png")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train IMU digit recognition model")

    # Model selection
    parser.add_argument(
        "--model-type",
        choices=["1D_CNN_Feature_Extractor", "GRU_temporal_modeling"],
        default="1D_CNN_Feature_Extractor",
        help="Select model architecture",
    )

    # Training controls
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Fraction of training data to use (e.g., 0.2 for 20%)",
    )

    # I/O controls
    parser.add_argument(
        "--preprocessed-root",
        type=str,
        default=None,
        help="Path to preprocessed_data folder (defaults to project root)",
    )
    parser.add_argument(
        "--use-raw",
        action="store_true",
        help="Use raw features instead of scaled features",
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help="Enable early stopping based on validation loss",
    )

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train_and_evaluate(args)
