"""Evaluate the trained model: accuracy, per-class report, confusion matrix.

Usage (from the src/ folder, venv activated):
    python evaluate.py                    # evaluates Validation and Testing
    python evaluate.py --split Validation
    python evaluate.py --model ../checkpoints/final_model.keras
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to files; no window needed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from data.dataset_loader import TEST_DIR, VAL_DIR, load_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def get_predictions(model, ds):
    """Returns true labels, predicted labels, and prediction confidences."""
    y_true = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in ds])
    probs = model.predict(ds, verbose=0)  # ds is unshuffled, so order matches
    y_pred = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    return y_true, y_pred, confidence


def plot_confusion_matrix(cm, class_names, title, path):
    # Normalize each row so cells show "fraction of this letter predicted as X"
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.45), max(7, n * 0.4)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Show percentages for the off-diagonal confusions that actually matter
    for i in range(n):
        for j in range(n):
            if i != j and cm_norm[i, j] >= 0.05:
                ax.text(j, i, f"{cm_norm[i, j]*100:.0f}", ha="center",
                        va="center", fontsize=7, color="black")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def top_confusions(cm, class_names, k=10):
    """Most frequent (true -> predicted) mistakes, by count."""
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], class_names[i], class_names[j]))
    return sorted(pairs, reverse=True)[:k]


def evaluate_split(model, split_name, class_names):
    print(f"\n{'=' * 60}\nEvaluating: {split_name}\n{'=' * 60}")
    ds = load_split(f"{DATA_DIR}/{split_name}", shuffle=False)

    if ds.class_names != class_names:
        raise ValueError(
            f"Class mismatch in {split_name}!\n"
            f"  Expected: {class_names}\n  Found:    {ds.class_names}\n"
            "Make sure all splits have identical letter folders."
        )

    y_true, y_pred, conf = get_predictions(model, ds)
    accuracy = float(np.mean(y_true == y_pred))
    print(f"Samples: {len(y_true)}   Accuracy: {accuracy:.4f}")
    print(f"Mean confidence (correct):   {conf[y_true == y_pred].mean():.3f}")
    if np.any(y_true != y_pred):
        print(f"Mean confidence (incorrect): {conf[y_true != y_pred].mean():.3f}")

    # Per-class precision / recall / F1
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=3, zero_division=0
    )
    print("\n" + report)

    # Confusion matrix + worst classes
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    per_class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)

    print("Weakest 5 letters (recall):")
    for idx in np.argsort(per_class_acc)[:5]:
        print(f"  {class_names[idx]}: {per_class_acc[idx]:.3f}")

    print("\nMost common mistakes (true -> predicted):")
    for count, true_c, pred_c in top_confusions(cm, class_names):
        print(f"  {true_c} -> {pred_c}: {count}x")

    # Save outputs
    tag = split_name.lower()
    plot_confusion_matrix(
        cm, class_names, f"Confusion matrix ({split_name}, acc={accuracy:.3f})",
        RESULTS_DIR / f"confusion_matrix_{tag}.png",
    )
    (RESULTS_DIR / f"classification_report_{tag}.txt").write_text(report)
    print(f"\nSaved to {RESULTS_DIR}/ (confusion_matrix_{tag}.png, "
          f"classification_report_{tag}.txt)")

    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(CKPT_DIR / "best_model.keras"))
    parser.add_argument(
        "--split", choices=[VAL_DIR, TEST_DIR, "both"], default="both"
    )
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    class_names = json.loads((CKPT_DIR / "class_names.json").read_text())
    print(f"Loaded {args.model}  ({len(class_names)} classes)")

    splits = [VAL_DIR, TEST_DIR] if args.split == "both" else [args.split]
    results = {s: evaluate_split(model, s, class_names) for s in splits}

    if len(results) == 2:
        gap = abs(results[VAL_DIR] - results[TEST_DIR])
        print(f"\nValidation vs Testing accuracy gap: {gap:.4f}")


if __name__ == "__main__":
    main()