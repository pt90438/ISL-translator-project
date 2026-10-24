"""Admin script to review, approve and merge user contributions.

Usage (from the src/ folder, venv activated):
    python admin_review.py            # interactive review
    python admin_review.py --summary  # just show contribution counts
    python admin_review.py --merge    # merge all approved into Training/Validation/Testing
"""
import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT        = Path(__file__).resolve().parents[1]
CONTRIB_DIR = ROOT / "dataset" / "Contributions"
CONTRIB_CSV = CONTRIB_DIR / "contributions.csv"
APPROVED    = CONTRIB_DIR / "approved"
REJECTED    = CONTRIB_DIR / "rejected"
EXTS        = {".jpg", ".jpeg", ".png"}

TRAIN_DIR = ROOT / "dataset" / "Combined_Training"
VAL_DIR   = ROOT / "dataset" / "Combined_Validation"
TEST_DIR  = ROOT / "dataset" / "Combined_Testing"

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def summary():
    """Print a summary of all contributions."""
    print(f"\n── Contribution summary ──")
    if not CONTRIB_DIR.exists():
        print("No contributions folder found.")
        return

    label_dirs = sorted(d for d in CONTRIB_DIR.iterdir() if d.is_dir()
                        and d.name not in ("approved", "rejected"))
    if not label_dirs:
        print("No contributions yet.")
        return

    total = 0
    for d in label_dirs:
        imgs = list(d.glob("*.jpg"))
        appr = list((APPROVED / d.name).glob("*.jpg")) if (APPROVED / d.name).exists() else []
        rej  = list((REJECTED / d.name).glob("*.jpg")) if (REJECTED / d.name).exists() else []
        print(f"  {d.name:<20}: {len(imgs):>4} pending  "
              f"{len(appr):>4} approved  {len(rej):>4} rejected")
        total += len(imgs)

    print(f"\n  Total pending: {total}")


def review():
    """Interactive review — show each image, approve or reject."""
    if not CONTRIB_DIR.exists():
        print("No contributions found.")
        return

    label_dirs = sorted(d for d in CONTRIB_DIR.iterdir() if d.is_dir()
                        and d.name not in ("approved", "rejected"))

    for label_dir in label_dirs:
        images = sorted(label_dir.glob("*.jpg"))
        if not images:
            continue

        print(f"\n── Reviewing {label_dir.name} ({len(images)} images) ──")
        print("  A = approve  R = reject  S = skip  Q = quit")

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            display = cv2.resize(img, (300, 300))
            cv2.putText(display, label_dir.name, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, img_path.name[:30], (10, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.imshow("Review contribution — A/R/S/Q", display)

            key = cv2.waitKey(0) & 0xFF

            if key == ord("q"):
                cv2.destroyAllWindows()
                return
            elif key == ord("a"):
                dest = APPROVED / label_dir.name
                dest.mkdir(parents=True, exist_ok=True)
                shutil.move(str(img_path), str(dest / img_path.name))
                print(f"  ✓ Approved: {img_path.name}")
            elif key == ord("r"):
                dest = REJECTED / label_dir.name
                dest.mkdir(parents=True, exist_ok=True)
                shutil.move(str(img_path), str(dest / img_path.name))
                print(f"  ✗ Rejected: {img_path.name}")
            elif key == ord("s"):
                print(f"  – Skipped: {img_path.name}")

    cv2.destroyAllWindows()
    print("\nReview complete.")


def merge():
    """Merge approved contributions into the training splits."""
    import random
    rng = random.Random(42)

    if not APPROVED.exists():
        print("No approved contributions to merge.")
        return

    label_dirs = sorted(d for d in APPROVED.iterdir() if d.is_dir())
    if not label_dirs:
        print("No approved contributions to merge.")
        return

    print("\n── Merging approved contributions ──")
    total = 0

    for label_dir in label_dirs:
        images = sorted(label_dir.glob("*.jpg"))
        if not images:
            continue

        rng.shuffle(images)
        n       = len(images)
        n_train = max(1, int(n * SPLIT_RATIOS["train"]))
        n_val   = max(0, int(n * SPLIT_RATIOS["val"]))

        splits = {
            TRAIN_DIR: images[:n_train],
            VAL_DIR:   images[n_train: n_train + n_val],
            TEST_DIR:  images[n_train + n_val:],
        }

        for split_dir, files in splits.items():
            dest = split_dir / label_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dest / f"contrib_{f.name}")

        print(f"  {label_dir.name:<20}: {n} images merged "
              f"(train {n_train} / val {n_val} / test {n - n_train - n_val})")
        total += n

    print(f"\nMerged {total} images total.")
    print("Run python train.py to retrain with the new contributions.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--merge",   action="store_true")
    args = parser.parse_args()

    if args.summary:
        summary()
    elif args.merge:
        summary()
        confirm = input("\nMerge all approved contributions? (y/n): ")
        if confirm.lower() == "y":
            merge()
    else:
        summary()
        review()


if __name__ == "__main__":
    main()