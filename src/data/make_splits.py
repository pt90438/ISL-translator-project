"""Split dataset/Indian into dataset/Training, Validation, Testing.

Copies (does not move) images so the original Indian/ folder stays intact.
Shuffles with a fixed seed so the split is reproducible.

Usage (from the src/ folder, venv activated):
    python data/make_splits.py
    python data/make_splits.py --train 0.7 --val 0.15 --test 0.15
    python data/make_splits.py --dry-run   # just print counts, copy nothing
"""
import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "dataset" / "Indian"
SPLITS = {"Training": 0.70, "Validation": 0.15, "Testing": 0.15}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42


def copy_files(files, dest_dir, dry_run):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        if not dry_run:
            shutil.copy2(src, dest_dir / src.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val",   type=float, default=0.15)
    parser.add_argument("--test",  type=float, default=0.15)
    parser.add_argument("--dry-run", action="store_true",
                        help="print counts without copying anything")
    args = parser.parse_args()

    ratios = {"Training": args.train, "Validation": args.val, "Testing": args.test}
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError("train + val + test must sum to 1.0")

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source not found: {SOURCE_DIR}")

    rng = random.Random(SEED)
    class_dirs = sorted(p for p in SOURCE_DIR.iterdir() if p.is_dir())

    if not class_dirs:
        raise FileNotFoundError(f"No class folders found inside {SOURCE_DIR}")

    print(f"Source : {SOURCE_DIR}")
    print(f"Classes: {len(class_dirs)}  "
          f"({', '.join(d.name for d in class_dirs)})\n")

    total = {"Training": 0, "Validation": 0, "Testing": 0}

    for class_dir in class_dirs:
        images = sorted(
            p for p in class_dir.iterdir() if p.suffix.lower() in EXTS
        )
        if not images:
            print(f"  WARNING: no images in {class_dir.name}, skipping")
            continue

        rng.shuffle(images)
        n = len(images)
        n_train = int(n * ratios["Training"])
        n_val   = int(n * ratios["Validation"])
        # give any rounding remainder to test
        splits_files = {
            "Training":   images[:n_train],
            "Validation": images[n_train : n_train + n_val],
            "Testing":    images[n_train + n_val :],
        }

        print(f"  {class_dir.name:>3}: {n:>5} total  →  "
              + "  ".join(f"{k} {len(v)}" for k, v in splits_files.items()))

        for split_name, files in splits_files.items():
            dest = ROOT / "dataset" / split_name / class_dir.name
            copy_files(files, dest, args.dry_run)
            total[split_name] += len(files)

    print(f"\n{'DRY RUN — nothing copied' if args.dry_run else 'Done!'}")
    print("Totals: " + "  |  ".join(f"{k}: {v}" for k, v in total.items()))


if __name__ == "__main__":
    main()