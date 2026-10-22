"""Split any flat class-folder dataset into Training / Validation / Testing.

Copies (does not move) images so the source folder stays intact.
Shuffles with a fixed seed so the split is reproducible.

Usage (from the src/ folder, venv activated):

  # Letter dataset (original usage)
  python data/make_splits.py

  # Word dataset
  python data/make_splits.py \
      --source "../dataset/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Frames_Word_Level" \
      --out-prefix Word

  # Dry run — print counts without copying anything
  python data/make_splits.py --source "..." --out-prefix Word --dry-run

Output folders:
  dataset/Word_Training/<class>/
  dataset/Word_Validation/<class>/
  dataset/Word_Testing/<class>/
"""
import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42


def copy_files(files, dest_dir, dry_run):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        if not dry_run:
            shutil.copy2(src, dest_dir / src.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default=str(ROOT / "dataset" / "Indian"),
        help="Path to the flat class-folder dataset (default: dataset/Indian)"
    )
    parser.add_argument(
        "--out-prefix", default="",
        help="Prefix for output split folders, e.g. 'Word' → Word_Training. "
             "Leave empty for the original Training/Validation/Testing names."
    )
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val",   type=float, default=0.15)
    parser.add_argument("--test",  type=float, default=0.15)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print counts without copying anything")
    args = parser.parse_args()

    ratios = {"Training": args.train, "Validation": args.val, "Testing": args.test}
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError("--train + --val + --test must sum to 1.0")

    source_dir = Path(args.source)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source not found: {source_dir}")

    prefix = f"{args.out_prefix}_" if args.out_prefix else ""
    split_names = {k: f"{prefix}{k}" for k in ratios}

    rng = random.Random(SEED)
    class_dirs = sorted(p for p in source_dir.iterdir() if p.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found inside {source_dir}")

    print(f"Source  : {source_dir}")
    print(f"Outputs : {' | '.join(split_names.values())}")
    print(f"Classes : {len(class_dirs)}  "
          f"({', '.join(d.name for d in class_dirs[:10])}"
          f"{'...' if len(class_dirs) > 10 else ''})\n")

    total = {k: 0 for k in split_names.values()}

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

        split_files = {
            "Training":   images[:n_train],
            "Validation": images[n_train: n_train + n_val],
            "Testing":    images[n_train + n_val:],
        }

        print(f"  {class_dir.name:<20}: {n:>4} total  →  "
              + "  ".join(f"{split_names[k]} {len(v)}"
                          for k, v in split_files.items()))

        for split_key, files in split_files.items():
            out_name = split_names[split_key]
            dest = ROOT / "dataset" / out_name / class_dir.name
            copy_files(files, dest, args.dry_run)
            total[out_name] += len(files)

    print(f"\n{'DRY RUN — nothing copied' if args.dry_run else 'Done!'}")
    print("Totals: " + "  |  ".join(f"{k}: {v}" for k, v in total.items()))


if __name__ == "__main__":
    main()