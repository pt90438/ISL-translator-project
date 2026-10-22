"""Merge multiple split datasets into one combined Training/Validation/Testing.

Copies images from each source dataset's split folders into a single
combined output, keeping per-class subfolder structure intact.

Usage (from the src/ folder, venv activated):

  # Merge Indian dataset splits with your personal captures
  python data/merge_datasets.py \
      --sources "Training:../dataset/Training" "Training:../dataset/Personal_Training" \
      --sources "Validation:../dataset/Validation" "Validation:../dataset/Personal_Validation" \
      --sources "Testing:../dataset/Testing" "Testing:../dataset/Personal_Testing" \
      --out-prefix Combined

  # Simpler: just point at the root split folders directly
  python data/merge_datasets.py \
      --datasets Indian Personal \
      --out-prefix Combined

  # Dry run first
  python data/merge_datasets.py --datasets Indian Personal --out-prefix Combined --dry-run
"""
import argparse
import shutil
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[2]
SPLITS = ["Training", "Validation", "Testing"]
EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def merge(sources, out_dir, dry_run):
    """Copy all images from sources into out_dir/<class>/, renaming to avoid collisions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in out_dir.glob("*.*")}
    copied = 0

    for src_dir in sources:
        if not src_dir.exists():
            print(f"    SKIP (not found): {src_dir}")
            continue
        files = [p for p in src_dir.iterdir() if p.suffix.lower() in EXTS]
        for f in files:
            # Avoid filename collisions by prefixing with source folder name
            new_name = f"{src_dir.parent.name}_{f.name}"
            dest = out_dir / new_name
            if not dry_run:
                shutil.copy2(f, dest)
            copied += 1

    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+",
        help="Dataset folder names under dataset/ to merge, e.g. --datasets Indian Personal"
    )
    parser.add_argument(
        "--out-prefix", default="Combined",
        help="Prefix for output folders, e.g. 'Combined' → Combined_Training etc."
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.datasets:
        parser.error("Provide --datasets, e.g. --datasets Indian Personal")

    print(f"Merging datasets: {args.datasets}")
    print(f"Output prefix   : {args.out_prefix}")
    print(f"Dry run         : {args.dry_run}\n")

    total = 0

    for split in SPLITS:
        out_split_dir = ROOT / "dataset" / f"{args.out_prefix}_{split}"
        print(f"── {split} ──")

        # Collect all class names across all datasets for this split
        all_classes = set()
        for ds in args.datasets:
            split_dir = ROOT / "dataset" / ds / split
            # Also check prefixed split names e.g. Personal_Training
            prefixed = ROOT / "dataset" / f"{ds}_{split}"
            check_dir = split_dir if split_dir.exists() else prefixed
            if check_dir.exists():
                all_classes.update(
                    p.name for p in check_dir.iterdir() if p.is_dir()
                )

        if not all_classes:
            print(f"  No class folders found for {split} in any dataset — skipping\n")
            continue

        split_total = 0
        for cls in sorted(all_classes):
            sources = []
            for ds in args.datasets:
                split_dir     = ROOT / "dataset" / ds / split / cls
                prefixed_dir  = ROOT / "dataset" / f"{ds}_{split}" / cls
                if split_dir.exists():
                    sources.append(split_dir)
                elif prefixed_dir.exists():
                    sources.append(prefixed_dir)

            out_class_dir = out_split_dir / cls
            n = merge(sources, out_class_dir, args.dry_run)
            src_counts = " + ".join(
                f"{s.parent.parent.name} {len([f for f in s.iterdir() if f.suffix.lower() in EXTS])}"
                if s.exists() else "0"
                for s in sources
            )
            print(f"  {cls:<20}: {n:>4} images  ({src_counts})")
            split_total += n

        print(f"  Total {split}: {split_total}\n")
        total += split_total

    print(f"{'DRY RUN — nothing copied' if args.dry_run else 'Done!'}")
    print(f"Grand total: {total} images")
    print(f"\nNext — retrain on the combined dataset:")
    print(f"  Update DATA_DIR in train.py to point at dataset/{args.out_prefix}")
    print(f"  Or run: python train.py  (after updating the path)")


if __name__ == "__main__":
    main()