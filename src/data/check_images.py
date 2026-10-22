"""Find image files TensorFlow cannot decode.

Usage (from the src/ folder, venv activated):
    python data/check_images.py          # just report bad files
    python data/check_images.py --fix    # also move them to <project>/bad_images/
"""
import argparse
import shutil
from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "dataset"
QUARANTINE = ROOT / "bad_images"
SPLITS = ["Training", "Validation", "Testing"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def is_valid(path):
    """True if TensorFlow can fully decode the file as an RGB image."""
    try:
        data = tf.io.read_file(str(path))
        tf.io.decode_image(data, channels=3, expand_animations=False)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true",
                        help="move bad files to bad_images/ instead of only reporting")
    args = parser.parse_args()

    bad_files = []
    for split in SPLITS:
        split_dir = DATA_DIR / split
        files = [p for p in split_dir.rglob("*") if p.suffix.lower() in EXTS]
        print(f"Checking {split}: {len(files)} files...")

        for i, path in enumerate(files, 1):
            if not is_valid(path):
                size = path.stat().st_size
                print(f"  BAD: {path.relative_to(DATA_DIR)}  ({size} bytes)")
                bad_files.append(path)
            if i % 2000 == 0:
                print(f"  ...{i}/{len(files)}")

    print(f"\nFound {len(bad_files)} bad file(s).")

    if bad_files and args.fix:
        for path in bad_files:
            dest = QUARANTINE / path.relative_to(DATA_DIR)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(dest))
        print(f"Moved to {QUARANTINE}. Re-run evaluate.py now.")
    elif bad_files:
        print("Re-run with --fix to move them out of the dataset.")


if __name__ == "__main__":
    main()