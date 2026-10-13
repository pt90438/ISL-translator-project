"""Show random images of chosen letters from Training / Validation / Testing.

Usage (from the src/ folder, venv activated):
    python inspect_classes.py --letters Z T S
    python inspect_classes.py --letters Q V U --n 10 --seed 1

Saves one image per letter to <project>/results/inspect_<LETTER>.png
"""
import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SPLITS = ["Training", "Validation", "Testing"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def sample_images(letter, split, n, rng):
    folder = DATA_DIR / split / letter
    if not folder.exists():
        print(f"  Missing folder: {folder}")
        return []
    files = [p for p in folder.iterdir() if p.suffix.lower() in EXTS]
    return rng.sample(files, min(n, len(files)))


def make_grid(letter, n, rng):
    fig, axes = plt.subplots(len(SPLITS), n, figsize=(n * 1.8, len(SPLITS) * 2.0))
    fig.suptitle(f"Letter {letter}: random samples per split", fontsize=14)

    for row, split in enumerate(SPLITS):
        images = sample_images(letter, split, n, rng)
        for col in range(n):
            ax = axes[row][col]
            ax.axis("off")
            if col < len(images):
                ax.imshow(Image.open(images[col]).convert("RGB"))
            if col == 0:
                ax.set_title(split, loc="left", fontsize=10, fontweight="bold")

    fig.tight_layout()
    path = RESULTS_DIR / f"inspect_{letter}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--letters", nargs="+", required=True,
                        help="folder names to inspect, e.g. Z T S")
    parser.add_argument("--n", type=int, default=8, help="images per split")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    for letter in args.letters:
        make_grid(letter, args.n, rng)


if __name__ == "__main__":
    main()