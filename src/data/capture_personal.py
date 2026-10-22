"""Guided webcam capture for personal ISL alphabet dataset.

Walks through each letter one by one. Press SPACE to capture a frame,
R to retake the last image, ENTER to move to the next letter, Q to quit.

Saves to dataset/Personal/<LETTER>/ — run make_splits.py on it afterwards.

Usage (from the src/ folder, venv activated):
    python data/capture_personal.py               # all 26 letters A-Z
    python data/capture_personal.py --letters A B C  # specific letters only
    python data/capture_personal.py --n 50        # 50 images per letter
"""
import argparse
import string
import time
from pathlib import Path

import cv2

ROOT       = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "dataset" / "Personal"

COL_GREEN  = (0, 220, 0)
COL_RED    = (0, 0, 220)
COL_WHITE  = (255, 255, 255)
COL_BLACK  = (0, 0, 0)
COL_YELLOW = (0, 220, 220)


def draw_overlay(frame, letter, captured, target, status_msg, last_crop):
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), COL_BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Big letter
    cv2.putText(frame, f"Sign:  {letter}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, COL_YELLOW, 3)

    # Progress
    pct = f"{captured}/{target}"
    cv2.putText(frame, pct, (w - 160, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                COL_GREEN if captured >= target else COL_WHITE, 2)

    # Progress bar
    bar_w = int((captured / target) * w)
    cv2.rectangle(frame, (0, 88), (bar_w, 92), COL_GREEN, -1)

    # Status message
    if status_msg:
        cv2.putText(frame, status_msg, (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_GREEN, 2)

    # Controls
    hints = "SPACE=capture  R=delete last  ENTER=next letter  Q=quit"
    cv2.putText(frame, hints, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Thumbnail of last captured image
    if last_crop is not None:
        thumb = cv2.resize(last_crop, (80, 80))
        frame[h - 100: h - 20, w - 90: w - 10] = thumb
        cv2.putText(frame, "last", (w - 90, h - 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_WHITE, 1)

    return frame


def countdown_and_capture(cap, frame, letter, path, seconds):
    """Show a countdown overlay then capture a frame automatically."""
    for i in range(seconds, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        # Big countdown number in centre
        cv2.putText(frame, str(i), (w // 2 - 40, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 220), 8)
        cv2.putText(frame, f"Get ready: {letter}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 220), 3)
        cv2.imshow("ISL — Personal Capture", frame)
        cv2.waitKey(1000)

    # Flash green on capture
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        flash = frame.copy()
        flash[:] = (0, 220, 0)
        cv2.addWeighted(flash, 0.3, frame, 0.7, 0, frame)
        cv2.imshow("ISL — Personal Capture", frame)
        cv2.waitKey(200)
        cv2.imwrite(str(path), frame)
        return frame
    return None


def capture_letter(cap, letter, target, out_dir, countdown=0):
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.jpg"))
    captured  = len(existing)
    img_idx   = captured
    last_crop = None

    if countdown > 0:
        status = f"Press SPACE to start {countdown}s countdown, then hold '{letter}'"
    else:
        status = f"Ready — show '{letter}' and press SPACE"

    print(f"\n--- Letter {letter}  (already have {captured}/{target}) ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        display = draw_overlay(
            frame.copy(), letter, captured, target, status, last_crop
        )
        cv2.imshow("ISL — Personal Capture", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return "quit"

        elif key == 13:   # ENTER
            if captured < target:
                status = f"Only {captured}/{target} — press ENTER again to skip"
                cv2.imshow("ISL — Personal Capture", display)
                k2 = cv2.waitKey(1500) & 0xFF
                if k2 == 13:
                    return "next"
            else:
                return "next"

        elif key == ord(" "):   # SPACE
            path = out_dir / f"{letter}_{img_idx:04d}.jpg"

            if countdown > 0:
                # Countdown mode — auto-capture after N seconds
                saved = countdown_and_capture(cap, frame, letter, path, countdown)
                if saved is not None:
                    last_crop = saved
                    img_idx  += 1
                    captured += 1
                    status    = (f"Captured {captured}/{target} — "
                                 f"SPACE for next, ENTER when done")
                    print(f"  Saved {path.name}")
            else:
                # Instant capture
                cv2.imwrite(str(path), frame)
                last_crop = frame.copy()
                img_idx  += 1
                captured += 1
                status    = f"Captured {captured}/{target}"
                print(f"  Saved {path.name}")

            if captured >= target:
                status = f"Done! {target} images. Press ENTER for next letter."

        elif key == ord("r"):   # R — delete last
            files = sorted(out_dir.glob("*.jpg"))
            if files:
                files[-1].unlink()
                captured  -= 1
                img_idx   -= 1
                last_crop  = None
                status     = f"Deleted last. Now {captured}/{target}"
                print(f"  Deleted {files[-1].name}")
            else:
                status = "Nothing to delete"

    return "quit"


def main():
    all_letters = list(string.ascii_uppercase)

    parser = argparse.ArgumentParser()
    parser.add_argument("--letters", nargs="+", default=all_letters,
                        help="Letters to capture (default: A-Z)")
    parser.add_argument("--n", type=int, default=40,
                        help="Target images per letter (default: 40)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--countdown", type=int, default=0,
                        help="Seconds to count down before auto-capturing "
                             "(0 = instant capture on SPACE). Use 3 for "
                             "two-hand signs so you have time to position both hands.")
    args = parser.parse_args()

    letters = [l.upper() for l in args.letters]
    print(f"Capturing {args.n} images each for: {' '.join(letters)}")
    print(f"Saving to: {OUTPUT_DIR}")
    print("Controls: SPACE=capture  R=delete last  ENTER=next  Q=quit\n")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    # Warmup
    for _ in range(5):
        cap.read()

    for letter in letters:
        out_dir = OUTPUT_DIR / letter
        result  = capture_letter(cap, letter, args.n, out_dir, args.countdown)
        if result == "quit":
            print("\nCapture stopped early.")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Summary
    print("\n── Capture summary ──")
    total = 0
    for letter in letters:
        d = OUTPUT_DIR / letter
        n = len(list(d.glob("*.jpg"))) if d.exists() else 0
        total += n
        status = "✓" if n >= args.n else f"only {n}"
        print(f"  {letter}: {status}")
    print(f"\nTotal images: {total}")
    print(f"\nNext step — run the split script:")
    print(f'  python data/make_splits.py --source "../dataset/Personal" --out-prefix Personal')
    print(f"Then retrain:")
    print(f"  python train.py  (update DATA_DIR in train.py to point at Personal splits)")


if __name__ == "__main__":
    main()