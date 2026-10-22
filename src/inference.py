"""Real-time ISL alphabet classifier using webcam + MediaPipe hand detection.

Controls:
    Q / ESC   — quit
    SPACE     — add a space to the current word
    BACKSPACE — delete last character
    ENTER     — speak the current text (requires pyttsx3)
    C         — clear the text buffer

Usage (from the src/ folder, venv activated):
    python inference.py
    python inference.py --camera 1
    python inference.py --no-speech
"""
import argparse
import collections
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import tensorflow as tf

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "checkpoints" / "best_model.keras"
NAMES_PATH = ROOT / "checkpoints" / "class_names.json"

# MediaPipe hand landmarker model — downloaded on first run
MP_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MP_MODEL_PATH = ROOT / "checkpoints" / "hand_landmarker.task"

# ── Inference settings ───────────────────────────────────────────────────────
IMG_SIZE       = 224
CONF_THRESHOLD = 0.92
SMOOTH_FRAMES  = 10
COOLDOWN_SEC   = 1.5
HAND_PADDING   = 30

# ── Colours (BGR) ────────────────────────────────────────────────────────────
COL_GREEN  = (0, 220, 0)
COL_ORANGE = (0, 165, 255)
COL_WHITE  = (255, 255, 255)
COL_BLACK  = (0, 0, 0)


def download_mp_model():
    """Download the MediaPipe hand landmarker .task file if not present."""
    if MP_MODEL_PATH.exists():
        return
    print("Downloading MediaPipe hand landmarker model (~9 MB)...")
    import urllib.request
    MP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MP_MODEL_URL, MP_MODEL_PATH)
    print("Downloaded.")


def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = json.loads(NAMES_PATH.read_text())
    return model, class_names


def init_tts():
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        return engine
    except Exception:
        return None


def speak(engine, text):
    if engine and text.strip():
        engine.say(text)
        engine.runAndWait()


def crop_hand(frame, landmarks, h, w):
    """Crop a padded square region around detected hand landmarks."""
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1 = max(0, int(min(xs)) - HAND_PADDING)
    y1 = max(0, int(min(ys)) - HAND_PADDING)
    x2 = min(w, int(max(xs)) + HAND_PADDING)
    y2 = min(h, int(max(ys)) + HAND_PADDING)

    bw, bh = x2 - x1, y2 - y1
    if bh > bw:
        diff = bh - bw
        x1 = max(0, x1 - diff // 2)
        x2 = min(w, x2 + diff // 2)
    else:
        diff = bw - bh
        y1 = max(0, y1 - diff // 2)
        y2 = min(h, y2 + diff // 2)

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def draw_landmarks_on_frame(frame, hand_landmarks_list):
    """Draw hand skeleton directly with cv2 — no protobuf dependency."""
    h, w = frame.shape[:2]

    # MediaPipe hand connections (pairs of landmark indices)
    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),           # thumb
        (0,5),(5,6),(6,7),(7,8),           # index
        (0,9),(9,10),(10,11),(11,12),      # middle
        (0,13),(13,14),(14,15),(15,16),    # ring
        (0,17),(17,18),(18,19),(19,20),    # pinky
        (5,9),(9,13),(13,17),              # palm
    ]

    for hand_landmarks in hand_landmarks_list:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 255, 0), -1)


def preprocess(crop):
    img = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img.astype(np.float32), axis=0)


def draw_ui(frame, letter, confidence, text_buffer, fps, hand_box, confirmed):
    h, w = frame.shape[:2]

    if hand_box:
        x1, y1, x2, y2 = hand_box
        color = COL_GREEN if confirmed else COL_ORANGE
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 80), (w, h), COL_BLACK, -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    if letter:
        label = f"{letter}  {confidence * 100:.0f}%"
        color = COL_GREEN if confirmed else COL_ORANGE
        cv2.putText(frame, label, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 3)

    display_text = "Text: " + (text_buffer if text_buffer else "_")
    cv2.putText(frame, display_text, (10, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_WHITE, 2)

    hints = "SPACE=space  BKSP=delete  ENTER=speak  C=clear  Q=quit"
    cv2.putText(frame, hints, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.putText(frame, f"FPS {fps:.0f}", (w - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-speech", action="store_true")
    args = parser.parse_args()

    download_mp_model()

    print("Loading model...")
    model, class_names = load_model_and_labels()
    print(f"Loaded. Classes: {len(class_names)}")

    tts_engine = None if args.no_speech else init_tts()
    if tts_engine is None and not args.no_speech:
        print("pyttsx3 not found — speech disabled.")

    # New MediaPipe hand landmarker API
    base_options = mp_python.BaseOptions(model_asset_path=str(MP_MODEL_PATH))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    pred_buffer       = collections.deque(maxlen=SMOOTH_FRAMES)
    text_buffer       = ""
    last_added_letter = ""
    last_added_time   = 0.0
    prev_time         = time.time()

    print("Camera open. Press Q or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detection = landmarker.detect(mp_image)

        letter    = None
        conf      = 0.0
        hand_box  = None
        confirmed = False

        if detection.hand_landmarks:
            hand_lms = detection.hand_landmarks[0]
            draw_landmarks_on_frame(frame, detection.hand_landmarks)

            crop, hand_box = crop_hand(frame, hand_lms, h, w)

            if crop.size > 0:
                inp   = preprocess(crop)
                preds = model.predict(inp, verbose=0)[0]
                idx   = int(np.argmax(preds))
                conf  = float(preds[idx])
                letter = class_names[idx] if conf >= CONF_THRESHOLD else None

                pred_buffer.append(letter)

                if (len(pred_buffer) == SMOOTH_FRAMES
                        and len(set(pred_buffer)) == 1
                        and letter is not None):
                    confirmed = True
                    now = time.time()
                    if (letter != last_added_letter
                            or now - last_added_time > COOLDOWN_SEC):
                        text_buffer       += letter
                        last_added_letter  = letter
                        last_added_time    = now
        else:
            pred_buffer.clear()

        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        frame = draw_ui(frame, letter, conf, text_buffer, fps, hand_box, confirmed)
        cv2.imshow("ISL Translator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            text_buffer += " "
        elif key == 8:
            text_buffer = text_buffer[:-1]
        elif key == 13:
            speak(tts_engine, text_buffer)
        elif key == ord("c"):
            text_buffer = ""

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()