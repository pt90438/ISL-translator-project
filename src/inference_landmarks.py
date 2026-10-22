"""Real-time ISL classifier using the 28x28 pixel CNN model.

Captures webcam frames, crops around the detected hand,
resizes to 28x28 grayscale, and classifies with the trained CNN.

Controls:
    Q / ESC   — quit
    SPACE     — add a space
    BACKSPACE — delete last character
    ENTER     — speak the text
    C         — clear buffer

Usage:
    python inference_landmarks.py
    python inference_landmarks.py --camera 1
    python inference_landmarks.py --no-speech
"""
import argparse
import collections
import json
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import tensorflow as tf

ROOT          = Path(__file__).resolve().parents[1]
MODEL_PATH    = ROOT / "checkpoints" / "best_landmark_model.keras"
NAMES_PATH    = ROOT / "checkpoints" / "class_names_landmarks.json"
MP_MODEL_PATH = ROOT / "checkpoints" / "hand_landmarker.task"
MP_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
                 "hand_landmarker/hand_landmarker/float16/latest/"
                 "hand_landmarker.task")

CONF_THRESHOLD = 0.80
SMOOTH_FRAMES  = 10
COOLDOWN_SEC   = 1.5
HAND_PADDING   = 20

COL_GREEN  = (0, 220, 0)
COL_ORANGE = (0, 165, 255)
COL_WHITE  = (255, 255, 255)
COL_BLACK  = (0, 0, 0)


def download_mp_model():
    if MP_MODEL_PATH.exists():
        return
    print("Downloading MediaPipe hand landmarker (~9 MB)...")
    MP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MP_MODEL_URL, MP_MODEL_PATH)
    print("Downloaded.")


def crop_hand(frame, landmarks, h, w):
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


def preprocess_for_model(crop):
    """Convert hand crop to 28x28 grayscale matching training format.

    Training images have a DARK background with a BRIGHT hand.
    If the webcam crop has a light background, we invert it so the
    hand becomes the bright element, matching what the model expects.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Histogram equalization improves contrast in varied lighting
    gray = cv2.equalizeHist(gray)

    resized = cv2.resize(gray, (28, 28))

    # If background is brighter than hand center, invert
    # (check corners as a proxy for background brightness)
    corners = [resized[0,0], resized[0,-1], resized[-1,0], resized[-1,-1]]
    center  = resized[10:18, 10:18].mean()
    bg_mean = np.mean(corners)
    if bg_mean > center:
        resized = 255 - resized

    norm = resized.astype(np.float32) / 255.0
    return norm.reshape(1, 28, 28, 1)


def draw_landmarks(frame, hand_landmarks_list):
    h, w = frame.shape[:2]
    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]
    for hand_lms in hand_landmarks_list:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, COL_GREEN, -1)


def draw_ui(frame, letter, confidence, text_buffer, fps, hand_box, confirmed):
    h, w = frame.shape[:2]
    if hand_box:
        x1, y1, x2, y2 = hand_box
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      COL_GREEN if confirmed else COL_ORANGE, 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 80), (w, h), COL_BLACK, -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    if letter:
        cv2.putText(frame, f"{letter}  {confidence*100:.0f}%", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                    COL_GREEN if confirmed else COL_ORANGE, 3)
    cv2.putText(frame, "Text: " + (text_buffer or "_"), (10, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_WHITE, 2)
    cv2.putText(frame, "SPACE=space  BKSP=delete  ENTER=speak  C=clear  Q=quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
    cv2.putText(frame, f"FPS {fps:.0f}", (w - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)
    return frame


def init_tts():
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        return engine
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-speech", action="store_true")
    args = parser.parse_args()

    download_mp_model()
    print("Loading model...")
    model       = tf.keras.models.load_model(MODEL_PATH)
    class_names = json.loads(NAMES_PATH.read_text())
    print(f"Loaded. Classes: {class_names}")

    tts_engine = None if args.no_speech else init_tts()

    base_options = mp_python.BaseOptions(model_asset_path=str(MP_MODEL_PATH))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

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
            draw_landmarks(frame, detection.hand_landmarks)
            crop, hand_box = crop_hand(
                frame, detection.hand_landmarks[0], h, w
            )
            if crop.size > 0:
                inp   = preprocess_for_model(crop)

                # Debug window — shows exactly what the model receives
                debug = (inp[0, :, :, 0] * 255).astype(np.uint8)
                debug_big = cv2.resize(debug, (224, 224),
                                       interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Model input (28x28)", debug_big)

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

        frame = draw_ui(frame, letter, conf, text_buffer,
                        fps, hand_box, confirmed)
        cv2.imshow("ISL Translator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            text_buffer += " "
        elif key == 8:
            text_buffer = text_buffer[:-1]
        elif key == 13:
            if tts_engine and text_buffer.strip():
                tts_engine.say(text_buffer)
                tts_engine.runAndWait()
        elif key == ord("c"):
            text_buffer = ""

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()