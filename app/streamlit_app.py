"""ISL Translator — Streamlit demo app.

Run from the project root:
    streamlit run app/streamlit_app.py

Requires the trained model at checkpoints/best_landmark_model.keras
and checkpoints/class_names_landmarks.json
"""
import collections
import json
import sys
import time
import urllib.request
from pathlib import Path

# Allow importing contribute_tab from the same folder
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import streamlit as st
import tensorflow as tf

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[1]
MODEL_PATH    = ROOT / "checkpoints" / "best_model.keras"
NAMES_PATH    = ROOT / "checkpoints" / "class_names.json"
MP_MODEL_PATH = ROOT / "checkpoints" / "hand_landmarker.task"
MP_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
                 "hand_landmarker/hand_landmarker/float16/latest/"
                 "hand_landmarker.task")

# ── Inference settings ───────────────────────────────────────────────────────
CONF_THRESHOLD = 0.80
SMOOTH_FRAMES  = 10
COOLDOWN_SEC   = 1.5
HAND_PADDING   = 20

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISL Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.letter-display {
    font-family: 'Space Mono', monospace;
    font-size: 7rem;
    font-weight: 700;
    line-height: 1;
    color: #00e5a0;
    text-align: center;
    padding: 0.5rem 0;
    letter-spacing: -4px;
}
.conf-display {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    color: #888;
    text-align: center;
    margin-top: -0.5rem;
}
.text-buffer {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    background: #0e0e0e;
    border: 1px solid #222;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    min-height: 60px;
    color: #f0f0f0;
    word-break: break-all;
    letter-spacing: 2px;
}
.status-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
}
.pill-confirmed { background: #00e5a020; color: #00e5a0; border: 1px solid #00e5a040; }
.pill-waiting   { background: #ffffff10; color: #888;    border: 1px solid #ffffff20; }
.pill-nohand    { background: #ff444420; color: #ff6666; border: 1px solid #ff444440; }
section[data-testid="stSidebar"] { background: #0a0a0a; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def download_mp_model():
    if MP_MODEL_PATH.exists():
        return
    with st.spinner("Downloading MediaPipe hand model (~9 MB)..."):
        MP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MP_MODEL_URL, MP_MODEL_PATH)


@st.cache_resource
def load_resources():
    download_mp_model()
    model       = tf.keras.models.load_model(MODEL_PATH)
    class_names = json.loads(NAMES_PATH.read_text())
    base_options = mp_python.BaseOptions(model_asset_path=str(MP_MODEL_PATH))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)
    return model, class_names, landmarker


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


def preprocess(crop):
    img = cv2.resize(crop, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img.astype(np.float32), axis=0)


def draw_skeleton(frame, hand_landmarks_list):
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
            cv2.line(frame, pts[a], pts[b], (0, 200, 120), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 230, 160), -1)


# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "text_buffer":       "",
        "pred_buffer":       collections.deque(maxlen=SMOOTH_FRAMES),
        "last_letter":       "",
        "last_time":         0.0,
        "current_letter":    "",
        "current_conf":      0.0,
        "confirmed":         False,
        "running":           False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    # Keep pred_buffer maxlen in sync with slider value
    if len(st.session_state.pred_buffer) == 0:
        st.session_state.pred_buffer = collections.deque(maxlen=SMOOTH_FRAMES)

init_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤟 ISL Translator")
    st.markdown("---")

    conf_thresh = st.slider(
        "Confidence threshold", 0.60, 0.99, CONF_THRESHOLD, 0.01,
        help="Minimum confidence before a letter is shown"
    )
    smooth = st.slider(
        "Smoothing frames", 3, 20, SMOOTH_FRAMES, 1,
        help="Consecutive matching frames needed to confirm a letter"
    )
    cooldown = st.slider(
        "Cooldown (s)", 0.5, 3.0, COOLDOWN_SEC, 0.5,
        help="Seconds before the same letter can be added again"
    )
    show_debug = st.checkbox("Show 28×28 model input", value=False)

    st.markdown("---")
    st.markdown("**Controls**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⌫ Delete", use_container_width=True):
            st.session_state.text_buffer = st.session_state.text_buffer[:-1]
    with col2:
        if st.button("Space", use_container_width=True):
            st.session_state.text_buffer += " "
    if st.button("🗑 Clear text", use_container_width=True):
        st.session_state.text_buffer = ""

    st.markdown("---")
    tts_enabled = st.checkbox("Text-to-speech", value=True)
    if st.button("🔊 Speak", use_container_width=True):
        if tts_enabled and st.session_state.text_buffer.strip():
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(st.session_state.text_buffer)
                engine.runAndWait()
            except Exception:
                st.warning("pyttsx3 not available")

    st.markdown("---")
    st.caption("Indian Sign Language — alphabet classifier")
    st.caption(f"Model: `{MODEL_PATH.name}`")

# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("# ISL Translator")
tab_translator, tab_contribute = st.tabs(["🤟 Translator", "📸 Contribute"])

with tab_contribute:
    from contribute_tab import render_contribute_tab
    render_contribute_tab()

with tab_translator:
    col_feed, col_panel = st.columns([3, 2], gap="large")

    with col_panel:
        st.markdown("#### Detected sign")
        letter_slot = st.empty()
        conf_slot   = st.empty()
        status_slot = st.empty()

        st.markdown("#### Text")
        buffer_slot = st.empty()

        if show_debug:
            st.markdown("#### Model input (28×28)")
            debug_slot = st.empty()

    with col_feed:
        feed_slot   = st.empty()
        start_col, stop_col = st.columns(2)
        with start_col:
            start_btn = st.button("▶ Start", use_container_width=True, type="primary")
        with stop_col:
            stop_btn  = st.button("⏹ Stop", use_container_width=True)

        if start_btn:
            st.session_state.running = True
        if stop_btn:
            st.session_state.running = False

# ── Inference loop ────────────────────────────────────────────────────────────
if st.session_state.running:
    try:
        model, class_names, landmarker = load_resources()
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam. Try a different camera index.")
        st.stop()

    pred_buffer = st.session_state.pred_buffer

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        h, w   = frame.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        det    = landmarker.detect(mp_img)

        letter    = None
        conf      = 0.0
        confirmed = False
        debug_img = None

        if det.hand_landmarks:
            draw_skeleton(frame, det.hand_landmarks)
            crop, box = crop_hand(frame, det.hand_landmarks[0], h, w)

            if crop.size > 0:
                inp   = preprocess(crop)
                preds = model.predict(inp, verbose=0)[0]
                idx   = int(np.argmax(preds))
                conf  = float(preds[idx])
                letter = class_names[idx] if conf >= conf_thresh else None

                pred_buffer.append(letter)

                if (len(pred_buffer) == smooth
                        and len(set(pred_buffer)) == 1
                        and letter is not None):
                    confirmed = True
                    now = time.time()
                    if (letter != st.session_state.last_letter
                            or now - st.session_state.last_time > cooldown):
                        st.session_state.text_buffer += letter
                        st.session_state.last_letter  = letter
                        st.session_state.last_time    = now

                if show_debug:
                    debug_img = (inp[0, :, :, 0] * 255).astype(np.uint8)

            # Draw bounding box
            _, (x1, y1, x2, y2) = crop_hand(frame, det.hand_landmarks[0], h, w)
            color = (0, 230, 160) if confirmed else (255, 140, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        else:
            pred_buffer.clear()

        # Update UI
        feed_slot.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB", use_container_width=True
        )

        if letter:
            letter_slot.markdown(
                f'<div class="letter-display">{letter}</div>',
                unsafe_allow_html=True
            )
            conf_slot.markdown(
                f'<div class="conf-display">{conf*100:.0f}% confidence</div>',
                unsafe_allow_html=True
            )
        else:
            letter_slot.markdown(
                '<div class="letter-display" style="color:#333">—</div>',
                unsafe_allow_html=True
            )
            conf_slot.markdown("", unsafe_allow_html=True)

        if confirmed:
            status_slot.markdown(
                '<span class="status-pill pill-confirmed">✓ confirmed</span>',
                unsafe_allow_html=True
            )
        elif det.hand_landmarks:
            status_slot.markdown(
                '<span class="status-pill pill-waiting">· reading</span>',
                unsafe_allow_html=True
            )
        else:
            status_slot.markdown(
                '<span class="status-pill pill-nohand">no hand detected</span>',
                unsafe_allow_html=True
            )

        buf = st.session_state.text_buffer
        buffer_slot.markdown(
            f'<div class="text-buffer">{buf if buf else "&nbsp;"}</div>',
            unsafe_allow_html=True
        )

        if show_debug and debug_img is not None:
            debug_slot.image(
                cv2.resize(debug_img, (140, 140), interpolation=cv2.INTER_NEAREST),
                clamp=True
            )

    cap.release()
else:
    # Idle state
    letter_slot.markdown(
        '<div class="letter-display" style="color:#222">🤟</div>',
        unsafe_allow_html=True
    )
    buf = st.session_state.text_buffer
    buffer_slot.markdown(
        f'<div class="text-buffer">{buf if buf else "&nbsp;"}</div>',
        unsafe_allow_html=True
    )
    feed_slot.info("Press **▶ Start** to begin signing.")