# predict_video.py
import cv2
import numpy as np
import joblib
import tensorflow as tf
import mediapipe as mp
import os
import urllib.request
from datetime import datetime
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from feature_extractor import extract_features

# ── PATHS ────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(PROJECT_DIR, "input")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
MODEL_PATH = os.path.join(PROJECT_DIR, "eye_emotion_model_v1.h5")
SCALER_PATH = os.path.join(PROJECT_DIR, "data", "eyet4empathy", "processed", "robust_scaler.pkl")
LANDMARKER_PATH = os.path.join(PROJECT_DIR, "models", "face_landmarker.task")
# ─────────────────────────────────────────────────────────────────────────────

SEQ_LEN    = 100
N_FEATURES = 38
SAVE_OUTPUT_VIDEO = False
HAPPY_ONLY_OUTPUT = True

# Change these names to match your trained model classes.
CLASS_NAMES = [
    "Neutral",
    "Calm",
    "Engaged",
    "High Arousal",
]

# Teacher-facing interpretation labels (derived view on top of 4-class model output).
EMOTION5_NAMES = ["happy", "anxious", "neutral", "sad", "angry"]


def map_to_emotion5(model_idx, conf):
    # Keep model output unchanged; provide an interpreted 5-emotion layer for reporting.
    if model_idx == 0:
        return "neutral"
    if model_idx == 1:
        return "sad"
    if model_idx == 2:
        return "happy" if conf >= 0.65 else "neutral"
    # model_idx == 3
    return "angry" if conf >= 0.75 else "anxious"

# Keras 3 removed `renorm*` args from BatchNormalization; older .h5 models may still contain them.
class CompatBatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self, *args, renorm=None, renorm_clipping=None, renorm_momentum=None, **kwargs):
        super().__init__(*args, **kwargs)

class CompatDense(tf.keras.layers.Dense):
    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)

class CompatInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, *args, optional=None, batch_shape=None, **kwargs):
        if batch_shape is not None and "shape" not in kwargs and len(batch_shape) > 1:
            kwargs["shape"] = tuple(batch_shape[1:])
        super().__init__(*args, **kwargs)

# ── Load model & scaler ──────────────────────────────────────────────────────
print("Loading model...")
model  = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "BatchNormalization": CompatBatchNormalization,
        "Dense": CompatDense,
        "InputLayer": CompatInputLayer,
    },
    compile=False,
)
print("Loading scaler...")
scaler = joblib.load(SCALER_PATH)
print("Model and scaler loaded")

# ── MediaPipe (Face Landmarker, new API) ────────────────────────────────────
if not os.path.exists(LANDMARKER_PATH):
    os.makedirs(os.path.dirname(LANDMARKER_PATH), exist_ok=True)
    print("Downloading MediaPipe face landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        LANDMARKER_PATH,
    )
    print(f"Downloaded: {LANDMARKER_PATH}")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def list_input_videos(input_dir):
    exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
    candidates = []
    for name in os.listdir(input_dir):
        full = os.path.join(input_dir, name)
        if os.path.isfile(full) and os.path.splitext(name)[1].lower() in exts:
            candidates.append(full)
    candidates.sort(key=lambda p: os.path.basename(p).lower())
    return candidates

def select_input_video(input_dir):
    videos = list_input_videos(input_dir)
    if not videos:
        print(f"No video files found in: {input_dir}")
        print("Add a file like input\\demo.mp4 and run again.")
        raise SystemExit(1)

    print("\nAvailable input videos:")
    for i, path in enumerate(videos, start=1):
        print(f"  {i}. {os.path.basename(path)}")

    while True:
        choice = input("Select video number: ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(videos):
                return videos[idx - 1]
        print("Invalid choice. Enter one of the listed numbers.")


def build_output_path(video_path, output_dir):
    base = os.path.splitext(os.path.basename(video_path))[0]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"{base}_pred_{stamp}.mp4")

base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

# ── Video setup ──────────────────────────────────────────────────────────────
VIDEO_PATH = select_input_video(INPUT_DIR)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Could not open video: {VIDEO_PATH}")
    print("Place a video in input\\ and try again.")
    exit()

fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Input video: {VIDEO_PATH}")
print(f"Video: {width}x{height} @ {fps}fps | {total} frames")
OUTPUT_PATH = build_output_path(VIDEO_PATH, OUTPUT_DIR)
print(f"Output video: {OUTPUT_PATH}")

out = None
if SAVE_OUTPUT_VIDEO:
    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

# ── State ────────────────────────────────────────────────────────────────────
buffer       = deque(maxlen=SEQ_LEN)
prev_gaze    = None
current_pred = 0
confidence   = 0.0
frame_count  = 0
class_counts = {k: 0 for k in range(len(CLASS_NAMES))}
prob_sums = np.zeros(len(CLASS_NAMES), dtype=np.float64)
emotion5_counts = {name: 0 for name in EMOTION5_NAMES}
num_predictions = 0

print("Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int((frame_count / max(fps, 1)) * 1000)
    results = face_landmarker.detect_for_video(mp_image, timestamp_ms)

    face_detected = False

    if results.face_landmarks:
        face_detected = True
        face_lm = results.face_landmarks[0]
        feats   = extract_features(face_lm, w, h)

        # velocity
        gaze_now = feats[:2].copy()
        if prev_gaze is not None:
            feats[36] = float(np.linalg.norm(gaze_now - prev_gaze))
        prev_gaze = gaze_now

        buffer.append(feats)

        # predict every 10 frames once buffer is full
        if len(buffer) == SEQ_LEN and frame_count % 10 == 0:
            seq        = np.array(buffer, dtype='float32')
            seq_scaled = scaler.transform(seq)
            seq_input  = seq_scaled[np.newaxis, ...]
            probs         = model.predict(seq_input, verbose=0)[0]
            current_pred  = int(np.argmax(probs))
            confidence    = float(probs[current_pred])
            class_counts[current_pred] += 1
            prob_sums += probs
            emotion5_counts[map_to_emotion5(current_pred, confidence)] += 1
            num_predictions += 1
            if num_predictions % 20 == 0:
                top2 = np.argsort(probs)[-2:][::-1]
                print(
                    f"  pred#{num_predictions:4d} frame={frame_count:5d} "
                    f"top1={CLASS_NAMES[top2[0]]}:{probs[top2[0]]:.3f} "
                    f"top2={CLASS_NAMES[top2[1]]}:{probs[top2[1]]:.3f}"
                )

    # Live overlay and saved video annotation
    current_label = map_to_emotion5(current_pred, confidence) if HAPPY_ONLY_OUTPUT else CLASS_NAMES[current_pred]
    shown_conf = min(max(confidence * 100.0, 0.0), 99.9)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (560, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(
        frame,
        f"Emotion: {current_label}",
        (15, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Confidence: {shown_conf:.1f}%",
        (15, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    if not face_detected:
        cv2.putText(
            frame,
            "No face detected",
            (15, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 100, 255),
            2,
        )

    if out is not None:
        out.write(frame)

    cv2.imshow("Emotion Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Stopped early by user.")
        break

    frame_count += 1
    if frame_count % 100 == 0:
        print(f"  {frame_count}/{total} frames done...")

# ── Cleanup ──────────────────────────────────────────────────────────────────
cap.release()
if out is not None:
    out.release()
face_landmarker.close()
cv2.destroyAllWindows()

print("\nDone!")
print(f"Frames processed: {frame_count}")
print(f"Prediction windows: {num_predictions}")
if num_predictions > 0:
    print("\nInterpreted 5-emotion summary:")
    for label in EMOTION5_NAMES:
        pct = (emotion5_counts[label] / max(num_predictions, 1)) * 100.0
        print(f"  {label:12s} {pct:6.2f}%")