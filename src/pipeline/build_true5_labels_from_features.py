import json
import os
from collections import Counter

import numpy as np


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "eyet4empathy", "processed")
REPORT_DIR = os.path.join(PROJECT_DIR, "training_reports")

X_PATH = os.path.join(DATA_DIR, "X_train.npy")
Y_OUT = os.path.join(DATA_DIR, "y_train_true5.npy")
THRESH_OUT = os.path.join(REPORT_DIR, "true5_thresholds.json")

os.makedirs(REPORT_DIR, exist_ok=True)


def main():
    X = np.load(X_PATH, mmap_mode="r")
    n = len(X)
    print("Loaded X:", X.shape)

    # Sequence-level stats from engineered features.
    vel = np.mean(X[:, :, 36], axis=1)  # velocity channel
    pup = np.mean(X[:, :, 37], axis=1)  # pupil proxy channel
    gaze_jitter = np.std(X[:, :, 0], axis=1) + np.std(X[:, :, 1], axis=1)

    thr = {
        "vel_p20": float(np.quantile(vel, 0.20)),
        "vel_p45": float(np.quantile(vel, 0.45)),
        "vel_p70": float(np.quantile(vel, 0.70)),
        "vel_p85": float(np.quantile(vel, 0.85)),
        "pup_p30": float(np.quantile(pup, 0.30)),
        "pup_p55": float(np.quantile(pup, 0.55)),
        "pup_p80": float(np.quantile(pup, 0.80)),
        "jit_p55": float(np.quantile(gaze_jitter, 0.55)),
        "jit_p80": float(np.quantile(gaze_jitter, 0.80)),
    }

    # Class IDs (requested order):
    # 0 happy, 1 anxious, 2 neutral, 3 sad, 4 angry
    y = np.full(n, 2, dtype=np.int64)  # default neutral

    angry = (vel >= thr["vel_p85"]) & (pup >= thr["pup_p80"])
    anxious = (vel >= thr["vel_p70"]) & (gaze_jitter >= thr["jit_p80"]) & ~angry
    sad = (vel <= thr["vel_p20"]) & (pup <= thr["pup_p30"])
    happy = (
        (vel >= thr["vel_p45"])
        & (vel < thr["vel_p85"])
        & (pup >= thr["pup_p55"])
        & (gaze_jitter <= thr["jit_p80"])
        & ~anxious
        & ~angry
    )

    y[sad] = 3
    y[happy] = 0
    y[anxious] = 1
    y[angry] = 4

    dist = Counter(y.tolist())
    pct = {int(k): float(v * 100.0 / n) for k, v in sorted(dist.items())}

    np.save(Y_OUT, y)
    with open(THRESH_OUT, "w", encoding="utf-8") as f:
        json.dump({"thresholds": thr, "distribution": dict(sorted(dist.items())), "percent": pct}, f, indent=2)

    print("Saved labels:", Y_OUT)
    print("Saved thresholds:", THRESH_OUT)
    print("Distribution:", dict(sorted(dist.items())))
    print("Percent:", pct)


if __name__ == "__main__":
    main()
