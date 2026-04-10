import json
import os
from collections import Counter, defaultdict

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "eyet4empathy", "processed")
ANNOT_DIR = os.path.join(PROJECT_DIR, "data", "annotations")
REPORT_DIR = os.path.join(PROJECT_DIR, "training_reports")
SCHEMA_PATH = os.path.join(PROJECT_DIR, "src", "annotation", "label_schema.json")

EYE_CLEAN_PATH = os.path.join(DATA_DIR, "eye_clean.csv")
ANNOT_PATH = os.path.join(ANNOT_DIR, "annotations.csv")

X_OUT = os.path.join(DATA_DIR, "X_train_annot.npy")
Y_OUT = os.path.join(DATA_DIR, "y_train_annot.npy")
GROUPS_OUT = os.path.join(DATA_DIR, "groups_annot.npy")
SCALER_OUT = os.path.join(DATA_DIR, "robust_scaler_annot.pkl")
REPORT_OUT = os.path.join(REPORT_DIR, "annot_dataset_stats.json")

SEQ_LEN = 100
STEP = 50
REQUIRED_ANNOT_COLS = {
    "participant_name",
    "recording_name",
    "start_timestamp_ms",
    "end_timestamp_ms",
    "label",
}


def load_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    label_to_id = {c["name"].strip().lower(): int(c["id"]) for c in data["classes"]}
    return label_to_id


def load_annotations(label_to_id):
    if not os.path.exists(ANNOT_PATH):
        raise FileNotFoundError(
            f"Missing annotations file: {ANNOT_PATH}\n"
            "Create it from template and fill labels first."
        )

    ann = pd.read_csv(ANNOT_PATH)
    missing = REQUIRED_ANNOT_COLS - set(ann.columns)
    if missing:
        raise ValueError(f"Missing annotation columns: {sorted(missing)}")

    ann["label"] = ann["label"].astype(str).str.strip().str.lower()
    bad = sorted(set(ann["label"]) - set(label_to_id.keys()))
    if bad:
        raise ValueError(f"Unknown labels in annotations: {bad}")

    ann["class_id"] = ann["label"].map(label_to_id).astype(int)
    ann["start_timestamp_ms"] = pd.to_numeric(ann["start_timestamp_ms"], errors="coerce")
    ann["end_timestamp_ms"] = pd.to_numeric(ann["end_timestamp_ms"], errors="coerce")
    ann = ann.dropna(subset=["start_timestamp_ms", "end_timestamp_ms"])
    ann = ann[ann["end_timestamp_ms"] >= ann["start_timestamp_ms"]].copy()

    by_key = defaultdict(list)
    for _, r in ann.iterrows():
        key = (str(r["participant_name"]), str(r["recording_name"]))
        by_key[key].append((float(r["start_timestamp_ms"]), float(r["end_timestamp_ms"]), int(r["class_id"])))

    return by_key


def label_timestamp(ts, intervals):
    for s, e, cls in intervals:
        if s <= ts <= e:
            return cls
    return -1


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    label_to_id = load_schema()
    ann_by_key = load_annotations(label_to_id)

    use_df = pd.read_csv(EYE_CLEAN_PATH, low_memory=False)
    exclude = {"Participant name", "Recording name", "Recording timestamp", "label"}
    feature_cols = [c for c in use_df.columns if c not in exclude]
    del use_df

    df = pd.read_csv(EYE_CLEAN_PATH, low_memory=False)
    df["Recording timestamp"] = pd.to_numeric(df["Recording timestamp"], errors="coerce")
    df = df.dropna(subset=["Recording timestamp"])

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(method="ffill").fillna(method="bfill").fillna(0.0)

    scaler = RobustScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols].values.astype(np.float32))
    joblib.dump(scaler, SCALER_OUT)

    X, y, groups = [], [], []
    skipped_groups = 0

    grp = df.groupby(["Participant name", "Recording name"], sort=False)
    for (p, r), gdf in grp:
        key = (str(p), str(r))
        intervals = ann_by_key.get(key, [])
        if not intervals:
            skipped_groups += 1
            continue

        gdf = gdf.sort_values("Recording timestamp")
        ts_vals = gdf["Recording timestamp"].values.astype(np.float64)
        feat_vals = gdf[feature_cols].values.astype(np.float32)
        frame_labels = np.array([label_timestamp(ts, intervals) for ts in ts_vals], dtype=np.int32)

        if len(feat_vals) < SEQ_LEN:
            continue

        for i in range(0, len(feat_vals) - SEQ_LEN, STEP):
            lbl_slice = frame_labels[i : i + SEQ_LEN]
            valid = lbl_slice[lbl_slice >= 0]
            if len(valid) < max(5, int(SEQ_LEN * 0.3)):
                continue
            seq_label = int(np.bincount(valid).argmax())
            X.append(feat_vals[i : i + SEQ_LEN])
            y.append(seq_label)
            groups.append((p, r))

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int64)
    g_arr = np.array(groups)

    np.save(X_OUT, X_arr)
    np.save(Y_OUT, y_arr)
    np.save(GROUPS_OUT, g_arr)

    dist = Counter(y_arr.tolist())
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_sequences": int(len(y_arr)),
                "distribution": dict(sorted(dist.items())),
                "skipped_groups_without_annotations": int(skipped_groups),
                "feature_count": len(feature_cols),
                "seq_len": SEQ_LEN,
                "step": STEP,
            },
            f,
            indent=2,
        )

    print("Saved:", X_OUT)
    print("Saved:", Y_OUT)
    print("Saved:", GROUPS_OUT)
    print("Saved:", SCALER_OUT)
    print("Saved:", REPORT_OUT)
    print("Distribution:", dict(sorted(dist.items())))


if __name__ == "__main__":
    main()
