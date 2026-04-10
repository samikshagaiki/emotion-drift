import json
import os
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "eyet4empathy", "processed")
REPORT_DIR = os.path.join(PROJECT_DIR, "training_reports")

X_PATH = os.path.join(DATA_DIR, "X_train.npy")
Y_PATH = os.path.join(DATA_DIR, "y_train_true5.npy")
GROUPS_PATH = os.path.join(DATA_DIR, "groups.npy")

MODEL_KERAS = os.path.join(PROJECT_DIR, "eye_emotion_model_true5_hybrid.keras")
MODEL_H5 = os.path.join(PROJECT_DIR, "eye_emotion_model_true5_hybrid.h5")
METRICS_JSON = os.path.join(REPORT_DIR, "true5_hybrid_metrics.json")
CM_CSV = os.path.join(REPORT_DIR, "true5_hybrid_confusion_matrix.csv")

NUM_CLASSES = 5

os.makedirs(REPORT_DIR, exist_ok=True)


def build_model(input_shape):
    inp = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.30)(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(96, return_sequences=True))(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def main():
    if not os.path.exists(Y_PATH):
        raise FileNotFoundError(f"Missing labels file: {Y_PATH}\nRun build_true5_labels_from_features.py first.")

    X = np.load(X_PATH, mmap_mode="r")
    y = np.load(Y_PATH)
    groups = np.load(GROUPS_PATH)

    n = min(len(X), len(y), len(groups))
    X = X[:n]
    y = y[:n]
    groups = groups[:n]
    groups_1d = np.array([f"{g[0]}_{g[1]}" for g in groups])

    dist = Counter(y.tolist())
    print("Label distribution:", dict(sorted(dist.items())))

    gss_test = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss_test.split(X, y, groups_1d))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_train = groups_1d[train_idx]

    gss_val = GroupShuffleSplit(test_size=0.2, random_state=42)
    tr2_idx, val_idx = next(gss_val.split(X_train, y_train, g_train))
    X_tr, X_val = X_train[tr2_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr2_idx], y_train[val_idx]

    classes = np.arange(NUM_CLASSES, dtype=np.int64)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

    print("Train/Val/Test:", X_tr.shape, X_val.shape, X_test.shape)
    print("Class weights:", class_weight)

    model = build_model((X_tr.shape[1], X_tr.shape[2]))
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_KERAS, monitor="val_loss", save_best_only=True, mode="min", verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ]

    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    report = classification_report(y_test, y_pred, labels=list(range(NUM_CLASSES)), output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))

    np.savetxt(CM_CSV, cm, fmt="%d", delimiter=",")
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_loss": float(test_loss),
                "test_accuracy": float(test_acc),
                "distribution": dict(sorted(dist.items())),
                "class_weight": class_weight,
                "classification_report": report,
                "history": {k: [float(x) for x in v] for k, v in history.history.items()},
            },
            f,
            indent=2,
        )

    model.save(MODEL_H5)
    print("Saved:", MODEL_KERAS)
    print("Saved:", MODEL_H5)
    print("Saved:", METRICS_JSON)
    print("Saved:", CM_CSV)
    print("Final test accuracy:", float(test_acc))
    print("Macro F1:", float(report["macro avg"]["f1-score"]))


if __name__ == "__main__":
    main()
