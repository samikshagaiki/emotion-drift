# Emotion Drift Project

Current working setup uses a 5-label CNN+LSTM hybrid pipeline and terminal prediction summary.

## Labels (true5 pipeline)

- `0`: happy
- `1`: anxious
- `2`: neutral
- `3`: sad
- `4`: angry

## Required files (current pipeline)

- `data/eyet4empathy/processed/X_train.npy`
- `data/eyet4empathy/processed/groups.npy`
- `data/eyet4empathy/processed/robust_scaler.pkl`
- `data/eyet4empathy/processed/y_train_true5.npy`
- `training_reports/true5_thresholds.json`
- `eye_emotion_model_true5_hybrid.h5`

## Environment setup

Run from project root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\emotion_env\Scripts\Activate.ps1
pip install -U pip
pip install opencv-python numpy pandas mediapipe tensorflow joblib scikit-learn
```

## Pipeline followed till now

### 1) Build true5 labels from feature sequences

```powershell
.\emotion_env\Scripts\python.exe .\src\pipeline\build_true5_labels_from_features.py
```

Output:
- `data/eyet4empathy/processed/y_train_true5.npy`
- `training_reports/true5_thresholds.json`

### 2) Train hybrid CNN+LSTM model

```powershell
.\emotion_env\Scripts\python.exe .\src\pipeline\train_true5_hybrid.py
```

Output:
- `eye_emotion_model_true5_hybrid.keras`
- `eye_emotion_model_true5_hybrid.h5`
- `training_reports/true5_hybrid_metrics.json`
- `training_reports/true5_hybrid_confusion_matrix.csv`

### 3) Predict from video (terminal summary)

```powershell
.\emotion_env\Scripts\python.exe .\src\annotation\predict_video_true5.py
```

It asks for video number from `input/` and prints:
- per-label percentage
- dominant label

## Fast run commands

- `sammy-1.mp4` -> choose `3`
- `Sammy-2.mp4` -> choose `4`
- `test_video.mp4` -> choose `5`

## Notes

- `predict_video.py` is the legacy 4-class script.
- `src/annotation/predict_video_true5.py` is the current true5 output script.
