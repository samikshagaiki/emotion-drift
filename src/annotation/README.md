# Annotation Pipeline (True Labels)

This pipeline creates true-label training data and trains a 5-class model:

- `0 = happy`
- `1 = anxious`
- `2 = neutral`
- `3 = sad`
- `4 = angry`

## Steps

1. Generate template:

```powershell
.\emotion_env\Scripts\python.exe .\src\annotation\create_annotation_template.py
```

Template file:
- `data/annotations/annotations_template.csv`

2. Fill annotations:
- Copy template to `data/annotations/annotations.csv`
- Add multiple rows per recording with:
  - `participant_name`
  - `recording_name`
  - `start_timestamp_ms`
  - `end_timestamp_ms`
  - `label` (happy/anxious/neutral/sad/angry)

3. Build annotated dataset:

```powershell
.\emotion_env\Scripts\python.exe .\src\annotation\build_annotated_dataset.py
```

Outputs:
- `data/eyet4empathy/processed/X_train_annot.npy`
- `data/eyet4empathy/processed/y_train_annot.npy`
- `data/eyet4empathy/processed/groups_annot.npy`
- `data/eyet4empathy/processed/robust_scaler_annot.pkl`

4. Train true-label model:

```powershell
.\emotion_env\Scripts\python.exe .\src\annotation\train_annotated_model.py
```

Outputs:
- `eye_emotion_model_true5.keras`
- `eye_emotion_model_true5.h5`
- `training_reports/true5_metrics.json`
- `training_reports/true5_confusion_matrix.csv`
