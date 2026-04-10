import os
import pandas as pd


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "eyet4empathy", "processed")
ANNOT_DIR = os.path.join(PROJECT_DIR, "data", "annotations")

EYE_CLEAN_PATH = os.path.join(DATA_DIR, "eye_clean.csv")
OUT_TEMPLATE_PATH = os.path.join(ANNOT_DIR, "annotations_template.csv")


def main():
    if not os.path.exists(EYE_CLEAN_PATH):
        raise FileNotFoundError(f"Missing file: {EYE_CLEAN_PATH}")

    os.makedirs(ANNOT_DIR, exist_ok=True)

    use_cols = ["Participant name", "Recording name", "Recording timestamp"]
    df = pd.read_csv(EYE_CLEAN_PATH, usecols=use_cols, low_memory=False)
    df["Recording timestamp"] = pd.to_numeric(df["Recording timestamp"], errors="coerce")
    df = df.dropna(subset=["Recording timestamp"])

    grouped = (
        df.groupby(["Participant name", "Recording name"])["Recording timestamp"]
        .agg(["min", "max"])
        .reset_index()
        .rename(
            columns={
                "Participant name": "participant_name",
                "Recording name": "recording_name",
                "min": "start_timestamp_ms",
                "max": "end_timestamp_ms",
            }
        )
    )

    grouped["label"] = ""
    grouped["notes"] = ""
    grouped = grouped.sort_values(["participant_name", "recording_name"]).reset_index(drop=True)

    grouped.to_csv(OUT_TEMPLATE_PATH, index=False)
    print(f"Template saved: {OUT_TEMPLATE_PATH}")
    print(f"Rows: {len(grouped)}")
    print("Fill multiple rows per recording to annotate ranges with labels: happy/anxious/neutral/sad/angry")


if __name__ == "__main__":
    main()
