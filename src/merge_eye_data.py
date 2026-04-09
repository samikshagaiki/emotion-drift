import pandas as pd
import os

folder = r"D:\ML_Project\emotion-drift-project\data\eyet4empathy\raw\eye_csvs"

dfs = []
for file in os.listdir(folder):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(folder, file))
        df["source_file"] = file
        dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)
merged.to_csv(
    r"D:\ML_Project\emotion-drift-project\data\eyet4empathy\processed\merged_eye.csv",
    index=False
)

print(merged.columns)
print("Total rows:", len(merged))
