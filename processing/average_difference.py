import pandas as pd
import numpy as np

# === CONFIGURATION ===
file1 = "data/vector/race_stats.csv"
file2 = "data/scalar/race_stats.csv"
output_file = "data/comparison/collision_difference.csv"

# Input feature columns
x_feat = "Theta_a2"
y_feat = "Theta_b2"
z_feat = "Theta_c2"
score_feat = "Collisions"
operation = "subtract"  # Options: 'subtract', 'ratio', 'absolute_diff'

# Custom output column names
x_title = "Theta A"
y_title = "Theta B"
z_title = "Theta C"
score_title = "Collision Percentage Difference"

# === LOAD DATA ===
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# === AVERAGE BY POINT COORDINATES ===
group_cols = [x_feat, y_feat, z_feat]

avg1 = df1.groupby(group_cols, as_index=False)[score_feat].mean()
avg2 = df2.groupby(group_cols, as_index=False)[score_feat].mean()

# === MERGE AVERAGED DATASETS ON COORDINATES ===
merged = pd.merge(avg1, avg2, on=group_cols, suffixes=('_1', '_2'))

# === COMPUTE DIFFERENCE ===
if operation == "subtract":
    merged["score_diff"] = merged[f"{score_feat}_1"] - merged[f"{score_feat}_2"]
elif operation == "ratio":
    merged["score_diff"] = np.where(
        merged[f"{score_feat}_2"] != 0,
        merged[f"{score_feat}_1"] / merged[f"{score_feat}_2"],
        np.nan
    )
elif operation == "absolute_diff":
    merged["score_diff"] = np.abs(merged[f"{score_feat}_1"] - merged[f"{score_feat}_2"])
else:
    raise ValueError(f"Unknown operation: {operation}")

merged["score_diff"] = merged["score_diff"] * 100

# === RENAME COLUMNS FOR OUTPUT ===
merged.rename(columns={
    x_feat: x_title,
    y_feat: y_title,
    z_feat: z_title,
    "score_diff": score_title
}, inplace=True)


# === SAVE TO CSV ===
output_columns = [x_title, y_title, z_title, score_title]
merged.to_csv(output_file, index=False)

print(f"Saved averaged differences to: {output_file}")

