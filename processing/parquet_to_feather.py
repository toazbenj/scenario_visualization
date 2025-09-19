import pandas as pd

# Input and output paths
input_parquet = "combined_stats_vec_cleaned.parquet"
output_feather = "combined_stats_vec_cleaned.feather"

# Read parquet
df = pd.read_parquet(input_parquet)

# Write to feather
df.to_feather(output_feather)

print(f"Converted {input_parquet} → {output_feather}")
