import pandas as pd

# Input and output paths
input_parquet = "data/vector/cost_stats_vec_cleaned_min_dist.parquet"
output_feather = "data/vector/cost_stats_vec_cleaned_min_dist.feather"

# Read parquet
df = pd.read_parquet(input_parquet)

# Write to feather
df.to_feather(output_feather)

print(f"Converted {input_parquet} → {output_feather}")
