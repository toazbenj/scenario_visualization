import pandas as pd

# Input and output paths
input_parquet = "out/vector_take1/action2/shap/baseline.feather"
output_parquet_zstd = "out/vector_take1/action2/shap/baseline_zstd.parquet"

# Read parquet
df = pd.read_feather(input_parquet)

# Write to parquet with ZSTD compression
df.to_parquet(output_parquet_zstd, compression="zstd", index=False)

print(f"Converted {input_parquet} → {output_parquet_zstd} with ZSTD compression")
