import pandas as pd
import numpy as np

def add_distance_columns(input_file, output_file,
                         x1_col='Initial X Position P1', y1_col='Initial Y Position P1',
                         x2_col='Initial X Position P2', y2_col='Initial Y Position P2',
                         dist_x_col='Spawn distance_x', dist_y_col='Spawn distance_y'):
    """
    Adds columns for x and y distances, and total Euclidean distance between (x1, y1) and (x2, y2).

    Parameters:
    - input_file (str): Path to input CSV file.
    - output_file (str): Path to output CSV file.
    - *_col (str): Column names for positions and distances.
    """
    try:
        df = pd.read_csv(input_file)

        df[dist_x_col] = df[x1_col] - df[x2_col]
        df[dist_y_col] = df[y1_col] - df[y2_col]

        df.to_csv(output_file, index=False)
        print(f"Added '{dist_x_col}', '{dist_y_col}' columns to '{output_file}'")

    except Exception as e:
        print(f"Error: {e}")

# Example usage
add_distance_columns("data/vector/race_stats.csv", "data/vector/race_stats_truncated.csv")
