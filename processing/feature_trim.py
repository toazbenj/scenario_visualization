import pandas as pd

def keep_columns_from_csv(input_file, output_file, columns_to_keep):
    """
    Keeps only the specified columns from a CSV file and writes the result to a new file.

    Parameters:
    - input_file (str): Path to the original CSV file.
    - output_file (str): Path to save the new CSV file.
    - columns_to_keep (list): List of column names to keep.
    """
    try:
        # Load the CSV
        df = pd.read_csv(input_file)

        # Keep only specified columns (ignore missing ones safely)
        available_cols = [col for col in columns_to_keep if col in df.columns]
        df = df[available_cols]

        # Save to new CSV
        df.to_csv(output_file, index=False)

        print(f"Saved new CSV with only columns {available_cols} to '{output_file}'")

    except Exception as e:
        print(f"Error: {e}")

# Example usage
input_csv = "data/vector/race_stats_truncated.csv"
output_csv = "data/vector/race_stats_distance.csv"
# columns_to_keep = ["Out of Bounds P2","Collisions", "Passes P2",
#                    'Spawn distance_x', 'Spawn distance_y'] 

columns_to_keep = ["Out of Bounds P2",'Spawn distance_x', 'Spawn distance_y'] 
keep_columns_from_csv(input_csv, output_csv, columns_to_keep)
