import pandas as pd

def keep_columns(input_file, output_file, drop):
    """
    Keeps only the specified columns from a CSV file and writes the result to a new file.

    Parameters:
    - input_file (str): Path to the original CSV file.
    - output_file (str): Path to save the new CSV file.
    - columns_to_keep (list): List of column names to keep.
    """
    try:
        # Load the CSV
        df = pd.read_feather(input_file)

        # Keep only specified columns (ignore missing ones safely)
        # available_cols = [col for col in columns_to_keep if col in df.columns]
        # df = df[available_cols]

        df = df.drop(columns=drop, errors='ignore')

        # Save to new CSV
        df.to_feather(output_file)

        print(f"Saved new CSV with only columns to '{output_file}'")

    except Exception as e:
        print(f"Error: {e}")

# Example usage
input_csv = "c:\\Users\\toazb\Documents\\GitHub\\race_simulation\\data\\race_stats_vec.feather"
output_csv = "race_stats_vec_truncated.feather"
# columns_to_keep = ["Out of Bounds P2","Collisions", "Passes P2",
#                    'Spawn distance_x', 'Spawn distance_y'] 

# columns_to_keep = ["Out of Bounds P2",'Spawn distance_x', 'Spawn distance_y'] 
drop = ['Scenario']
keep_columns(input_csv, output_csv, drop)
