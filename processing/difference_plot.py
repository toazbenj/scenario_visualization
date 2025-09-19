import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.join(os.path.dirname(__file__), 'scenario-visualization-toolkit-v1.0.4-dark'))
import utils

class Plot3DHandler:
    def __init__(self, dataset_df, label_column, dataset_df2=None, label_column2=None,
                 default_x_axis=None, default_y_axis=None, default_z_axis=None):
        """
        Initialize the 3D plotting handler
        
        Args:
            dataset_df: pandas DataFrame containing the first dataset
            label_column: string, name of the label/score column for first dataset
            dataset_df2: pandas DataFrame containing the second dataset (optional)
            label_column2: string, name of the label/score column for second dataset (optional)
            default_x_axis: string, default column name for x-axis (optional)
            default_y_axis: string, default column name for y-axis (optional)
            default_z_axis: string, default column name for z-axis (optional)
        """
        self.data = dataset_df
        self.label = label_column
        self.data2 = dataset_df2
        self.label2 = label_column2 if label_column2 else label_column
        self.output_dir = "output"  # Default output directory
        
        # Store default axis names
        self.default_x_axis = default_x_axis
        self.default_y_axis = default_y_axis
        self.default_z_axis = default_z_axis
    
    
    def compute_difference_dataset(self, x_feat=None, y_feat=None, z_feat=None, score_feat=None, 
                                merge_on=None, operation='subtract', summarize_similar_points=False):
        """
        Compute the difference between two datasets
        
        Args:
            x_feat: string, column name for x-axis (auto-detected if None)
            y_feat: string, column name for y-axis (auto-detected if None)
            z_feat: string, column name for z-axis (auto-detected if None)
            score_feat: string, column name for score (uses self.label if None)
            merge_on: list of columns to merge on (if None, assumes same row order)
            operation: string, 'subtract' or 'ratio' or 'absolute_diff'
            
        Returns:
            pandas DataFrame with difference data
        """
        if self.data2 is None:
            raise ValueError("Second dataset not loaded. Cannot compute difference.")
            
        if score_feat is None:
            score_feat = self.label
            
        # Auto-detect axis columns if not provided
        if x_feat is None or y_feat is None or z_feat is None:
            suggestions = self.suggest_axis_columns(1)
            x_feat = x_feat or suggestions['x']
            y_feat = y_feat or suggestions['y']
            z_feat = z_feat or suggestions['z']
            
            print(f"Auto-detected axis columns - X: {x_feat}, Y: {y_feat}, Z: {z_feat}")
            
        if merge_on is not None:
            # Merge datasets on specified columns
            merged_df = pd.merge(self.data, self.data2, on=merge_on, suffixes=('_1', '_2'))
            
            # Compute difference
            if operation == 'subtract':
                merged_df[f'{score_feat}_diff'] = merged_df[f'{score_feat}_1'] - merged_df[f'{score_feat}_2']
            elif operation == 'ratio':
                # Avoid division by zero
                merged_df[f'{score_feat}_diff'] = np.where(
                    merged_df[f'{score_feat}_2'] != 0,
                    merged_df[f'{score_feat}_1'] / merged_df[f'{score_feat}_2'],
                    np.nan
                )
            elif operation == 'absolute_diff':
                merged_df[f'{score_feat}_diff'] = np.abs(merged_df[f'{score_feat}_1'] - merged_df[f'{score_feat}_2'])
            
            # Use coordinates from first dataset (assuming they're the same)
            result_df = merged_df[[f'{x_feat}_1', f'{y_feat}_1', f'{z_feat}_1', f'{score_feat}_diff']].copy()
            result_df.columns = [x_feat, y_feat, z_feat, f'{score_feat}_diff']
            
        else:
            # Assume same row order, direct subtraction
            if len(self.data) != len(self.data2):
                print("Warning: Datasets have different lengths. Using minimum length.")
                min_len = min(len(self.data), len(self.data2))
                data1_subset = self.data.iloc[:min_len]
                data2_subset = self.data2.iloc[:min_len]
            else:
                data1_subset = self.data
                data2_subset = self.data2
                
            result_df = data1_subset[[x_feat, y_feat, z_feat]].copy()
            
            if operation == 'subtract':
                result_df[f'{score_feat}_diff'] = data1_subset[score_feat] - data2_subset[score_feat]
            elif operation == 'ratio':
                result_df[f'{score_feat}_diff'] = np.where(
                    data2_subset[score_feat] != 0,
                    data1_subset[score_feat] / data2_subset[score_feat],
                    np.nan
                )
            elif operation == 'absolute_diff':
                result_df[f'{score_feat}_diff'] = np.abs(data1_subset[score_feat] - data2_subset[score_feat])
                
        if summarize_similar_points:
            group_cols = [x_feat, y_feat, z_feat]
            diff_col = f'{score_feat}_diff'
            result_df = result_df.groupby(group_cols, as_index=False)[diff_col].mean()
        
        return result_df        
    
    def create_3d_plot(self, x_feat=None, y_feat=None, z_feat=None, score_feat=None, 
                      x_title=None, y_title=None, z_title=None, score_title=None,
                      summarize_similar_points=False, invert_scores=False, 
                      cmap_id="gist_heat"):
        """
        Create a 3D plot using the utils.plot3d function
        
        Args:
            x_feat: string, column name for x-axis (auto-detected if None)
            y_feat: string, column name for y-axis (auto-detected if None)
            z_feat: string, column name for z-axis (auto-detected if None)
            score_feat: string, column name for color mapping (uses self.label if None)
            x_title: string, custom title for x-axis (uses x_feat if None)
            y_title: string, custom title for y-axis (uses y_feat if None)
            z_title: string, custom title for z-axis (uses z_feat if None)
            score_title: string, custom title for colorbar (uses score_feat if None)
            summarize_similar_points: bool, whether to average similar points
            invert_scores: bool, whether to invert the color scale
            cmap_id: string, matplotlib colormap name
        """
        if score_feat is None:
            score_feat = self.label
            
        utils.plot3d(
            df=self.data,
            x_feat=x_feat,
            y_feat=y_feat,
            z_feat=z_feat,
            score_feat=score_feat,
            summarize_similar_points=summarize_similar_points,
            invert_scores=invert_scores,
            cmap_id=cmap_id
        )
        
        # Set custom axis titles if provided
        ax = plt.gca()
        if x_title:
            ax.set_xlabel(x_title)
        if y_title:
            ax.set_ylabel(y_title)
        if z_title:
            ax.set_zlabel(z_title)
        
        # Set custom colorbar title if provided
        if score_title:
            cbar = plt.colorbar(ax.collections[0] if ax.collections else None)
            if cbar:
                cbar.set_label(score_title)
        
        return
    

    def create_3d_difference_plot(self, x_feat=None, y_feat=None, z_feat=None, score_feat=None,
                                x_title=None, y_title=None, z_title=None, score_title=None,
                                merge_on=None, operation='subtract',
                                summarize_similar_points=False, invert_scores=False,
                                cmap_id="RdBu_r"):
        """
        Create a 3D plot showing the difference between two datasets
        """
        diff_df = self.compute_difference_dataset(
            x_feat, y_feat, z_feat, score_feat, merge_on, operation, summarize_similar_points
        )

        
        if diff_df is None:
            return
            
        # Get the actual column names used
        if x_feat is None or y_feat is None or z_feat is None:
            suggestions = self.suggest_axis_columns(1)
            x_feat = x_feat or suggestions['x']
            y_feat = y_feat or suggestions['y']
            z_feat = z_feat or suggestions['z']
            
        score_diff_col = f'{score_feat if score_feat else self.label}_diff'
        
        # Create a new figure with 3D axes explicitly
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        print("Creating 3D scatter plot manually...")
        
        # Create 3D scatter plot manually
        x_data = diff_df[x_feat]
        y_data = diff_df[y_feat]
        z_data = diff_df[z_feat]
        c_data = diff_df[score_diff_col]
        
        if invert_scores:
            c_data = -c_data
            
        scatter = ax.scatter(x_data, y_data, z_data, c=c_data, cmap=cmap_id, s=60, alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
        if score_title:
            cbar.set_label(score_title)
        else:
            cbar.set_label(score_diff_col)
        
        # Now we can safely set 3D labels
        if x_title:
            ax.set_xlabel(x_title)
        else:
            ax.set_xlabel(x_feat)
            
        if y_title:
            ax.set_ylabel(y_title)
        else:
            ax.set_ylabel(y_feat)
            
        if z_title:
            ax.set_zlabel(z_title)
        else:
            ax.set_zlabel(z_feat)
        
        return

    def create_and_show_3d_difference_plot(self, x_feat=None, y_feat=None, z_feat=None, score_feat=None,
                                         x_title=None, y_title=None, z_title=None, score_title=None,
                                         merge_on=None, operation='subtract',
                                         summarize_similar_points=False, invert_scores=False,
                                         cmap_id="RdBu_r"):
        """
        Create and display a 3D difference plot
        
        Args:
            x_feat: string, column name for x-axis (auto-detected if None)
            y_feat: string, column name for y-axis (auto-detected if None)
            z_feat: string, column name for z-axis (auto-detected if None)
            score_feat: string, column name for score comparison
            x_title: string, custom title for x-axis (uses x_feat if None)
            y_title: string, custom title for y-axis (uses y_feat if None)
            z_title: string, custom title for z-axis (uses z_feat if None)
            score_title: string, custom title for colorbar (uses score_feat if None)
            merge_on: list of columns to merge on (if None, assumes same row order)
            operation: string, 'subtract' or 'ratio' or 'absolute_diff'
            summarize_similar_points: bool, whether to average similar points
            invert_scores: bool, whether to invert the color scale (reverses colormap)
            cmap_id: string, matplotlib colormap name
        """
        plt.close("all")
        
        self.create_3d_difference_plot(
            x_feat=x_feat,
            y_feat=y_feat,
            z_feat=z_feat,
            score_feat=score_feat,
            x_title=x_title,
            y_title=y_title,
            z_title=z_title,
            score_title=score_title,
            merge_on=merge_on,
            operation=operation,
            summarize_similar_points=summarize_similar_points,
            invert_scores=invert_scores,
            cmap_id=cmap_id
        )
        
        plt.show()
        return


# Example usage with parametrized axes:
if __name__ == "__main__":
    # Load your two datasets
    df1 = pd.read_csv("data/scalar/race_stats.csv")  # First dataset
    df2 = pd.read_csv("data/vector/race_stats.csv")  # Second dataset
    
    # Initialize with default axes (optional)
    plotter = Plot3DHandler(
        df1, 
        label_column="Passes P2", 
        dataset_df2=df2, 
        label_column2="Passes P2"
    )
    
    # # Example 2: Specify custom axes with custom titles
    plotter.create_and_show_3d_difference_plot(
        x_feat="Theta_a2",
        y_feat="Theta_b2", 
        z_feat="Theta_c2",
        score_feat="Passes P2",
        x_title="Alpha Parameter",
        y_title="Beta Angle (degrees)",
        z_title="Gamma Angle (degrees)",
        score_title="Performance Difference",
        operation='subtract',
        cmap_id="coolwarm",
        invert_scores=False,
        summarize_similar_points=True
    )
    