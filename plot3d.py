import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'scenario-visualization-toolkit-v1.0.4-dark'))
import utils

class Plot3DHandler:
    def __init__(self, dataset_df, label_column):
        """
        Initialize the 3D plotting handler
        
        Args:
            dataset_df: pandas DataFrame containing the data
            label_column: string, name of the label/score column
        """
        self.data = dataset_df
        self.label = label_column
        self.output_dir = "output"  # Default output directory
        
    def create_3d_plot(self, x_feat, y_feat, z_feat, score_feat=None, 
                      summarize_similar_points=False, invert_scores=False, 
                      cmap_id="gist_heat"):
        """
        Create a 3D plot using the utils.plot3d function
        
        Args:
            x_feat: string, column name for x-axis
            y_feat: string, column name for y-axis  
            z_feat: string, column name for z-axis
            score_feat: string, column name for color mapping (uses self.label if None)
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
        return
    
    def create_and_save_3d_plot(self, x_feat, y_feat, z_feat, score_feat=None,
                               summarize_similar_points=False, invert_scores=False,
                               cmap_id="gist_heat", output_dir=None):
        """
        Create and save a 3D plot to file
        
        Args:
            x_feat: string, column name for x-axis
            y_feat: string, column name for y-axis  
            z_feat: string, column name for z-axis
            score_feat: string, column name for color mapping (uses self.label if None)
            summarize_similar_points: bool, whether to average similar points
            invert_scores: bool, whether to invert the color scale
            cmap_id: string, matplotlib colormap name
            output_dir: string, output directory path (uses self.output_dir if None)
        """
        plt.close("all")
        
        if score_feat is None:
            score_feat = self.label
            
        if output_dir is None:
            output_dir = self.output_dir
            
        self.create_3d_plot(
            x_feat=x_feat,
            y_feat=y_feat, 
            z_feat=z_feat,
            score_feat=score_feat,
            summarize_similar_points=summarize_similar_points,
            invert_scores=invert_scores,
            cmap_id=cmap_id
        )

        # Create output directory if it doesn't exist
        out_dir = f"{output_dir}/{self.label}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Save the plot
        filename = f"{out_dir}/3d-{x_feat}-{y_feat}-{z_feat}.png"
        plt.savefig(filename, bbox_inches="tight")
        
        print(f"3D plot saved to {filename}")
        return filename
    
    def create_and_show_3d_plot(self, x_feat, y_feat, z_feat, score_feat=None,
                               summarize_similar_points=False, invert_scores=False,
                               cmap_id="gist_heat"):
        """
        Create and display a 3D plot
        
        Args:
            x_feat: string, column name for x-axis
            y_feat: string, column name for y-axis  
            z_feat: string, column name for z-axis
            score_feat: string, column name for color mapping (uses self.label if None)
            summarize_similar_points: bool, whether to average similar points
            invert_scores: bool, whether to invert the color scale
            cmap_id: string, matplotlib colormap name
        """
        plt.close("all")
        
        if score_feat is None:
            score_feat = self.label
            
        self.create_3d_plot(
            x_feat=x_feat,
            y_feat=y_feat,
            z_feat=z_feat, 
            score_feat=score_feat,
            summarize_similar_points=summarize_similar_points,
            invert_scores=invert_scores,
            cmap_id=cmap_id
        )
        
        plt.show()
        return

# Example usage:
if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv("data/scalar/race_stats.csv")  # Replace with your data loading
    
    # Initialize the 3D plotter
    plotter = Plot3DHandler(df, label_column="Passes P2")
    
    # Create and show a 3D plot
    plotter.create_and_show_3d_plot(
        x_feat="Theta_a2",
        y_feat="Theta_b2", 
        z_feat="Theta_c2",
        cmap_id="coolwarm",
        summarize_similar_points=True
    )
    
    # Create and save a 3D plot
    plotter.create_and_save_3d_plot(
        x_feat="Theta_a2",
        y_feat="Theta_b2", 
        z_feat="Theta_c2",
        cmap_id="coolwarm",
        invert_scores=True
    )