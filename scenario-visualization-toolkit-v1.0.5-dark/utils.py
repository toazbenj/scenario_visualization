import pickle
import pycaret.regression
from pycaret.internal.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

import os
import platform
import subprocess

def save(obj : any, fn : str):
    """Saves any @obj to a file @fn"""
    with open(fn, "wb") as f:
        pickle.dump(obj,f)
    return

def load(fn : str):
    """
    Load a python object binary from file @fn
    """
    with open(fn, "rb") as f:
        return pickle.load(f)
    return 

def load_pipeline(fn : str) -> Pipeline:
    """
    Loads a pycaret Pipeline from file @fn.
    Removes the .pkl extension if it exists.
    """
    if (len(fn) >= 4 and fn[-4:] == ".pkl"):
        fn = fn[:-4]
    return pycaret.regression.load_model(fn)

def get_model(pipeline : Pipeline):
    """
    Extracts the surrogate model from a pycaret @pipeline
    """
    return pipeline[-1]

def load_model(fn : str):
    """
    Loads a surrogate model from file @fn, and automatically extracts the model
    from the pipeline.
    Removes the .pkl extension if it exists.
    """
    return load_pipeline(fn)[-1]

def remove_outliers(df : pd.DataFrame, p : float):
    """
    Remove a proportion p of outliers from a dataframe @df
    """
    assert p > 0 and p < 1.0
    # Calculate z-scores for numerical columns
    z_scores = np.abs((df - df.mean()) / df.std())
    
    # Compute the maximum z-score across all columns for each row
    max_z_scores = z_scores.max(axis=1)
    
    # Define a threshold for outliers (e.g., z-score greater than 3)
    threshold = max_z_scores.quantile(1 - p)
    
    # Remove rows with z-scores exceeding the threshold
    cleaned_df = df[max_z_scores <= threshold]
    
    return cleaned_df

def lattice_plot(
        df : pd.DataFrame,
        features : list[str],
        score_feat : str,
        fig_size : list[float, float] = None,
        dpi : int = 300,
        img_fn : str = "",
        trim : float = 0,
        rasterized : bool = False,
        summarize_similar_points : bool = False,
        invert_scores : bool = False,
        marker_size : float = None,
        marker_type : str = ",",
        cmap_id : str = "gist_heat",
    ):
    """
    Generates a lattice plot from a dataframe.

    :: Parameters ::
    df : pd.Dataframe
        Dataframe with features and scores.
    features : list[str]
        List of feature names
    score_feat : str
        Score/label feature name
    fig_size : list[float, float]
        Figure width and height in inches
    dpi : int
        Dots per inch
    img_fn : str
        Filename to save img. Reccomended is .png or .pdf
    trim : float
        Trim the data by p proportion of outliers.
    rasterized : bool
        Rosterize graph when True. Reduces file size at the cost of quality.
        But useful for large datasets. 
    summarize_similar_points : bool
        Averages data at the same location when True
    invert_scores : bool
        Inverts colormap when True
    cmap_id : str
        Colormap to use
    """
    df = df.copy()

    assert trim >= 0 and trim < 1
    if trim > 0:
        df = remove_outliers(df, trim)

    # Clear plot
    # plt.clf()

    # Colors
   
    if invert_scores:
        if cmap_id[-2:] == "_r":
            cmap_id = cmap_id[:-2]
        else:
            cmap_id += "_r"

    cmap = mpl.colormaps[cmap_id]
    
    # Normalize score
    scores_norm = (df[score_feat]-df[score_feat].min())\
                    /(df[score_feat].max()-df[score_feat].min())
    df["scores_norm"] = scores_norm
    df["color"] = scores_norm.apply(cmap)
    a = df[score_feat].min()
    b = df[score_feat].max()
    norm_ticks = [.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
    score_ticks = [np.round((b-a)*x + a, decimals=2) for x in norm_ticks]

    # Make the plots
    plt.rc("font", size=8)
    if fig_size is None:
        fig = plt.figure(dpi=dpi)
    else:
        fig = plt.figure(figsize=fig_size, dpi=dpi)
    nrow = len(features)
    ncol = len(features)
    n_total = nrow * ncol
    counter = 0
    for irow, y_feat in enumerate(features):
        for icol, x_feat in enumerate(features):
            counter += 1
            print("Generating Lattice Plot %.2f%%" \
                  % ((counter/n_total) * 100), end="\r")

            ax : Axes = plt.subplot2grid((ncol, nrow), (irow, icol))

            # Setup labels
            is_left = icol == 0
            is_bottom = irow == nrow - 1

            ax.tick_params(
                left = is_left, 
                right = False , 
                labelleft = is_left , 
                labelbottom = is_bottom, 
                bottom = is_bottom
            )

            if is_left:
                ax.set_ylabel(y_feat)
            if is_bottom:
                ax.set_xlabel(x_feat)

            # Skip same plot
            if x_feat == y_feat:
                ax.set_facecolor("grey")
                continue

            if summarize_similar_points:
                data = avg_similar_points(
                    df[x_feat],
                    df[y_feat],
                    df["scores_norm"]
                )
                data["color"] = data["scores_norm"].apply(cmap)
            else:
                data = df.copy()

            if marker_size is None:
                # Make scatter plot
                ax.scatter(
                    data[x_feat],
                    data[y_feat],
                    color = data["color"],
                    marker=marker_type,
                    s=(72/dpi)**2,
                    rasterized = rasterized
                )
            else:
                # Make scatter plot
                ax.scatter(
                    data[x_feat],
                    data[y_feat],
                    color = data["color"],
                    marker=marker_type,
                    s=marker_size,
                    rasterized = rasterized
                )

            ax.set_facecolor("lightgray")
            continue
        continue

    print()
    
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0,wspace=0)
    

    # Put the colorbar on the grid
    cbar_ax = fig.add_axes([0.91, 0.12, 0.01, .75])
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(0, 1), 
            cmap=cmap_id
        ),
        cax=cbar_ax, 
        orientation="vertical",
        label=score_feat,
    )
    cb.set_ticks(norm_ticks)
    cb.set_ticklabels(score_ticks)

    
    if not img_fn == "":
        plt.savefig(img_fn, bbox_inches="tight")
    return

def plot3d(
    df : pd.DataFrame, 
    x_feat : str,
    y_feat : str,
    z_feat : str,
    score_feat : str,
    figsize : list[float,float] = (6,5),
    summarize_similar_points : bool = False,
    invert_scores : bool = False,
    cmap_id : str = "gist_heat"
) -> Axes:
    """
    Creates a 3D Plot
    
    :: Parameters ::
    df : DataFrame
        The Dataset
    x_feat : str
        Column name of the x-axis feature.
    y_feat : str
        Column name of the y-axis feature
    z_feat : str
        Column name of the z-axis feature.
    score_feat : str
        Column name of the label/score/target feature.
    figsize : list[float,float]
        Figure size for saving the plot.
    summarize_similar_points : bool
        Averages data at the same location when True
    invert_scores : bool
        Inverts colormap when True
    """
    df = df.copy()[[x_feat,y_feat,z_feat,score_feat]]

    if summarize_similar_points:
        # Build Dataset
        data = {}
        for i in range(len(df.index)):
            s = df.iloc[i]
            data[(s[x_feat],s[y_feat],s[z_feat])] = []

        for i in range(len(df.index)):
            s = df.iloc[i]
            data[(s[x_feat],s[y_feat],s[z_feat])].append(s[score_feat])

        stats = {
            x_feat : [],
            y_feat : [],
            z_feat : [],
            score_feat : []
        }
        for key, val in data.items():
            stats[x_feat].append(key[0])
            stats[y_feat].append(key[1])
            stats[z_feat].append(key[2])
            stats[score_feat].append(np.mean(val))
        
        df = pd.DataFrame(stats)
        

    # Colors
    if invert_scores:
        if cmap_id[-2:] == "_r":
            cmap_id = cmap_id[:-2]
        else:
            cmap_id += "_r"
            
    cmap = mpl.colormaps[cmap_id]
    
    # Normalize score
    scores_norm = (df[score_feat]-df[score_feat].min())\
                    /(df[score_feat].max()-df[score_feat].min())
    # if invert_scores:
    #     scores_norm = 1-scores_norm
    
    df["color"] = scores_norm.apply(cmap)
    a = df[score_feat].min()
    b = df[score_feat].max()
    norm_ticks = [.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
    score_ticks = [np.round((b-a)*x + a, decimals=2) for x in norm_ticks]
    
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        df[x_feat],
        df[y_feat],
        df[z_feat],
        color=df["color"],
        alpha=.8
    )
    
    ax.set_xlabel(x_feat)
    ax.set_ylabel(y_feat)
    ax.set_zlabel(z_feat)

    # Put the colorbar on the grid
    cbar_ax = fig.add_axes([0.91, 0.12, 0.01, .75])
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(0, 1), 
            cmap=cmap_id
        ),
        cax=cbar_ax, 
        orientation="vertical",
        label=score_feat
    )
    cb.set_ticks(norm_ticks)
    cb.set_ticklabels(score_ticks)

    # plt.show()
    return

def open_folder(path):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", path])
    else:  # Linux and others
        subprocess.run(["xdg-open", path])

def avg_similar_points(
        s_x : pd.Series, 
        s_y : pd.Series,
        s_score : pd.Series,
        round2decimal_places : int = 3
    ) -> pd.DataFrame:
    """
    Summarizes similar (x,y) points for a given score

    :: Parameters ::
    s_x : pd.Series
        x values
    s_y : pd.Series
        y values
    s_score : pd.Series
        score/label values
    round2decimal_places : int
        Number of decimal places right of the decimal to round score.
    
    :: Return ::
    A dataframe with averaged points.
    """
    df = pd.DataFrame({
        s_x.name : s_x,
        s_y.name : s_y,
        s_score.name : s_score
    })

    x_feat = s_x.name
    y_feat = s_y.name
    score_feat = s_score.name
    data = {}
    for i in range(len(df.index)):
        s = df.iloc[i]
        data[(s[x_feat],s[y_feat])] = []
    
    for i in range(len(df.index)):
        s = df.iloc[i]
        score = np.round(s[score_feat], decimals=round2decimal_places)
        data[(s[x_feat],s[y_feat])].append(score)

    stats = {
        x_feat : [],
        y_feat : [],
        score_feat : []
    }
    for key, val in data.items():
        stats[x_feat].append(key[0])
        stats[y_feat].append(key[1])
        stats[score_feat].append(np.mean(val))
    
    df = pd.DataFrame(stats)
    return df

def test():
    fn = "example.csv"
    features = \
        ["Initial X Position P2","Initial Y Position P2","Out of Bounds P2"]
    df = pd.read_csv(fn)[features]
    avg_similar_points(
        df["Initial X Position P2"],
        df["Initial Y Position P2"],
        df["Out of Bounds P2"]
    )
    return

if __name__ == "__main__":
    test()