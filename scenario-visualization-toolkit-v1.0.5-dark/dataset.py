import pandas as pd
import numpy as np
from pycaret import regression
import os
import shap
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import utils
import constants


class Dataset:
    def __init__(self, 
            data : pd.DataFrame,
            seed : int = 444,
            surrogate_model = None,
            label : str = None
        ):
        """
        Datasetup.

        :: Parameters ::
        data : pd.Dataframe
            A numerical dataset in int or float.
        seed : int (optional)
            Psuedo-random seed
        surrogate_model :  (optional)
            Pre-trained surrogate model.
        label : str (optional)
            Column name of the label/target feature
        """
        self._data = data
        self._rng = np.random.RandomState(444)
        self._surrogate_model = surrogate_model
        self._label = label
        self._explainer = None
        self._explanation = None
        return
    
    @property
    def data(self) -> pd.DataFrame:
        return self._data
    
    @property
    def explainer(self) -> shap.TreeExplainer:
        return self._explainer
    
    @property
    def explanation(self) -> shap.Explanation:
        return self._explanation
    
    @property
    def label(self) -> str:
        return self._label

    @property
    def rng(self) -> np.random.RandomState:
        return self._rng
    
    @property
    def surrogate_model(self):
        return self._surrogate_model
    
    def create_surrogate_model(self, 
            label : str,
            fast : bool = False,
            models : list[str] = None
        ) -> pd.DataFrame:
        assert label in self.data.columns

        print("Setting up PyCaret regression.")
        self._label = label

        out_dir = "%s/%s/%s" % \
            (constants.output_dir, label, constants.pycart_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        setup = {
            "data" : self.data,
            "target" : label,
            "session_id" : self.rng.randint(0,1028),
            "normalize" : True,
            "transform_target" : False,
            "remove_outliers" : False,
            "polynomial_features" : False,
            "feature_selection" : False,
            "verbose" : False,
            "fold" : 10,
        }

        regression.setup(**setup)

        print("Comparing Surrogate Models\n")

        if models is None:
            models = [
                'dt',        # Decision Tree
                'rf',        # Random Forest
                'gbr',       # Gradient Boosting
                'xgboost',   # XGBoost
                # 'lightgbm',  # LightGBM
            ]
            if not fast:
                models.append("lightgbm")

        best_model = regression.compare_models(
            include = models,
            sort='RMSE',
            n_select = 1,
            cross_validation=True,
            verbose=True,
        )
        print("\nThe best model is", best_model, type(best_model))

        """
        Collect model comparison data.
        """
        print("Saving comparison report and best model.")
        df = regression.pull()       

        # print(df)

        df.to_csv(
            "%s/model-comparison.csv" % out_dir,
            index=False
        )

        """
        Save model 
        """
        final_model = regression.finalize_model(best_model)
        pipeline, fn = regression.save_model(
            final_model, "%s/model" % out_dir)
        model = pipeline[-1]
        print(model)
        print("Files store in: %s/" % out_dir)

        self._surrogate_model = model
        return df
    
    def waterfall_plot(self, index : int = 0) -> Axes:
        """
        Constructs a shap waterfall plot for a local case.
        """
        assert self.explainer != None, "No Explainer. Call init_shap() before this function!"
        assert self.explanation != None, "No Explanation. Call init_shap() before this function!"
        assert self.surrogate_model != None, "No surrogate model."
        assert self.label != None, "No label is set."
        assert (index < len(self.data.index) and index >= 0), \
            "Provided index %d must be non-negative and < the size of the dataset (%d)"\
            % (index, len(self.data.index))
        

        # Prepare the data
        # features = self.data.columns.to_list()
        # features.remove(self.label)
        # df = self.data[features]

        # explainer = shap.TreeExplainer(self.surrogate_model)
        
        # shap_values = explainer.shap_values(df)
        # explanation = shap.Explanation(
        #     values = shap_values[index],
        #     base_values = explainer.expected_value,
        #     data = df.iloc[index],
        #     feature_names = features
        # )
    
        
        fig = plt.figure()
        ax = fig.gca()
        shap.waterfall_plot( self.explanation )
        return ax

    def summary_plot(self, 
            bar_plot : bool = False, 
            max_display : int = 20
        ) -> Axes:
        """
        Constructs a summary plot for global analysis.

        :: Parameters ::
        bar_plot : bool (option)
            False (default) = beeswarm plot
            True = bar plot
        """
        assert self.explainer != None, "No Explainer. Call init_shap() before this function!"
        assert self.explanation != None, "No Explanation. Call init_shap() before this function!"
        assert self.surrogate_model != None, "No surrogate model."
        assert self.label != None, "No label is set."

        # Prepare the data
        features = self.data.columns.to_list()
        features.remove(self.label)
        df = self.data[features]

        # explainer = shap.TreeExplainer(self.surrogate_model)
        
        # shap_values = explainer.shap_values(df)
        # explanation = shap.Explanation(
        #     values = shap_values,
        #     base_values = explainer.expected_value,
        #     data = df.iloc,
        #     feature_names = features
        # )
        shap_values = self.explanation.values
        

        fig = plt.figure()
        ax = fig.gca()

        if bar_plot:
            shap.summary_plot(
                shap_values, 
                df, 
                show=False, 
                plot_type="bar",
                max_display=max_display
            )    
        else:
            shap.summary_plot(
                shap_values, 
                df, 
                show=False, 
                max_display = max_display
            )
        return ax
    
    def beeswarm_plot(self, max_display : int = 20) -> Axes:
        """
        Constructs a beeswarm plot for global analysis
        """
        return self.summary_plot(max_display=max_display)
    
    def bar_plot(self, max_display : int = 20) -> Axes:
        """
        Constructs a bar plot for global analysis
        """
        return self.summary_plot(max_display=max_display, bar_plot = True)
    
    def init_shap(self):
        assert self.surrogate_model != None, "No surrogate model."
        assert self.label != None, "No label is set."

        # Prepare the data
        features = self.data.columns.to_list()
        features.remove(self.label)
        df = self.data[features]

        explainer = shap.TreeExplainer(self.surrogate_model)
        self._explainer = explainer

        shap_values = explainer.shap_values(df)
        self._explanation = shap.Explanation(
            values = shap_values,
            base_values = explainer.expected_value,
            data = df.iloc,
            feature_names = features
        )
        return 
    
    def generate_explanation_dataframe(self) -> pd.DataFrame:
        assert self.explainer != None, "No Explainer. Call init_shap() before this function!"
        assert self.explanation != None, "No Explanation. Call init_shap() before this function!"
        # Prepare the data
        features = self.data.columns.to_list()
        features.remove(self.label)
        df = self.data[features]
        shap_values  = self.explanation.values
        return pd.DataFrame(shap_values, columns=df.columns)

    def lattice_plot(self,
        label : str = None,
        features : list[str] = None,
        fig_size : list[float,float] = None,
        dpi : int = 72,
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
        label : str
            Score/label feature name
        features : list[str]
            Specify features to include
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
        """

        if label is None:
            assert self.label != None, "No label is set within this Dataset object. Please set the label within the lattice_plot function using the keyword 'label'."
            label = self.label

        if features is None:
            features = [feat for feat in self.data.columns if feat != label]

        ax = utils.lattice_plot(
            df = self.data,
            features = features,
            score_feat = label,
            fig_size = fig_size,
            dpi = dpi,
            img_fn = img_fn,
            trim = trim,
            rasterized = rasterized,
            invert_scores = invert_scores,
            summarize_similar_points = summarize_similar_points,
            marker_size = marker_size,
            marker_type = marker_type,
            cmap_id = cmap_id
        )
        return
    
    def voronoi(self):
        return
    
    def histogram(self):
        return
    
    def network_graph(self):
        return
    
    