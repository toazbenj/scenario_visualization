import FreeSimpleGUI as sg
import pandas as pd
import os
import matplotlib.pyplot as plt
import json

from dataset import Dataset
import constants
import utils


# sg.theme('Default1') # Light Mode
sg.theme("DarkGrey14")

class GUI:
    def __init__(self):
        # self.debug = True
        self.debug = False

        self.values = {}
        self.event = {}
        self.label = None
        self.layout_dataset = []
        self.layout_lattice = []
        self.layout_xai = []
        self.layout_3d = []
        self.layout_credits = []

        self.init_credits()
        self.init_layout_dataset()
        self.init_xai()
        self.init_lattice()
        self.init_3d()
        self.init_window()
        return
    
    @property
    def dataset(self) -> Dataset:
        return self._dataset
    
    def init_window(self):
        # All the stuff inside your window.
        layout = [ 
            [self.layout_dataset],
            [sg.TabGroup(
                [[
                    sg.Tab(
                        "XAI", 
                        self.layout_xai, 
                        key="tab-xai"
                    ),
                    sg.Tab(
                        "Lattice",
                        self.layout_lattice,
                        key="tab-lattice"
                    ),
                    sg.Tab(
                        "3D",
                        self.layout_3d,
                        key="tab-3d"
                    ),
                    sg.Tab(
                        "Credits",
                        self.layout_credits,
                        key="tab-credits"
                    )
                ]],
            )]
        ]


        # Create the Window
        window = sg.Window('Scenario Visualization Toolkit', layout)
        self.window = window

        # Event Loop to process "events" and get the "values" of the inputs
        while True:
            event, values = window.read()
            self.values = values
            self.event = event

            if self.debug:
                self.values["dataset-fn"] = "example.feather"
                self.values["label"] = "Proximity Cost P2"

            if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
                break
            
            elif event == "load-dataset":
                self.load_dataset()

            elif event == "confirm-label":
                self.confirm_label()
            elif event == "train-models":
                self.train_models()
            elif event in ["open-output-folder-3d", "open-output-folder", 
                    "open-output-folder-lattice"]:
                self.open_output_folder()
            elif event == "show-beeswarm":
                self.show_beeswarm()
            elif event == "show-bar":
                self.show_bar()
            elif event == "lattice-save":
                self.create_and_save_lattice_plot()
            elif event == "lattice-show":
                self.create_and_show_lattice_plot()
            elif event == "3d-save":
                self.create_and_save_3d_plot()
            elif event == "3d-show":
                self.create_and_show_3d_plot() 
            elif event == "save-config":
                self.dump_values()
            elif event == "load-config":
                self.load_config()

        window.close()
        return 

    # def init_colors(self):
    #     self.layout_colors(
    #         ["Colormaps are "]
    #     )
    #     return
    
    def init_credits(self):
        txt = "This software contains interpretation and visualization\n"
        txt += "strategies used by Flexible & Intelligent Complex Systems\n"
        txt += "(FICS) research group at Embry-Riddle Aeronautical\n"
        txt += "University in Daytona Beach, Florida. If you use these\n"
        txt += "visualizations in your research. Please consider citing\n"
        txt += "our paper."

        citation = "@article{Goss2024Dec,\n"
        citation += "\tauthor = {Goss, Quentin and Pate, Williams Clay and {\ifmmode\dot{I}\else\.{I}\fi}lhan Akba{\ifmmode\mbox{\c{s}}\else\c{s}\fi}, Mustafa},\n"
        citation += "\ttitle = {{An Integrated Framework for Scenario-Based Safety Validation and Explainability of Autonomous Vehicles}},\n"
        citation += "\tjournal = {ACM J. Auton. Transport. Syst.},\n"
        citation += "\tyear = {2024},\n"
        citation += "\tmonth = dec,\n"
        citation += "\tpublisher = {Association for Computing Machinery},\n"
        citation += "\tdoi = {10.1145/3746286}\n"
        citation += "}"

        l_size = 6
        self.layout_credits = [
            [
                sg.Text(txt)
            ],
            [
                sg.Text("Author:", size=l_size, justification="right"),
                sg.Text("Quentin Goss")
            ],
            [
                sg.Text("Email:", size=l_size, justification="right"),
                sg.Text("gossq@my.erau.edu")
            ],
            [
                sg.Multiline(
                    citation, 
                    size=(50,9),
                    horizontal_scroll=True,
                    disabled=True
                )
            ]
            
        ]
        return

    def init_3d(self):
        size = 10
        self.layout_3d = [
            [sg.Text("3d Graph Options")],
            [
                sg.Text("Feature 1:",s=size, justification="right"),
                sg.Combo([], s=10, readonly=True,key="feat1")
            ],
            [
                sg.Text("Feature 2:",s=size, justification="right"),
                sg.Combo([], s=10, readonly=True,key="feat2")
            ],
            [
                sg.Text("Feature 3:",s=size, justification="right"),
                sg.Combo([], s=10, readonly=True,key="feat3")
            ],
            [
                sg.Text("Colormap:", size=size, justification="right"),
                sg.Combo(
                    plt.colormaps(),
                    size=10, 
                    key="3d-cmap",
                    default_value="gist_heat",
                ),
                sg.Text("MatplotLib Color Maps")
            ],
            [
                sg.Text("",s=size-1),
                sg.Checkbox("Average similar points.", default=False,
                            key='cb-average-similar-points')
            ],
            [
                sg.Text("",s=size-1),
                sg.Checkbox("Invert color scale.", default=False,
                            key='cb-invert-color-scale')
            ],
            [
                sg.Button("Create & Save", key="3d-save"),
                sg.Button("Create & Show", key="3d-show")
            ],
            [
                sg.Button("Open output folder.", 
                          key="open-output-folder-3d", disabled=True)
            ],
            [sg.Text("",key="3d-status")],
        ]
        return
    
    def init_xai(self):
        self.layout_xai = [
            [sg.T("Choose Surrogate Models")],
            [sg.Checkbox("Decision Tree",default=True,key="cb-dt")],
            [sg.Checkbox("Random Forest",default=True,key="cb-rf")],
            [sg.Checkbox("Gradient Boosted",default=True,key="cb-gbr")],
            [sg.Checkbox("XGBoost",default=True,key="cb-xgboost")],
            [sg.Checkbox("LightGBM (SLOW)",default=False,key="cb-lightgbm")],
            [sg.Button("Train Models",key="train-models")],
            [sg.Text("Click \"Train Models\" to proceed.", key="train-models-text")],
            [
                sg.Button("Open output folder.", key="open-output-folder", disabled=True),
                sg.Button("Show Beeswarm", key="show-beeswarm", disabled=True),
                sg.Button("Show Bar", key="show-bar", disabled=True)
            ]
        ]
        return
    
    def init_lattice(self):
        size = 12
        self.layout_lattice = [
            [sg.Text("Lattice Plot Options.")],
            [
                sg.Text("Cell Width:",size=size,
                        justification="right"),
                sg.Input("1",size=5,key="cell-width"),
                sg.Text("inches")
            ],
            # [
            #     sg.Text("DPI:",size=size,justification="right"),
            #     sg.Input("72", size=5, key="lattice-dpi")
            # ],
            [
                sg.Text("Trim Data:", size=size, justification="right"),
                sg.Input(".0", size=5,key="lattice-trim"),
                sg.Text("p where 0 ≤ p ≤ 1")
            ],
            [
                sg.Text("Marker Type:", size=size, justification="right"),
                sg.Combo(
                    [",",".","+","o"],
                    size=4, 
                    key="lattice-marker-type",
                    default_value=","
                ),
                sg.Text("MatplotLib Marker")
            ],
            [
                sg.Text("Colormap:", size=size, justification="right"),
                sg.Combo(
                    plt.colormaps(),
                    size=10, 
                    key="lattice-cmap",
                    default_value="gist_heat",
                ),
                sg.Text("MatplotLib Color Maps")
            ],
            [
                sg.Text("Marker Size:", size=size, justification="right"),
                sg.Input("0", size=5,key="lattice-marker-size"),
                sg.Text("(72/marker_size) ^ 2")
            ],
            [
                sg.Text("",s=size-1),
                sg.Checkbox("Average similar points. (Warning: Slow)", default=False,
                            key='cb-average-similar-points-lattice')
            ],
            [
                sg.Text("",s=size-1),
                sg.Checkbox("Invert color scale.", default=False,
                            key='cb-invert-color-scale-lattice')
            ],
            [
                sg.Text("",size=size-1),
                sg.Checkbox("Rasterized",default=False,key="lattice-rasterized")
            ],
            [
                sg.Button("Create & Save", key="lattice-save"),
                sg.Button("Create & Show", key="lattice-show")
            ],
            [
                sg.Button("Open output folder.", 
                          key="open-output-folder-lattice", disabled=True)
            ],
            [sg.Text("",key="lattice-status")]
        ]
        return
    
    def create_and_save_3d_plot(self):
        plt.close("all")
        self.create_3d_plot()

        out_dir = "%s/%s" % (constants.output_dir, self.values["label"])
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        x_feat = self.values["feat1"]
        y_feat = self.values["feat2"]
        z_feat = self.values["feat3"]

        fn = "%s/3d-%s-%s-%s.png" % (out_dir, x_feat, y_feat, z_feat)
        plt.savefig(fn, bbox_inches = "tight")
        self.window.find_element("open-output-folder-3d")\
            .update(disabled=False)
        
        self.window.find_element("3d-status")\
            .update("OKAY!")

        print("3D plot saved to %s" % fn)
        return
    
    def create_and_show_3d_plot(self):
        plt.close("all")
        self.create_3d_plot()

        plt.show()
        return
    
    def create_3d_plot(self):
        utils.plot3d(
            df = self.dataset.data,
            x_feat = self.values["feat1"],
            y_feat = self.values["feat2"],
            z_feat = self.values["feat3"],
            score_feat = self.values["label"],
            summarize_similar_points = self.values["cb-average-similar-points"],
            invert_scores = self.values['cb-invert-color-scale'],
            cmap_id = self.values["3d-cmap"]
        )
        return

    def create_and_show_lattice_plot(self):
        plt.close("all")
        self.create_lattice_plot()
        self.window.find_element("lattice-status")\
            .update("OKAY!")
        plt.show()
        return

    def create_and_save_lattice_plot(self):
        plt.close("all")
        self.create_lattice_plot()

        out_dir = "%s/%s" % (constants.output_dir, self.values["label"])
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        fn = "%s/lattice.png" % out_dir
        plt.savefig(fn, bbox_inches = "tight")
        self.window.find_element("open-output-folder-lattice")\
            .update(disabled=False)
        
        self.window.find_element("lattice-status")\
            .update("OKAY!")

        print("Lattice plot saved to %s" % fn)
        return

    def create_lattice_plot(self):

        # for key, val in self.values.items():
        #     print(key,val)

        label = self.values["label"]
        features = [feat for feat in self.dataset.data.columns \
                    if feat != label]
        # dpi = int(self.values["lattice-dpi"])
        cell_width = float(self.values["cell-width"])
        fig_size = (len(features)*cell_width, len(features)*cell_width)
        trim = float(self.values["lattice-trim"])
        rasterized = self.values["lattice-rasterized"]
        marker_type = self.values["lattice-marker-type"]
        marker_size = float(self.values["lattice-marker-size"])
        cmap_id = self.values["lattice-cmap"]

        if marker_size == 0:
            marker_size = None

        self.dataset.lattice_plot(
            label = label,
            features = features,
            fig_size = fig_size,
            # dpi = dpi,
            trim = trim,
            rasterized = rasterized,
            summarize_similar_points = \
                self.values['cb-average-similar-points-lattice'],
            invert_scores = \
                self.values['cb-invert-color-scale-lattice'],
            marker_type = marker_type,
            marker_size = marker_size,
            cmap_id=cmap_id
        )
        return

    def init_layout_dataset(self):
        self.layout_dataset = [
            [sg.T("Choose a dataset to load. (.csv/.tsv supported.)\nThe dataset should contain numeric values or None only.")],
            [
                sg.Input(key = "dataset-fn"),
                sg.FileBrowse(
                    file_types=( [["CSV/TSV/Feather Files", ["*.csv","*.tsv","*.feather"]]] )
                )
            ],
            [sg.Checkbox("Remove extra header rows.", default=True, 
                key="remove-extra-headers")],
            [
                sg.Button("Load Dataset", key="load-dataset"),
                sg.Button("Save Config", key="save-config"),
                sg.Button("Load Config", key="load-config")
            ],
            [sg.Text("", key="dataset-status")],
            [sg.HorizontalSeparator()],
            [
                sg.T("Label Feature:"),sg.Combo([], 
                    key="label", s=10, readonly=True),
                sg.Button("Confirm", key="confirm-label",disabled=True)
            ],
            [sg.Text("Confirm label before proceeding.", key="confirm-label-text")]
        ]
        return
    
    def confirm_label(self):
        label = self.values["label"]
        if label in self.dataset.data.columns.tolist():
            # self.window.find_element("tab-xai").update(disabled=False)
            self.window.find_element("confirm-label-text")\
                .update("OK: Label feature confirmed.")
            self.label = label
        else:
            self.window.find_element("confirm-label-text")\
                .update("ERR: Check label feature.")
        return

    def check_and_load_dataset(self):
        if self.dataset is None:
            self.load_dataset()
        return

    def load_dataset(self):
        

        # Check if file exists
        if not os.path.exists(self.values["dataset-fn"]):
            self.window.find_element("dataset-status")\
                .update("ERR: File does not exist: %s" % self.values["dataset-fn"])
            return
            
        # Check if the file extension is correct
        fn : str = self.values["dataset-fn"]
        ext = fn[fn.rfind("."):]
        if not (ext in [".csv", ".tsv", ".feather"]):
        # if not ((ext == ".csv") or (ext == ".tsv")):
            self.window.find_element("dataset-status")\
                .update("ERR: File extension must be .csv, .tsv or .feather.")
            return
        
        if ext in [".csv",".tsv"]:
            sep = ","
            if ext == ".tsv":
                sep = "\t"
            df = pd.read_csv(self.values["dataset-fn"],sep = sep)
        else:
            df = pd.read_feather(self.values["dataset-fn"])
        
        if self.values["remove-extra-headers"]:
            # Remove string data
            features = df.columns.tolist()
            keep = []
            for i in range(len(df.index)):
                val = df.iloc[i][0]
                if isinstance(val, str) and val in features:
                    keep.append(False)
                    continue
                keep.append(True)
                continue

            df = df[keep]
            
            
            # FIx Types
            for feat in features:
                val = df[feat].iloc[0]
                if val in ["True", "False"]:
                    df[feat] = df[feat].astype(bool)
                    continue
                df[feat] = df[feat].astype(float)
                continue
            
            
            

        self._dataset = Dataset(df)
        self.window.find_element("dataset-status")\
                .update("OK: Dataset Loaded.")
        
        self.window.find_element("label")\
            .update(
                values=self.dataset.data.columns.tolist(),
                # readonly = False
            )
        self.window.find_element("confirm-label").update(disabled=False)

        for i in range(1,3+1):
            self.window.find_element("feat%d" % i)\
                .update(
                    values = self.dataset.data.columns.tolist(),
                    # readonly = False
                )
        return
    
    def train_models(self):
        
        self.window.find_element("train-models-text")\
            .update("Training models. See terminal for progress.")

        model_ids = ["dt", "rf", "gbr", "xgboost", "lightgbm"]
        models = []
        for model in model_ids:
            is_checked = self.values["cb-%s" % model]
            if is_checked:
                models.append(model)
            continue
        
        self.dataset.create_surrogate_model(
            label = self.label, 
            models = models
        )

        self.window.find_element("train-models-text")\
            .update("Training Complete.")
        
        out_dir = "%s/%s" % (constants.output_dir, self.dataset.label)
        shap_dir = "%s/%s" % (out_dir, constants.shap_dir)

        if not os.path.exists(shap_dir):
            os.makedirs(shap_dir)

        self.dataset.beeswarm_plot()
        plt.savefig("%s/beeswarm.png" % shap_dir, bbox_inches = "tight")

        plt.clf()
        self.dataset.bar_plot()
        plt.savefig("%s/bar.png" % shap_dir, bbox_inches = "tight")
        
        for element in ["open-output-folder", "show-beeswarm", "show-bar"]:
            self.window.find_element(element).update(disabled=False)

        return

    def open_output_folder(self):
        print(self.values["label"])
        out_dir = "%s/%s" % (constants.output_dir, self.values["label"])
        utils.open_folder(out_dir)
        return

    def show_beeswarm(self):
        plt.close("all")
        self.dataset.beeswarm_plot()
        plt.show()
        return
    
    def show_bar(self):
        plt.close("all")
        self.dataset.bar_plot()
        plt.show()
        return
    
    def dump_values(self):
        with open(constants.config_json, "w") as f:
            json.dump(self.values,f,indent=4)
        return
    
    def load_config(self):
        if not os.path.exists(constants.config_json):
            return
        
        with open(constants.config_json, "r") as f:
            values = json.load(f)

        self.values = values
        if len(values["dataset-fn"]) > 0:
            self.load_dataset()
        if len(values["label"]) > 0:
            self.confirm_label()

        # print(values)
        for key, val in values.items():
            if key.isdigit():
                key = int(key)
            element = self.window.find_element(key)
            if "button" in str(type(element)).lower():
                print(element)
                continue
            element.update(val)
            continue

        
        return
    


if __name__ == "__main__":
    GUI()
