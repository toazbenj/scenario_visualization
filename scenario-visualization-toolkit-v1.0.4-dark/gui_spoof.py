from gui import *

gui = GUI()

gui.load_dataset("c:\\Users\\toazb\Documents\GitHub\\race_simulation\data_processing\\version2_data\weighted_sum\cost_stats.feather")

gui.values["label"] = "action1"

gui.dataset.bar_plot()