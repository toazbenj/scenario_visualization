# Scenario Visualization Toolkit

The purpose of this research software is to provide and interpretation and explanation of complex systems through anaylsis of tabular data in `.csv` or `.tsv` format with a graphical user interface (GUI). 

This software will provides a complete SHAP interpretation pipeline, including rapid testing of multiple surrogate data models, a lattice plot, and a 3D plot tool.

## Credits


**Phd. Student**<br>
Quentin Goss, gossq@my.erau.edu<br>

**Faculty Advisor**<br>
Mustafa Ilhan Akbas, akbasm@erau.edu

**Alpha Testing**<br>
Ben Toaz, toazbenj@msu.edu

**Flexible & Intelligent Complex Systems (FICS) Research Group<br>**
*Dept. of Electrical Engineering & Computer Science*<br>
Embry-Riddle Aeronautical University<br>
Daytona Beach, Florida

## Installation

This software is developed and tested using Python v3.9 on Ubuntu 20.04 LTS. To install this software run:
```bash
python3.9 -m pip install -r requirements.txt
```
Then check that all dependencies are met with:
```bash
python3.9 check_requirements.py
```
## Running the GUI

To run this software run:

```bash
python3.9 gui.py
```