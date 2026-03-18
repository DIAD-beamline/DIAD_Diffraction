## DIAD_Diffraction: Diffraction-Imaging Coerelated Data Explorer

This repository provides Python libraries and Jupyter notebooks to explore diffraction and imaging data recorded at the DIAD beamline.
The core component is the DiffractionDataExplorer class, which offers interactive visualization, data browsing, and correlated diffraction–imaging analysis.
This README explains how to set up the environment, load the library inside a notebook, and run the viewer in both interactive and manual modes.

1. Local Environment Setup
First ensure you are inside the root folder of this repository:
DIAD_Diffraction/

Create and synchronize the uv environment:
> uv sync

Launch JupyterLab:
> uv run jupyter lab

Inside Jupyter, select the kernel corresponding to this project’s virtual environment.

2. Notebook Preparation
Before importing any modules, run the following cell in your Jupyter notebook to correctly configure import paths and local data folders:

///// ///// ///// ///// ///// ///// ///// ///// ///// ///// ///// /////

import sys
from pathlib import Path

# Notebook's working directory (the folder containing the .ipynb)
notebook_dir = Path().resolve()

# Project root = parent of the notebook folder
project_root = notebook_dir.parent

# src directory
src_path = project_root / "src"
dat_path = project_root / "data"

# Prepend src/ to sys.path
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

print("src path added:", src_path)
print("test data path:", dat_path)

from diffraction_data_explorer import DiffractionDataExplorer

///// ///// ///// ///// ///// ///// ///// ///// ///// ///// ///// /////

This ensures the notebook can import the library code directly from src/.

3. Diffraction Viewer Overview
The main class used for exploration is:
DiffractionDataExplorer

It provides:

File import utilities
Interactive imaging + diffraction browsing
Scatter plotting and parameter maps
Histogram visualization
Integrated GUI for selecting input/output files
Support for Nexus .nxs, HDF5 .h5, TIFF .tiff, and MP4 outputs

After defining the viewer object, two main commands are used:
Viewer.InputOutput()   # imports and prepares all necessary dataViewer.DataExplorer()  # launches the interactive viewerShow more lines

4. Parameters Explained
The DiffractionDataExplorer constructor accepts:

Parameter ... Meaning
diff_path ... Diffraction post‑processing Nexus file (.nxs)
img_path  ... Imaging Nexus file (radiography raw, tomo raw, or reconstruction)
out_path  ... Directory where plots and TIFF/MP4 outputs will be written
proj_idx  ... Optional projection index (used for radiography datasets)

If any parameters are omitted, the viewer will request them using a file–browser GUI when InputOutput() is run.

5. Usage — Mode 1 (Explicit Arguments)
Use when you already know the file paths:

Viewer_A = DiffractionDataExplorer(
	dat_path / "mg12123-1" / "processed" / "k11-66526-diffraction-DiffInt_AzzTxt.nxs",
	dat_path / "mg12123-1" / "nexus"    / "k11-66525.nxs",
	dat_path / "mg12123-1" / "processing" / "output")
Viewer_A.InputOutput()
Viewer_A.DataExplorer()

This loads the dataset and opens the interactive viewer.

6. Usage — Mode 2 (GUI File Selection)
Use when you want the viewer to ask for files interactively:

Viewer_A = DiffractionDataExplorer()Viewer_A.InputOutput()Viewer_A.DataExplorer()Show more lines

In this mode:

A GUI window will open
You will select diffraction, imaging, and output files/directories
Missing parameters will be prompted automatically


7. Partial Parameter Entry
You may initialize only some parameters and let InputOutput() prompt for the rest:

Viewer_A = DiffractionDataExplorer(
	diff_path = dat_path / "mg12123-1" / "processed" / "k11-66526-diffraction-DiffInt_AzzTxt.nxs")
Viewer_A.InputOutput()  # selects img_path and out_path interactively
Viewer_A.DataExplorer()

This provides flexibility for different datasets.

8. Test Data Provided
A lightweight DIAD reference dataset is included:
data/mg12123-1/
    nexus/
    processed/
    processing/

The example notebook uses this dataset directly.

9. Typical Workflow in a Notebook

Run environment setup cell (adds src/ and data/ paths).
Import the DiffractionDataExplorer module.
Define the viewer object (Mode 1 or Mode 2).
Run InputOutput() to load data.
Run DataExplorer() to explore interactively.
Save plots, TIFFs, or movies using the built‑in tools.

10. Notes and Troubleshooting

On Windows PowerShell, if imports fail, set this in the notebook before importing:
$env:PYTHONPATH = "src"


Ensure this appears at the top of interactive notebooks:
%matplotlib widgetimport ipympl

This enables sliders, buttons, and live UI updates.

The library uses a safe, off‑screen Agg renderer for saving plots; interactive backends are preserved.

TIFF or MP4 files are written to the out_path specified when creating the viewer.