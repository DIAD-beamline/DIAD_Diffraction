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
DiffractionDataAnalysis_Azzimuthal

It provides:

File‑import utilities
Automated and interactive azimuthal peak analysis
Chebichef polynomial background modelling
GoF‑filtered histogram statistics
Cake and flipped‑cake analysis modes
Interactive peak exploration by clicking on the imaging map
TIFF output for batch processing
Interactive sequence viewer for reviewing saved maps and generating movie files

After defining the analysis object, the core workflow is:

Viewer.InputOutput()     # imports the dataset, requests missing parameters
Viewer.<analysis_method>()   # runs a chosen azimuthal/cake analysis mode


4. Parameters Explained
The DiffractionDataAnalysis_Azzimuthal constructor accepts the following parameters:

Parameter        ... Meaning
diff_path        ... Diffraction post‑processing Nexus file (*.nxs)
kbmap_path       ... Raw KB 2D map used for flux normalization
img_path         ... Imaging Nexus file (radiography raw, tomography raw, reconstruction)
out_path         ... Directory and file root where output TIFFs/plots are saved
projection_index ... Optional radiography projection index

If parameters are omitted, InputOutput() will prompt the user via GUI dialogs.

5. Usage — Mode 1 (Explicit Arguments)
Use this mode when all necessary file paths are known:

Viewer = DiffractionDataAnalysis_Azzimuthal(
	dat_path / "mg12123-1" / "processed" / "k11-66526-diffraction-DiffInt_AzzTxt.nxs",
	dat_path / "mg12123-1" / "nexus"    / "66350",
	dat_path / "mg12123-1" / "nexus"    / "k11-66525.nxs",
	dat_path / "mg12123-1" / "processing" / "output")
Viewer.InputOutput()

This loads the dataset and prepares it for analysis.

6. Usage — Mode 2 (GUI File Selection)
Use this mode for user‑driven selection of input files:

Viewer = DiffractionDataAnalysis_Azzimuthal()
Viewer.InputOutput()

InputOutput() will open a GUI to select:

diffraction .nxs
KB map
imaging .nxs
output folder

Missing entries will be requested automatically.

7. Partial Parameter Entry
You may initialize only some parameters and let the GUI prompt for the rest:


Viewer = DiffractionDataAnalysis_Azzimuthal(
    diff_path = dat_path / "mg12123-1" / "processed" / "k11-66526-diffraction-DiffInt_AzzTxt.nxs")
Viewer.InputOutput()

This provides flexibility for varied dataset layouts.

8. Running the Analysis
Set the parameters:

Qmin = 2.85
Qmax = 3.05
cheb = 1
gofx = 0.25
thrc = -1

8.1. Interactive Peak‑by‑Peak Analysis

Explore individual peaks by clicking on them in the imaging view:

Viewer.ImageCorrelatedCrystallography_Azimuthal_Explore(Qmin, Qmax, cheb, gofx, thrc)

8.2. Parameter‑Map Generation (Mean, Area, FWHM)

Viewer.ImageCorrelatedCrystallography_Azimuthal(
    Qmin, Qmax, cheb, gofx, thrc, 'view',
    Mean_min=2.93, Mean_max=2.97,
    Area_min=0.0, Area_max=1.5,
    FWHM_min=0.00, FWHM_max=0.04,
    Mean_relax=2.943
)

9. Automated Analysis of Multiple Datasets

The following batch mode processes multiple diffraction/imaging pairs and writes all output plots as TIFF files:

///// ///// ///// ///// ///// ///// ///// ///// ///// ///// ///// /////
Cycle = "Sample"

pairs = [(66525, 66526),
         (66537, 66538)]

for idx, (iM, iD) in enumerate(pairs):
    Viewer = DiffractionDataAnalysis_Azzimuthal(
        dat_path / "mg12123-1" / "processed" / f"k11-{iD}-diffraction-DiffInt_AzzTxt.nxs",
        dat_path / "mg12123-1" / "nexus" / "66350",
        dat_path / "mg12123-1" / "nexus" / f"k11-{iM}.nxs",
        dat_path / "mg12123-1" / "processing" / f"{Cycle}_{idx}"
    )
    Viewer.InputOutput()
    Viewer.ImageCorrelatedCrystallography_Azimuthal(
        Qmin, Qmax, cheb, gofx, thrc, 'save',
        Mean_min=2.93, Mean_max=2.97,
        Area_min=0.0, Area_max=1.5,
        FWHM_min=0.00, FWHM_max=0.04,
        Mean_relax=2.943
    )
    del Viewer
    print(f"Completed processing frame {idx}")
///// ///// ///// ///// ///// ///// ///// ///// ///// ///// ///// /////

10. Interactive Result Browser & Movie Export

Use the notebook’s interactive movie viewer to scroll through the saved TIFFs and export MP4 animations:

Explore_Sequence(folder, Cycle, "Mean",  pairs)
Explore_Sequence(folder, Cycle, "Area",  pairs)
Explore_Sequence(folder, Cycle, "FWHM",  pairs)

This tool provides:

a frame slider
dynamic image resizing
a “Save Movie” button
automatic backend switching for Jupyter interactive mode

///// ///// ///// ///// ///// ///// ///// ///// ///// ///// ///// /////
import os
import glob
import imageio
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from PIL import Image

# --- Configuration ---
folder = dat_path / "mg12123-1" / "processing"

def Explore_Sequence(folder, Cycle, Data, pairs):
    # Force local interactive backend for this plotting session
    try:
        plt.switch_backend("module://ipympl.backend_nbagg")
    except Exception:
        # fallback if ipympl not available
        plt.switch_backend("nbAgg")
    
    # --- Load images ---
    images = []
    for idx, (iM, iD) in enumerate(pairs):
        path = folder / f"{Cycle}_{idx}_{Data}.tiff"
        img = Image.open(path)
        images.append(img)
    
    fig = plt.figure(figsize=(10, 8))
    
    # Slider (keep this as the anchor for width and horizontal position)
    ax_slider = plt.axes([0.25, 0.08, 0.50, 0.03])   # (left, bottom, width, height)
    slider = Slider(ax_slider, "Frame", 0, len(images)-1, valinit=0, valstep=1)
    
    # Button: centered and ≈1/3 smaller height
    BTN_W = 0.18
    BTN_H = 0.04   # was 0.06 → ~1/3 smaller
    BTN_BOTTOM = 0.02
    
    # Compute centered left position
    center_left = 0.5 - BTN_W / 2.0
    ax_button = plt.axes([center_left, BTN_BOTTOM, BTN_W, BTN_H])
    button = Button(ax_button, "Save Movie")
    
    # Image axes: width/left equal to slider; minimal gap above slider
    ax_img = plt.axes([0.25, 0.15, 0.50, 0.80])  # initial; will be aligned below
    img_display = ax_img.imshow(images[0], cmap="gray")
    ax_img.axis("off")
    
    def layout_match_image_to_slider():
        sbox = ax_slider.get_position()   # Bbox in figure coordinates
        left = sbox.x0
        width = sbox.width
    
        # Place image just above the slider with a small gap
        gap = 0.0                        # ↓ reduce/increase to taste
        bottom = sbox.y0 + sbox.height + gap
    
        # Fill remaining height up to near the top
        top_margin = 0.03                  # small margin from top of figure
        height = 1.0 - bottom - top_margin
        height = max(0.05, height)         # guard: keep positive
    
        ax_img.set_position([left*0.8, bottom*0.6, width*1.2, height*1.2])
    
        # Re-center the button (in case figure width changes)
        btn_left = 0.5 - BTN_W / 2.0
        ax_button.set_position([btn_left, BTN_BOTTOM, BTN_W, BTN_H])
    
    # Initial alignment
    layout_match_image_to_slider()
    
    def on_resize(event):
        layout_match_image_to_slider()
        fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect("resize_event", on_resize)
    
    def update(val):
        idx = int(slider.val)
        img_display.set_data(images[idx])
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    def save_movie(event):
        out_file = os.path.join(folder, f"{Cycle}_{Data}.mp4")
        print(f"Saving movie to {out_file} ...")
        imageio.mimsave(out_file, images, fps=5)
    
    button.on_clicked(save_movie)

    plt.ion()
    plt.show()
///// ///// ///// ///// ///// ///// ///// ///// ///// ///// ///// /////

11. Test Data Provided

A minimal DIAD dataset is included:

data/mg12123-1/
    nexus/
    processed/
    processing/

The notebook examples use this dataset directly.

12. Notes and Troubleshooting

On Windows PowerShell, if imports fail, set this in the notebook before importing:
$env:PYTHONPATH = "src"

Ensure this appears at the top of interactive notebooks:
%matplotlib widgetimport ipympl

This enables sliders, buttons, and live UI updates.

The library uses a safe, off‑screen Agg renderer for saving plots; interactive backends are preserved.

TIFF or MP4 files are written to the out_path specified when creating the viewer.