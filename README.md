# DIAD_Diffraction

This repository contains tools, Python libraries, Jupyter notebooks, and a minimal test dataset for
DIAD diffraction and azimuthal analysis.
The project is designed to be fully reproducible using `uv` for environment management and to run
consistently on Windows (PowerShell), Linux, macOS, and WSL.
Below are complete instructions for cloning the repository, creating the environment, and running
notebooks with interactive backends, all in a single continuous description.

## Quick start

- Create/sync env: `uv sync`
- Run code: `uv run python src/main.py`
- Run tests: `uv run pytest -q`

## Bootstrap on a new machine

After cloning:
- `python project_setup.py bootstrap`

## Layout

- `data/` — datasets (ignored by default)
- `data/mg12123-1` — reference test datasets
- `notebooks/` — Jupyter notebooks
- `src/` — Python code (flat; add subfolders if needed)


## Windows Users

If you use PowerShell, you may need to set PYTHONPATH manually before running code:

```powershell
$env:PYTHONPATH = "src"

## Use and License

To obtain the repository, clone it from GitHub using either HTTPS or SSH. For HTTPS, run:
    > git clone https://github.com/DIAD-beamline/DIAD_Diffraction.git

For SSH (if your GitHub credentials are configured for SSH):
    > git clone git@github.com:DIAD-beamline/DIAD_Diffraction.git

After cloning, enter the project directory with:
    > cd DIAD_Diffraction

The project uses `uv` for fast, reproducible Python virtual environments. To create the environment
and install all dependencies, execute:
    > uv sync

This will create a `.venv` directory and install all required dependencies exactly as pinned in
`uv.lock`.
If you clone this repository onto a new machine and want to fully re-bootstrap it, run:
    > python project_setup.py bootstrap

This recreates the environment in a fully reproducible manner.

To run Jupyter notebooks, different instructions apply depending on your platform.
On Linux, macOS, or WSL, simply launch:
    > uv run jupyter lab

Then select the kernel corresponding to the virtual environment (usually named something like
`diad_diffraction (.venv)`).
No manual path configuration is required.

On Windows PowerShell, start JupyterLab the same way:
    > uv run jupyter lab

However, PowerShell sometimes does not honor the `.env` file contained in the project.
If imports from `src/` fail (for example: `ModuleNotFoundError: No module named
'diffraction_data_explorer'`), set the environment variable manually:
    > $env:PYTHONPATH = "src"

This usually needs to be done once per PowerShell session.

Inside a Jupyter notebook, to enable interactive widgets (buttons, sliders, interactive image
exploration), you may need to activate the ipympl widget backend.
If so, add the following to the top of every notebook that requires interactive functionality:
    %matplotlib widget
    import ipympl
This ensures that all interactive Matplotlib widgets (Sliders, Buttons) behave correctly.
The project's internal figure‑saving functions automatically use a private Agg renderer so that
saving TIFF/PNG figures never interferes with interactive plotting.


The typical project layout is as follows:
    DIAD_Diffraction/
    ├── data/
    │   ├── mg12123-1/     (reference test dataset; NXS/HDF5/TIFF/MP4 files)
    │   └── .gitignore     (ignores other data folders by default)
    ├── notebooks/
    │   ├── ExampleNotebook_DiffractionDataAnalysis.ipynb
    │   └── ExampleNotebook_DiffractionDataExplorer.ipynb
    ├── src/
    │   ├── diffraction_data_explorer/
    │   └── diffraction_data_analysis_azimuthal/
    ├── project_setup.py
    ├── pyproject.toml
    ├── uv.lock
    ├── WINDOWS_SETUP.txt
    └── README.md


To run Python scripts in the environment without activating `.venv` manually, use:
    > uv run python path/to/script.py
To run tests (if you add them later), use:
    > uv run pytest -q

The repository includes a small DIAD reference dataset at `data/mg12123-1/`, containing Nexus
files, intermediate processed data, TIFF output images, and MP4 output files.
These allow the notebooks to run end‑to‑end without requiring access to DLS storage.

Some common troubleshooting tips:
• If interactive buttons or sliders appear visually inactive in a notebook, you likely
are not using the widget backend. Ensure:
      %matplotlib widget
      import ipympl
• If imports from `src/` fail on Windows, set:
      $env:PYTHONPATH = "src"
• If saved TIFF figures appear blank, restart the notebook kernel. The library automatically
uses a private Agg renderer for saving figures, which avoids backend conflicts.
• Do not call `matplotlib.use("Agg")` in notebooks or scripts; the library handles backend
separation internally.


***********************************************************************************

LICENSE

This software is provided under the MIT License, with Additional Terms requiring 
explicit acknowledgment of the DIAD_Diffraction repository and contributors.
See ADDITIONAL_TERMS.txt.

Copyright (c) 2026 DIAD-beamline

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in  
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING  
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER  
DEALINGS IN THE SOFTWARE.

ADDITIONAL TERMS OF USE

In addition to complying with the MIT License, any redistribution, modification, 
or publication of this software (including derivative works) MUST include an 
explicit acknowledgment of:

    “DIAD_Diffraction Repository — https://github.com/DIAD-beamline/DIAD_Diffraction”
    and its contributors.

This acknowledgment must appear in documentation, publications, or software notices 
associated with any redistributed or derivative work.

These additional terms are a mandatory condition of use.


Contributions are welcome; please open issues or pull requests at:
https://github.com/DIAD-beamline/DIAD_Diffraction

