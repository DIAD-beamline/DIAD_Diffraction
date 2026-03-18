# diad_diffraction

## Quick start

- Create/sync env: `uv sync`
- Run code: `uv run python src/main.py`
- Run tests: `uv run pytest -q`

## Bootstrap on a new machine

After cloning:
- `python project_setup.py bootstrap`

## Layout

- `data/` — datasets (ignored by default)
- `notebooks/` — Jupyter notebooks
- `src/` — Python code (flat; add subfolders if needed)


## Windows Users

If you use PowerShell, you may need to set PYTHONPATH manually before running code:

```powershell
$env:PYTHONPATH = "src"

