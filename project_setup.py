#!/usr/bin/env python3
"""
project_setup.py — project scaffold + clone bootstrap for uv + VS Code (cross-platform)

Purpose
-------
You work across multiple machines/OSes and keep projects in GitHub. This script supports a
repeatable workflow in two modes:

1) init  — Create a new project scaffold (folders + starter files) and optionally:
          - run 'uv sync' to create .venv and install deps
          - run 'git init' and an initial commit
          - embed a copy of THIS script into the new project so the repo is self-bootstrapping

2) bootstrap — After you clone the repo onto another machine:
             - run 'uv sync' in the repo to recreate/sync .venv from pyproject.toml + uv.lock
             - (optionally) enforce locked installs, include/exclude dev deps, etc.

Why embed this script into each new repo?
----------------------------------------
Because then every cloned repo already includes the bootstrap command:
    python project_setup.py bootstrap

That makes “continue where I left off” consistent across machines.

Typical workflows
-----------------
Create a new project:
    python project_setup.py init ~/dev/my_project --uv-sync --git-init

Clone an existing project on another machine, then inside the repo:
    python project_setup.py bootstrap

Notes
-----
- This scaffold uses a flat 'src/' folder (no extra package subfolder). You can add subfolders later.
- For imports, VS Code loads .env (PYTHONPATH=src) so src/ is importable in notebooks/tests.
- The data/ folder is ignored by default to avoid accidentally committing large datasets.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

# The name we want this script to have when embedded into a new project.
EMBEDDED_SCRIPT_NAME = "project_setup.py"


# ----------------------------
# Small utilities / primitives
# ----------------------------

def die(msg: str, code: int = 1) -> None:
    """Print a fatal error and exit with a non-zero code."""
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    """
    Run a subprocess command, echoing it first.

    - cmd: list of tokens (recommended for cross-platform safety)
    - cwd: working directory to run the command in
    """
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def which_or_die(exe: str, hint: str) -> None:
    """Ensure an executable exists on PATH; if not, fail with a helpful hint."""
    if shutil.which(exe) is None:
        die(f"'{exe}' not found on PATH. {hint}")


def ensure_dir(path: Path) -> None:
    """Create a directory (and parents) if it doesn't already exist."""
    path.mkdir(parents=True, exist_ok=True)


def write_file(path: Path, content: str, force: bool) -> None:
    """
    Write a UTF-8 text file.

    - If the file already exists, we refuse to overwrite unless --force is set.
    """
    if path.exists() and not force:
        die(f"Refusing to overwrite existing file: {path} (use --force to overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"wrote {path}")


def find_project_root(start: Path) -> Path:
    """
    Walk upward from 'start' to locate the repository/project root,
    defined here as the first folder containing pyproject.toml.
    """
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    die("Could not find pyproject.toml in this directory or any parent directory.")


def slugify_project_name(name: str) -> str:
    """
    Make a safe project name for pyproject.toml [project].name.

    Rules:
    - lowercase
    - spaces -> hyphens
    - keep letters, digits, dot, underscore, hyphen
    """
    s = name.strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "my-project"


def embed_self_into_project(target_dir: Path, force: bool) -> None:
    """
    Copy this script into the newly created project so it can bootstrap itself later.

    This is the key “self-contained repo” feature:
        python project_setup.py bootstrap
    will work after cloning the repo elsewhere.
    """
    try:
        src_path = Path(__file__).resolve()
        content = src_path.read_text(encoding="utf-8")
    except Exception as e:
        die(f"Could not read this script file for embedding: {e}")

    dest_path = target_dir / EMBEDDED_SCRIPT_NAME
    write_file(dest_path, content, force=force)


# ----------------------------
# Command: init (new project)
# ----------------------------

def init_project(args: argparse.Namespace) -> None:
    """
    Create a new project scaffold with your preferred layout:

      data/        (ignored by default; includes README + .gitignore)
      notebooks/   (includes README)
      src/         (flat; starter main.py)
      tests/       (pytest config + smoke test)
      .vscode/     (settings)
      .env         (PYTHONPATH=src)
      pyproject.toml
      .python-version
      README.md
      .gitignore
      project_setup.py   (embedded copy of this script)

    Optionally:
      --uv-sync   run uv sync immediately (creates .venv and installs deps)
      --git-init  initialize git + create initial commit
    """
    target = Path(args.path).expanduser().resolve()

    # Guardrails: prevent accidental overwrites unless --force
    if target.exists() and any(target.iterdir()) and not args.force:
        die(f"Target directory exists and is not empty: {target} (use --force to proceed)")

    ensure_dir(target)

    # 1) Embed this script early so it becomes part of the project from day 0.
    embed_self_into_project(target, force=args.force)

    # 2) Create folders
    ensure_dir(target / "data")
    ensure_dir(target / "notebooks")
    ensure_dir(target / "src")
    ensure_dir(target / "tests")
    ensure_dir(target / ".vscode")

    # Project name defaults to folder name unless overridden
    project_display_name = target.name
    project_name = slugify_project_name(args.name or project_display_name)

    # 3) data/ — ignore by default to prevent accidental large/sensitive commits
    write_file(
        target / "data" / ".gitignore",
        dedent(
            """\
            # Ignore all data files by default
            *
            # Keep this file and the README
            !.gitignore
            !README.md
            """
        ),
        force=args.force,
    )
    write_file(
        target / "data" / "README.md",
        dedent(
            """\
            # data/

            Put raw or intermediate datasets here.

            This folder ignores contents by default (via .gitignore) so you don't
            accidentally commit large or sensitive data. If you *do* want to commit
            a specific file, add an exception rule to data/.gitignore.
            """
        ),
        force=args.force,
    )

    # 4) notebooks/ — tracked (README keeps folder present even if empty)
    write_file(
        target / "notebooks" / "README.md",
        dedent(
            """\
            # notebooks/

            Jupyter notebooks live here.
            """
        ),
        force=args.force,
    )

    # 5) .env — environment vars for VS Code (not your virtualenv!)
    # PYTHONPATH=src makes flat src/ importable in notebooks/tests without packaging.
    write_file(target / ".env", "PYTHONPATH=src\n", force=args.force)

    # 6) VS Code settings — cross-platform (avoid hardcoding interpreter path)
    write_file(
        target / ".vscode" / "settings.json",
        dedent(
            """\
            {
              "python.terminal.activateEnvironment": true,
              "python.envFile": "${workspaceFolder}/.env",
              "python.analysis.extraPaths": ["src"],
              "python.analysis.typeCheckingMode": "basic",
              "python.testing.pytestEnabled": true,
              "python.testing.pytestArgs": ["tests"],
              "jupyter.interactiveWindow.creationMode": "perFile",
              "[python]": {
                "editor.formatOnSave": true,
                "editor.defaultFormatter": "ms-python.black-formatter",
                "editor.formatOnSaveMode": "file"
              }
            }
            """
        ),
        force=args.force,
    )

    # 7) Starter code directly in src/
    write_file(
        target / "src" / "main.py",
        dedent(
            """\
            def hello() -> str:
                return "Hello from src!"

            if __name__ == "__main__":
                print(hello())
            """
        ),
        force=args.force,
    )

    # 8) Tests: ensure src/ is importable even outside VS Code (CI, terminal, etc.)
    write_file(
        target / "tests" / "conftest.py",
        dedent(
            """\
            import sys
            from pathlib import Path

            SRC = Path(__file__).resolve().parents[1] / "src"
            sys.path.insert(0, str(SRC))
            """
        ),
        force=args.force,
    )
    write_file(
        target / "tests" / "test_smoke.py",
        dedent(
            """\
            import main

            def test_hello():
                assert "Hello" in main.hello()
            """
        ),
        force=args.force,
    )

    # 9) Python version hint for uv (and for humans)
    pyver = args.python or f"{sys.version_info.major}.{sys.version_info.minor}"
    write_file(target / ".python-version", f"{pyver}\n", force=args.force)

    # 10) pyproject.toml — minimal project metadata + dev tools
    write_file(
        target / "pyproject.toml",
        dedent(
            f"""\
            [project]
            name = "{project_name}"
            version = "0.1.0"
            description = ""
            readme = "README.md"
            requires-python = ">={pyver}"
            dependencies = []

            [dependency-groups]
            dev = [
              "ipykernel",
              "pytest",
              "ruff",
              "black"
            ]
            """
        ),
        force=args.force,
    )

    # 11) README + gitignore
    write_file(
        target / "README.md",
        dedent(
            f"""\
            # {project_name}

            ## Quick start

            - Create/sync env: `uv sync`
            - Run code: `uv run python src/main.py`
            - Run tests: `uv run pytest -q`

            ## Bootstrap on a new machine

            After cloning:
            - `python {EMBEDDED_SCRIPT_NAME} bootstrap`

            ## Layout

            - `data/` — datasets (ignored by default)
            - `notebooks/` — Jupyter notebooks
            - `src/` — Python code (flat; add subfolders if needed)
            """
        ),
        force=args.force,
    )

    write_file(
        target / ".gitignore",
        dedent(
            """\
            # uv / venv
            .venv/
            # Python
            __pycache__/
            *.py[cod]
            .pytest_cache/
            .ruff_cache/
            # Jupyter
            .ipynb_checkpoints/
            # OS/editor
            .DS_Store
            Thumbs.db
            """
        ),
        force=args.force,
    )

    # Optional: initialize git + initial commit
    if args.git_init:
        which_or_die("git", "Install Git and ensure it is available in your shell.")
        if not (target / ".git").exists():
            run(["git", "init"], cwd=target)
        run(["git", "add", "-A"], cwd=target)
        run(["git", "commit", "-m", "Initial project scaffold"], cwd=target)

    # Optional: create/sync .venv and install dev deps
    if args.uv_sync:
        which_or_die("uv", "Install uv and ensure 'uv' is on PATH.")
        run(["uv", "sync"], cwd=target)

    print("\nDone.")
    print(f"Project: {target}")
    print("Next:")
    print(f"  cd {target}")
    print("  uv sync")
    print("  uv run python src/main.py")
    print(f"  (after cloning elsewhere) python {EMBEDDED_SCRIPT_NAME} bootstrap")


# ----------------------------
# Command: bootstrap (existing)
# ----------------------------

def bootstrap(args: argparse.Namespace) -> None:
    """
    Recreate/sync the project's local virtual environment from pyproject.toml + uv.lock.

    This is designed for the “I just cloned this repo onto a new machine” case.
    """
    which_or_die("uv", "Install uv and ensure 'uv' is on PATH.")
    root = find_project_root(Path.cwd())

    cmd = ["uv", "sync"]
    if args.all_extras:
        cmd.append("--all-extras")
    if args.all_groups:
        cmd.append("--all-groups")
    if args.no_dev:
        cmd.append("--no-dev")
    if args.locked:
        cmd.append("--locked")

    run(cmd, cwd=root)

    print("\nEnvironment ready.")
    if os.name == "nt":
        print(r"Activate (PowerShell): .venv\Scripts\activate")
    else:
        print("Activate (bash/zsh): source .venv/bin/activate")
    print("Or run without activating:")
    print("  uv run python -c \"import sys; print(sys.executable)\"")


# ----------------------------
# CLI / argument parsing
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    """
    Define the command line interface.

    Subcommands:
      init      create a new project scaffold
      bootstrap sync environment for an existing project
    """
    p = argparse.ArgumentParser(
        prog="project_setup",
        description="Project scaffold + bootstrap for uv + VS Code (cross-platform)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Create a new project scaffold")
    p_init.add_argument("path", help="Project directory to create")
    p_init.add_argument("--name", help="Project name for pyproject (default: folder name)")
    p_init.add_argument("--python", help="Python version for .python-version and requires-python (e.g. 3.12)")
    p_init.add_argument("--git-init", action="store_true", help="Run git init + initial commit")
    p_init.add_argument("--uv-sync", action="store_true", help="Run uv sync after creating files")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing files / allow non-empty dir")
    p_init.set_defaults(func=init_project)

    p_boot = sub.add_parser("bootstrap", help="Create/sync .venv + install deps for an existing project")
    p_boot.add_argument("--all-extras", action="store_true", help="Include all optional dependencies")
    p_boot.add_argument("--all-groups", action="store_true", help="Include all dependency groups")
    p_boot.add_argument("--no-dev", action="store_true", help="Exclude dev dependency group")
    p_boot.add_argument("--locked", action="store_true", help="Do not update uv.lock (fail if out of date)")
    p_boot.set_defaults(func=bootstrap)

    return p


def main() -> None:
    """Entry point."""
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
