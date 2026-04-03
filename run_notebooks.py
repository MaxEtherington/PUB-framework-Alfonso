#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import papermill as pm

# Local imports
from .paths import PathConfigManager

p: PathConfigManager = None

# Disable ipykernel warnings
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"


def _get_notebook_path(name) -> Path:
    """Find the path to a notebook given a portion or all of its name."""
    candidates = list(PathConfigManager.ROOT.glob(f"*{name}*.ipynb"))
    if not candidates:
        raise FileNotFoundError(f"No notebook found containing '{name}'")
    if len(candidates) > 1:
        raise ValueError(f"Multiple notebooks found for '{name}': {candidates}")
    return candidates[0]


def _get_notebook_filepaths(names) -> list[Path]:
    """Find and validate paths to notebooks given portions or all of their names."""
    missing = []
    ambiguous = []
    paths = []
    for name in names:
        try:
            paths.append(_get_notebook_path(name))
        except FileNotFoundError as e:
            missing.append(str(e))
        except ValueError as e:
            ambiguous.append(str(e))
    if missing:
        raise FileNotFoundError("One or more notebooks not found:\n" + "\n  ".join(missing))
    if ambiguous:
        raise ValueError("One or more notebook names are ambiguous:\n" + "\n  ".join(ambiguous))
    return paths


def _copy_config_to(output_path):
    """Copy the active config file to an output directory"""
    if not p.CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Could not copy config: Config file not found: {p.CONFIG_PATH}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(p.CONFIG_PATH.read_text())


def _find_config_path(config_path) -> Path:
    """Resolve shorthand config path to canonical path."""
    path = Path(config_path).resolve()
    if not path.is_file():
        path = PathConfigManager.CONFIG_DIR / config_path
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path} (tried {path})")
    return path


def _prepare_run(config_path):
    """Copy config to config/.run_config.yml and write config_snapshot.yml."""
    config_path = _find_config_path(config_path)
    
    global p
    p = PathConfigManager(config_path)

    # Write run config (used by all notebooks)
    _copy_config_to(p.RUN_CONFIG_PATH)

    # Write config snapshot to outputs/{run_name}/ for reproducibility
    _copy_config_to(p.OUTPUT_DIR / "config_snapshot.yml")
    
    # Update paths and create directories based on chosen config
    p.create_directories()
    
    # Validate expected input paths exist (e.g. deposits, mantle features)
    p.validate_input_paths()

    print(f"Config:          {p.CONFIG_PATH}", file=sys.stderr)
    print(f"Run config:      {p.RUN_CONFIG_PATH}", file=sys.stderr)
    print(f"Config snapshot: {p.OUTPUT_DIR / 'config_snapshot.yml'}", file=sys.stderr)


def run_notebook(
    input_nb_filepath: Path, 
    output_nb_filepath: Path = None, 
    parameters=None
):
    """Run a notebook via papermill, with optional parameters and output path.
       If output_nb_filepath is not provided, defaults to input_nb_filepath with '_output' suffix"""

    input_nb_filepath = Path(input_nb_filepath)
    
    if not input_nb_filepath.is_file():
        raise FileNotFoundError(f"Input notebook not found: {input_nb_filepath}")
    if output_nb_filepath is None:
        output_nb_filepath = input_nb_filepath.with_name(input_nb_filepath.stem + "_output.ipynb")
    
    print(f"Running notebook: {input_nb_filepath}", file=sys.stderr)
    print(f"Output file:      {output_nb_filepath}", file=sys.stderr)
    
    pm.execute_notebook(
        input_nb_filepath,
        output_nb_filepath,
        parameters,
        kernel_name="python3",
        cwd=input_nb_filepath.resolve().parent,
    )


def _main(args):
    if args.list_defaults:
        print(
            "Available notebooks:",
            *[f"  {filepath.name}" for filepath in sorted(PathConfigManager.ROOT.glob("*.ipynb"))],
            sep="\n",
            flush=True,
        )
        return 0

    if args.config is None:
        raise ValueError(
            "--config is required. Use --list-defaults to see available notebooks.\n"
            "  Example: python run_notebooks.py --config config/notebook_parameters_default.yml --notebooks 00b 01"
        )

    if not args.notebooks:
        raise ValueError(
            "Must specify at least one notebook via --notebooks (e.g. --notebooks 00b 00c 01)."
        )

    # Prepare paths and directories based on config,
    # validate inputs, create config snapshot, etc.
    _prepare_run(args.config)

    # Collect and validate notebook paths
    notebook_filepaths = _get_notebook_filepaths(args.notebooks)

    if args.setup:
        from lib.setup_run import run_setup
        run_setup(p)
        return 0


    for notebook_filepath in notebook_filepaths:
        output_filepath = notebook_filepath if args.overwrite else None
        run_notebook(notebook_filepath, output_filepath, parameters=None)

    return 0


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Execute pipeline notebooks from the command line.",
    )
    parser.add_argument(
        "--config",
        required=False,
        default=None,
        metavar="CONFIG",
        help="path to config YAML file; required unless --list-defaults is set",
        dest="config",
    )
    parser.add_argument(
        "--notebooks",
        nargs="+",
        metavar="NOTEBOOK",
        help="notebook codes to run in order (e.g. 00b 00c 01)",
        dest="notebooks",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        help="overwrite input notebook files with execution output",
        action="store_true",
        dest="overwrite",
    )
    parser.add_argument(
        "-l",
        "--list-defaults",
        help="list available notebook codes",
        action="store_true",
        dest="list_defaults",
    )
    parser.add_argument(
        "--cache-remote",
        help="cache remotely-hosted data (e.g. plate models), then exit; use when notebooks are run on a machine without internet access (e.g. HPC)",
        action="store_true",
        dest="setup",
    )
    args = parser.parse_args()
    _main(args)
