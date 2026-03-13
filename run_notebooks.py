#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import papermill as pm
from ruamel.yaml import YAML

# Disable ipykernel warnings
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

REPO_ROOT = Path(__file__).parent.resolve()
RUN_CONFIG_PATH = REPO_ROOT / "config" / ".run_config.yml"

# Maps short notebook codes to filenames (pipeline order)
NOTEBOOK_MAP = {
    "00a": "00a-generate_data",
    "00b": "00b-extract_training_data",
    "00c": "00c-extract_grid_data",
    "00d": "00d-extract_mantle_features",
    "01":  "01-create_classifiers",
    "02":  "02-create_probability_maps",
    "03":  "03-create_probability_animations",
    "04":  "04-create_erosion_distribution",
    "05":  "05-create_preservation_maps",
    "06":  "06-create_preservation_animations",
    "07":  "07-partial_dependence",
    "08":  "08-time_series",
}


def _resolve_config_paths(config):
    """Resolve relative data_dir and output_dir to absolute paths (repo-root-relative)."""
    all_nb = config.get("all_notebooks", {})
    for key in ("data_dir", "output_dir"):
        val = all_nb.get(key)
        if val and not Path(str(val).strip()).is_absolute():
            all_nb[key] = str(REPO_ROOT / str(val).strip())


def prepare_run(config_path):
    """Copy config to config/.run_config.yml and write config_snapshot.yml."""
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096  # prevent line-wrapping of long paths

    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.load(f)

    _resolve_config_paths(config)

    # Write run config (used by all notebooks)
    with open(RUN_CONFIG_PATH, "w") as f:
        yaml.dump(config, f)

    # Write config snapshot to outputs/{run_name}/ for reproducibility
    run_name = config.get("all_notebooks", {}).get("run_name", "baseline")
    output_dir_base = config.get("all_notebooks", {}).get("output_dir", str(REPO_ROOT / "output"))
    snapshot_dir = Path(output_dir_base) / run_name
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    with open(snapshot_dir / "config_snapshot.yml", "w") as f:
        yaml.dump(config, f)

    print(f"Config:          {config_path}", file=sys.stderr)
    print(f"Run config:      {RUN_CONFIG_PATH}", file=sys.stderr)
    print(f"Config snapshot: {snapshot_dir / 'config_snapshot.yml'}", file=sys.stderr)


def run_notebook(input_filename, output_filename=None, parameters=None):
    if not input_filename.endswith(".ipynb"):
        input_filename += ".ipynb"
    if not os.path.isfile(input_filename):
        raise FileNotFoundError(f"Input file not found: {input_filename}")
    if output_filename is None:
        output_filename = input_filename[:-6] + "_output.ipynb"
    print(f"Running notebook: {input_filename}", file=sys.stderr)
    print(f"Output file:      {output_filename}", file=sys.stderr)
    pm.execute_notebook(
        input_filename,
        output_filename,
        parameters,
        kernel_name="python3",
        cwd=os.path.dirname(os.path.abspath(input_filename)),
    )


def _resolve_notebook_names(names):
    """Map short codes (e.g. '00b') to full filenames; pass through full names unchanged."""
    return [NOTEBOOK_MAP.get(name, name) for name in names]


def _main(args):
    if args.setup:
        from lib.setup_run import run_setup
        run_setup(args.config)
        return 0

    if args.list_defaults:
        print(
            "Available notebooks:",
            *[f"  {code}: {name}.ipynb" for code, name in NOTEBOOK_MAP.items()],
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

    prepare_run(args.config)

    filenames = _resolve_notebook_names(args.notebooks)

    # Validate all notebooks exist before starting any
    for filename in filenames:
        full = filename if filename.endswith(".ipynb") else filename + ".ipynb"
        if not os.path.isfile(full):
            raise FileNotFoundError(f"Notebook not found: {full}")

    for filename in filenames:
        output_filename = filename if args.overwrite else None
        run_notebook(filename, output_filename, parameters=None)

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
        "--setup",
        help="prepare run directories and plate model, then exit",
        action="store_true",
        dest="setup",
    )
    args = parser.parse_args()
    _main(args)
