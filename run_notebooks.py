#!/usr/bin/env python3

import os
import shutil
import sys

import papermill as pm
from ruamel.yaml import YAML

# Disable ipykernel warnings
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

RUN_CONFIG = os.path.join(REPO_ROOT, "config", ".run_config.yml")

ALL_NOTEBOOKS = (
    "00a-generate_data",
    "00b-extract_training_data",
    "00c-extract_grid_data",
    "01-create_classifiers",
    "02-create_probability_maps",
    "03-create_probability_animations",
    "04-create_erosion_distribution",
    "05-create_preservation_maps",
    "06-create_preservation_animations",
    "07-partial_dependence",
    "08-time_series",
)

# Path keys in all_notebooks that should be resolved to absolute paths
_PATH_KEYS = ("output_dir", "extracted_data_dir", "regions_filename")


def _resolve_paths(data, root):
    """Resolve relative path values to absolute paths relative to *root*."""
    for section_key in list(data.keys()):
        section = data[section_key]
        if not isinstance(section, dict):
            continue
        for key in _PATH_KEYS:
            if key in section and section[key] is not None:
                value = str(section[key])
                if not os.path.isabs(value):
                    section[key] = os.path.join(root, value)
        # Handle nested defaults section
        if section_key == "defaults":
            _resolve_paths(section, root)


def _prepare_config(config_path):
    """Load config, resolve paths, write .run_config.yml and snapshot."""
    yaml = YAML()
    yaml.preserve_quotes = True

    config_path = os.path.abspath(config_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        data = yaml.load(f)

    run_name = data.get("run_name", "default")

    # Resolve relative paths to absolute
    _resolve_paths(data, REPO_ROOT)

    # Write resolved config to config/.run_config.yml
    os.makedirs(os.path.dirname(RUN_CONFIG), exist_ok=True)
    with open(RUN_CONFIG, "w") as f:
        yaml.dump(data, f)

    # Write config snapshot for reproducibility
    output_dir = data.get("all_notebooks", {}).get("output_dir", "outputs")
    snapshot_dir = os.path.join(output_dir, run_name)
    os.makedirs(snapshot_dir, exist_ok=True)
    snapshot_path = os.path.join(snapshot_dir, "config_snapshot.yml")
    shutil.copy2(config_path, snapshot_path)

    return run_name


def _select_notebooks(prefixes):
    """Return notebook names from ALL_NOTEBOOKS matching the given prefixes."""
    selected = []
    for prefix in prefixes:
        matches = [nb for nb in ALL_NOTEBOOKS if nb.startswith(prefix)]
        if not matches:
            raise ValueError(
                f"No notebook matching prefix '{prefix}'. "
                f"Use --list-defaults to see available notebooks."
            )
        selected.extend(matches)
    # Preserve order from ALL_NOTEBOOKS, deduplicate
    seen = set()
    ordered = []
    for nb in ALL_NOTEBOOKS:
        if nb in selected and nb not in seen:
            seen.add(nb)
            ordered.append(nb)
    return ordered


def run_notebook(
    input_filename: str,
    output_filename=None,
    parameters=None,
):
    if not input_filename.endswith(".ipynb"):
        input_filename += ".ipynb"
    if not os.path.isfile(input_filename):
        raise FileNotFoundError(
            f"Input file not found: {input_filename}"
        )
    if output_filename is None:
        output_filename = input_filename[:-6] + "_output.ipynb"
    print(f"Running notebook: {input_filename}", file=sys.stderr)
    print(f"Output file: {output_filename}", file=sys.stderr)
    pm.execute_notebook(
        input_filename,
        output_filename,
        parameters,
        kernel_name="python3",
        cwd=os.path.dirname(os.path.abspath(input_filename)),
    )


def _main(args):
    if args.list_defaults:
        print(
            "Default notebooks to execute:",
            *[
                f" - {f}.ipynb"
                for f in ALL_NOTEBOOKS
            ],
            sep="\n",
            flush=True,
        )
        return 0

    if args.config is None:
        raise ValueError("--config is required.")

    run_name = _prepare_config(args.config)
    print(f"Run name: {run_name}", file=sys.stderr)
    print(f"Config written to: {RUN_CONFIG}", file=sys.stderr)

    # Determine which notebooks to run
    if args.notebooks:
        notebooks = _select_notebooks(args.notebooks)
    else:
        notebooks = list(ALL_NOTEBOOKS)

    for notebook_name in notebooks:
        filename = os.path.join(REPO_ROOT, notebook_name)
        if args.overwrite:
            output_filename = filename
        else:
            output_filename = None
        run_notebook(filename, output_filename, parameters=None)
    return 0


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Execute notebooks from the command line.",
    )
    parser.add_argument(
        "--config",
        help="path to config YAML file (e.g. config/run_X.yml)",
        dest="config",
    )
    parser.add_argument(
        "--notebooks",
        help="notebook prefixes to run (e.g. 00b 00c 01)",
        nargs="+",
        dest="notebooks",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        help="overwrite input files",
        action="store_true",
        dest="overwrite",
    )
    parser.add_argument(
        "-l",
        "--list-defaults",
        help="list default input files",
        action="store_true",
        dest="list_defaults",
    )
    args = parser.parse_args()
    _main(args)
