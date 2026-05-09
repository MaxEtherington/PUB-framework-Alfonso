---
name: run-notebook-pipeline
description: Runs one or more pipeline notebooks via run_notebooks.py with a config YAML. Use when user says 'run the pipeline', 'execute notebooks', 'run 00b and 01', 'process training data', 'set up directories', or asks to run any combination of notebooks 00a–08. Handles --config, --notebooks, --setup, --cache-remote, and -o flags. Do NOT use for editing notebook cell content or modifying config files.
---
# Run Notebook Pipeline

## Critical

- **Never run notebooks directly with Jupyter or papermill.** Always use `run_notebooks.py` — it resolves config, creates directories, and writes `config/.run_config.yml` which every notebook reads.
- **`--config` is always required** unless using `--list-defaults`.
- Config is passed via `config/.run_config.yml` on disk — not via papermill parameters. Do not attempt to inject parameters any other way.
- The `conda activate prospectivity` environment must be active before running any command.
- Notebooks write output to `output/{run_name}/` as defined by `run_name` in the config's `all_notebooks` section.

## Instructions

### 1. Identify the config file

Available configs live in `config/`. Current options:

```
config/notebook_parameters_default.yml   ← general-purpose baseline
config/zahirovic_baseline.yml            ← Zahirovic2022 reconstruction
config/mantle_test.yml                   ← with mantle feature integration
config/cache_test.yml
```

If the user has not specified a config, ask which reconstruction/run they intend. Do not guess.

Verify the file exists before proceeding: `ls config/<chosen>.yml`

### 2. Identify which notebooks to run

Map user intent to notebook codes:

| Code | File | Purpose |
|------|------|---------|
| `00a` | `00a-generate_data.ipynb` | Download plate model + generate rasters |
| `00b` | `00b-extract_training_data.ipynb` | Extract features for deposit points |
| `00c` | `00c-extract_grid_data.ipynb` | Extract features for 0.5° prediction grid |
| `01` | `01-create_classifiers.ipynb` | Train PU + SVM classifiers |
| `02` | `02-create_probability_maps.ipynb` | Prospectivity maps |
| `03` | `03-create_probability_animations.ipynb` | Time-animated maps |
| `04` | `04-create_erosion_distribution.ipynb` | Erosion/deposition analysis |
| `05` | `05-create_preservation_maps.ipynb` | Preservation maps |
| `06` | `06-create_preservation_animations.ipynb` | Preservation animations |
| `07` | `07-partial_dependence.ipynb` | PDP plots |
| `08` | `08-time_series.ipynb` | Time series analysis |

Notebook codes are matched by substring — `00b` matches `00b-extract_training_data.ipynb`.

List available notebooks if unsure:
```bash
python run_notebooks.py --list-defaults
```

### 3. Choose the right command form

**Run one or more notebooks:**
```bash
python run_notebooks.py --config config/<name>.yml --notebooks <codes...>
```

**Setup directories only (no execution):**
```bash
python run_notebooks.py --config config/<name>.yml --setup
```
Use this to validate config and create output directories before a run, or before executing on HPC.

**Pre-cache remote data for HPC (no internet later):**
```bash
python run_notebooks.py --config config/<name>.yml --cache-remote
```
Run this before transferring to a machine without internet access.

**Overwrite notebook files in-place with execution output:**
```bash
python run_notebooks.py --config config/<name>.yml --notebooks <codes...> -o
```
Without `-o`, output is written to `<notebook>_output.ipynb` alongside the source file.

### 4. Verify the run completed

After execution, confirm:
- `config/.run_config.yml` was updated (modification time should be recent)
- `output/{run_name}/config_snapshot.yml` exists
- Expected output files exist under `output/{run_name}/`

```bash
ls output/<run_name>/
```

## Examples

**User says:** "Run the training data extraction and classifier notebooks with the zahirovic baseline config"

**Actions taken:**
1. Config identified: `config/zahirovic_baseline.yml` (run_name: `zahirovic-baseline`)
2. Notebooks identified: `00b` (extract training data), `01` (create classifiers)
3. Command constructed and run:

```bash
python run_notebooks.py --config config/zahirovic_baseline.yml --notebooks 00b 01
```

**Result:** `config/.run_config.yml` written, directories created, both notebooks executed in order, output at `output/zahirovic-baseline/`.

---

**User says:** "Set up directories for a mantle test run before I transfer to HPC"

```bash
# Step 1: prepare dirs and validate config
python run_notebooks.py --config config/mantle_test.yml --setup

# Step 2: cache remote plate model data
python run_notebooks.py --config config/mantle_test.yml --cache-remote
```

---

**User says:** "Run the full pipeline"

```bash
python run_notebooks.py --config config/<chosen>.yml --notebooks 00a 00b 00c 01 02
```

Ask the user which config to use if not specified. Notebooks 03–08 are optional analysis/visualisation steps.

## Common Issues

**`FileNotFoundError: Config file not found: config/foo.yml`**
- Config shorthand is resolved relative to `config/`. Run `ls config/` to see available files.
- You can pass a full path: `--config /absolute/path/to/config.yml`

**`ValueError: --config is required`**
- `--config` was omitted. Always required unless using `--list-defaults`.

**`ValueError: Must specify at least one notebook via --notebooks`**
- Neither `--notebooks`, `--setup`, nor `--cache-remote` was provided. Add at least one.

**`FileNotFoundError: No notebook found containing '00x'`**
- The notebook code does not match any `.ipynb` file in the repo root. Run `python run_notebooks.py --list-defaults` to see valid names.

**`ValueError: Multiple notebooks found for '00'`**
- The code is too short and matches more than one notebook. Use a more specific code (e.g. `00b` not `00`).

**Notebook fails mid-run with import errors or missing data**
- Confirm `conda activate prospectivity` is active: `conda info --envs`
- If data is missing, run `00a` first to generate rasters, then re-run the failing notebook.
- If `config/.run_config.yml` is stale from a previous run with different settings, run `--setup` to refresh it before re-executing.

**Output notebook written to wrong location**
- Without `-o`, output goes to `<notebook>_output.ipynb` next to the source. Use `-o` to overwrite the source notebook in-place instead.