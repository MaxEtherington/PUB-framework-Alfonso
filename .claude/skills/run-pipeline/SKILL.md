---
name: run-pipeline
description: Constructs and executes run_notebooks.py commands to run one or more pipeline notebooks (00a–08) with a given config YAML. Use when user says 'run notebook', 'run the pipeline', 'execute 00b and 00c', 'generate prospectivity maps', or asks how to produce output. Handles --config, --notebooks, -o, --setup, --cache-remote flags, and output inspection in output/{run_name}/. Do NOT use for editing notebook logic or lib/ code.
paths:
  - run_notebooks.py
  - config/*.yml
---
# Run Pipeline

## Critical

- **Never edit notebook files directly.** All notebook logic lives in `nb_scripts/*.py`. The `PostToolUse` hook in `.claude/settings.json` automatically runs `jupytext --sync` after any edit.
- `--config` is required for every run except `--list-defaults`.
- Config shorthand is resolved relative to `config/`: `notebook_parameters_default` expands to `config/notebook_parameters_default.yml`.
- The active config is written to `config/.run_config.yml` (gitignored) before notebooks execute — this file must exist for notebooks to run.
- Always run `--setup` first when changing configs or on a fresh clone, to validate input paths and create directories before running notebooks.
- The conda environment must be active: `conda activate prospectivity`.

## Instructions

### 1. Identify the config file

All config YAMLs live in `config/`. The default is `config/notebook_parameters_default.yml`.

Key parameters to check before running:
- `run_name` (under `all_notebooks`) — determines the output subdirectory under `output/`
- `plate_model.plate_model_name` — e.g. `zahirovic2022`
- `timespan` section (`min` / `max`) in Ma
- `feature_sets.subduction.enabled`, `feature_sets.mantle.enabled`, etc.
- `deposits_filename` — must exist in `data_source/deposits/`

Verify the config file exists before proceeding:
```bash
ls config/
```

### 2. Validate config and set up directories

Always run `--setup` before the first run with a new config or after editing a config:

```bash
python run_notebooks.py --config config/notebook_parameters_default.yml --setup
```

This:
- Writes `config/.run_config.yml`
- Writes the config snapshot to the output directory
- Validates that source data paths exist (deposits CSV, regions GeoJSON, mantle data dir if enabled)
- Creates all required output directories

Verify the output shows no `FileNotFoundError` before proceeding.

### 3. List available notebooks (if needed)

```bash
python run_notebooks.py --list-defaults
```

Notebook codes and their purpose:
| Code | Purpose |
|------|--------|
| `00a` | Generate/download plate model data |
| `00b` | Extract training data (deposit points) |
| `00c` | Extract grid data (regular grid for maps) |
| `01` | Train PU classifiers |
| `02` | Create prospectivity probability maps |
| `03` | Create probability animations |
| `04` | Erosion/preservation distribution analysis |
| `05` | Create preservation maps |
| `06` | Create preservation animations |
| `07` | Partial dependence plots |
| `08` | Time series analysis |

Full pipeline order: `00a → 00b → 00c → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08`

### 4. Run notebooks

**Run one or more notebooks (execution output saved alongside source notebooks):**
```bash
python run_notebooks.py --config config/notebook_parameters_default.yml --notebooks 00b 00c
```

**Run and overwrite the source notebook in-place:**
```bash
python run_notebooks.py --config config/notebook_parameters_default.yml --notebooks 01 -o
```

**Run full pipeline from data extraction through classifiers:**
```bash
python run_notebooks.py --config config/notebook_parameters_default.yml --notebooks 00a 00b 00c 01
```

Verify each notebook completes without error before running the next stage.

### 5. Inspect outputs

Outputs land in the output directory configured by `run_name` in `all_notebooks`. Use `ls output/` to find the subdirectory, then inspect its contents:

```bash
ls output/mantle_test/
cat output/mantle_test/config_snapshot.yml
```

Key output files:
- `config_snapshot.yml` in the run directory — exact config used for the run
- `*.joblib` files — trained classifier objects (after notebook `01`)
- `probability_grids/` subdirectory — prospectivity rasters (after notebook `02`)
- `figures/` subdirectory — plots and maps

## Examples

**User says:** "Run the training data extraction and classifier notebooks with the default config."

**Actions:**
```bash
# Step 1: validate config and set up dirs
python run_notebooks.py --config config/notebook_parameters_default.yml --setup

# Step 2: extract training data and grid data
python run_notebooks.py --config config/notebook_parameters_default.yml --notebooks 00b 00c

# Step 3: train classifiers
python run_notebooks.py --config config/notebook_parameters_default.yml --notebooks 01
```

**Result:** Execution output notebooks saved. Trained classifiers written to the configured output directory.

---

**User says:** "Re-run notebook 02 and save the output into the notebook file itself."

**Actions:**
```bash
python run_notebooks.py --config config/notebook_parameters_default.yml --notebooks 02 -o
```

**Result:** `02-create_probability_maps.ipynb` is overwritten with execution output in-place.

## Common Issues

**`FileNotFoundError: Expected source files not found` for deposits CSV**
- The deposits CSV is missing. Check `data_source/deposits/` for available files, then update `deposits_filename` in the config YAML.

**`FileNotFoundError: Config file not found: myconfig`**
- The config shorthand must resolve to an existing file under `config/`. Run `ls config/` to see available YAMLs.

**`ValueError: --config is required`**
- Every command except `--list-defaults` requires `--config`. Add `--config config/notebook_parameters_default.yml`.

**`ValueError: Must specify at least one notebook via --notebooks`**
- Add `--notebooks` with at least one code, e.g. `--notebooks 01`.

**`FileNotFoundError: No notebook found containing '00d'`**
- The notebook code doesn't exist. Run `--list-defaults` to see valid codes.

**`KeyError: Missing required config key: 'run_name'`**
- The config YAML is missing a required key. Compare against `config/notebook_parameters_default.yml`; the `defaults` section at the bottom defines all required keys.

**Notebook hangs or crashes with mantle feature errors**
- Check `feature_sets.mantle.data_dir` in the config matches a directory under `data_source/mantle_outputs/`. Run `ls data_source/mantle_outputs/`.

**`conda: command not found` or import errors**
- Activate the environment first: `conda activate prospectivity`. If the env doesn't exist: `conda env create --file environment.yml`.
