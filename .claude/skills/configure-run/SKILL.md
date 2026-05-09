---
name: configure-run
description: Creates or modifies a config YAML in config/ following the 3-level merge pattern (defaults → all_notebooks → notebook_XXX). Use when user says 'new config', 'change plate model', 'set up a run with X', 'configure for zahirovic2022', 'adjust timespan', 'enable mantle features', or 'change n_jobs'. References lib/load_params.py:get_params() merge logic and lib/paths.py:PathConfigManager for path derivation. Do NOT use for running notebooks — that is run_notebooks.py's job.
paths:
  - config/*.yml
---
# Configure Run

## Critical

- **Never modify the `defaults` block** in any config file. It is the last-resort fallback read by `lib/load_params.py:get_params()` and must be left intact.
- **Never point notebooks at `notebook_parameters_default.yml` directly.** Notebooks read only `config/.run_config.yml`, which is written at runtime by `run_notebooks.py`.
- **Never hardcode derived paths** (`data_dir`, `output_dir`, `extracted_data_dir`, etc.). All paths are resolved by `lib/paths.py:PathConfigManager` from the config keys `plate_model_name`, `reference_feature`, and `study_zone_buffer`.
- When `feature_sets.mantle.enabled: true`, the `feature_sets.mantle.data_dir` key **must** point to a directory that exists under `data_source/mantle_outputs/`. If it doesn't, `PathConfigManager.validate_input_paths()` will raise `FileNotFoundError`.
- `run_name` controls the output directory: `output/{run_name}/`. Choose a name that is unique and descriptive — collisions will mix outputs from different runs unless `overwrite_output: true`.

## Instructions

### Step 1 — Copy the canonical template

Copy an existing config as your starting point. `config/notebook_parameters_default.yml` is the canonical template; for Zahirovic2022 runs use `config/zahirovic_baseline.yml`:

```bash
cp config/notebook_parameters_default.yml config/{run_name}.yml
```

Verify the new file exists before editing.

### Step 2 — Edit `all_notebooks` (run-level settings)

All keys in `all_notebooks` apply to every notebook unless overridden in a per-notebook section. Set the keys you want to change; omit keys you want to fall through to `defaults`.

Minimum required keys to set:

```yaml
all_notebooks:
  run_name: my_run_name          # → output/my_run_name/
  plate_model:
    use_provided_plate_model: false   # false = fetch via PlateModelManager
    plate_model_name: zahirovic2022   # names both PMM model and data/ subdir
  timespan:
    min: 0
    max: 400                     # Ma; zahirovic2022 supports up to 400

  use_extracted_data: true
  deposits_filename: "deposits-Etherington.csv"  # or "deposits.csv"
  regions_filename: "regions.geojson"
  reference_feature: trenches
  study_zone_buffer: 6.0

  feature_sets:
    subduction:
      enabled: true
    crustal:
      enabled: false
    mantle:
      enabled: false
      data_dir: zahirovic2022         # required even when disabled; must match a subdir of data_source/mantle_outputs/
    erodep:
      enabled: false

  n_jobs: 8
  verbose: true
  random_seed: 1234
  overwrite_output: false
```

**Verify**: Every key in `all_notebooks` that you set must also appear in the `defaults` block at the bottom (as a type check). If a key is absent from `defaults`, it will not be merged correctly.

### Step 3 — Override per-notebook settings (optional)

Per-notebook sections inherit all `all_notebooks` values. Set a key to a non-null value to override; set it to bare null (empty) to explicitly inherit from `all_notebooks`. Only override what genuinely differs per notebook.

Common per-notebook overrides:

```yaml
notebook_00b:
  num_unlabelled: 200          # number of unlabelled training points
  deposits_filename: "deposits-Etherington.csv"  # override deposit list for this notebook only

notebook_00c:
  grid_resolution: 0.5         # degrees; 0.5 is standard, 1.0 for faster testing

notebook_01:
  use_same_features: true      # use same feature set across all regions
  automatic_feature_selection: false
  create_svm: true
  create_regional_models: true

notebook_04:
  outlier_contamination: 0.06  # IsolationForest contamination fraction

notebook_07:
  grid_resolution: 100         # PDP grid points (not spatial degrees)
```

Leave all other per-notebook keys as bare nulls to inherit. Do not delete per-notebook sections entirely — the loader expects them.

**Verify**: Open `lib/load_params.py:get_params()` (lines 1–34) and trace the merge for one notebook to confirm your overrides will take effect.

### Step 4 — Leave `defaults` unchanged

The `defaults` block at the bottom of the file **must not be modified**. It is the last-resort fallback. Copy it verbatim from the template you used in Step 1.

**Verify**: `diff config/{run_name}.yml config/notebook_parameters_default.yml | grep '^>' | grep defaults` should be empty (no additions to the defaults block).

### Step 5 — Validate with a dry-run setup

```bash
python run_notebooks.py --config config/{run_name}.yml --setup
```

This runs `lib/paths.py:PathConfigManager.validate_input_paths()` and creates required directories without executing any notebook. Fix any `FileNotFoundError` or `KeyError: Missing required config key` errors before proceeding.

## Examples

**User says**: "Set up a new run using zahirovic2022 with mantle features enabled, 0–350 Ma, and call it `mantle_v2`."

**Actions taken**:
1. `cp config/notebook_parameters_default.yml config/mantle_v2.yml`
2. Edit `all_notebooks`:
   ```yaml
   run_name: mantle_v2
   plate_model:
     use_provided_plate_model: false
     plate_model_name: zahirovic2022
   timespan:
     min: 0
     max: 350
   deposits_filename: "deposits-Etherington.csv"
   feature_sets:
     subduction:
       enabled: false
     crustal:
       enabled: false
     mantle:
       enabled: true
       data_dir: zahirovic2022_cratonfix   # must exist in data_source/mantle_outputs/
     erodep:
       enabled: false
   ```
3. In `notebook_01`, set `create_svm: false` (mantle runs skip SVM by convention).
4. Run `python run_notebooks.py --config config/mantle_v2.yml --setup` — passes with no errors.

**Result**: `config/mantle_v2.yml` is ready; outputs will go to `output/mantle_v2/`; `PathConfigManager.MANTLE_DATA_DIR` resolves to `data_source/mantle_outputs/zahirovic2022_cratonfix`.

## Common Issues

**`KeyError: Missing required config key: 'feature_sets'`**
Your `all_notebooks` block is missing the `feature_sets` nested dict. Add the full block from the template (Step 2). This error is raised by `lib/paths.py:PathConfigManager.update_paths()` line 58.

**`FileNotFoundError: Expected source files not found: data_source/mantle_outputs/{data_dir}`**
The `feature_sets.mantle.data_dir` value does not match any directory under `data_source/mantle_outputs/`. Run `ls data_source/mantle_outputs/` to see available directories and correct the config.

**`FileNotFoundError: ... deposits/{deposits_filename}`**
The `deposits_filename` value does not exist in `data_source/deposits/`. Valid options: `deposits.csv` (Alfonso original) or `deposits-Etherington.csv` (extended DB). Check with `ls data_source/deposits/`.

**Per-notebook override not taking effect**
`get_params()` only applies a per-notebook value when it is non-null OR the key is not already in `all_notebooks`. A bare null in `notebook_XXX` means "inherit". If you want to *clear* an `all_notebooks` value for one notebook, set an explicit value — not null.

**`overwrite_output: false` causes stale outputs**
If re-running with new config but same `run_name`, previous outputs are not overwritten. Either change `run_name` or set `overwrite_output: true` in `all_notebooks`.