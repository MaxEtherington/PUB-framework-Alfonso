---
name: create-config
description: Creates a new config YAML in `config/` following the three-section structure (`all_notebooks`, per-notebook overrides, `defaults`) used by every existing config in this project. Use when user says 'new config', 'create a config', 'set up a run', 'test a plate model', or wants to change plate model (zahirovic2022, clennett2020, alfonso2024_default), deposit set (Porphyry-deposits.csv, IOCG-deposits.csv, VMS-deposits.csv, SedCu-deposits.csv, deposits-Etherington.csv), timespan, or feature sets. Do NOT use for modifying existing configs.
paths:
  - config/*.yml
---
# create-config

## Critical

- **Never modify `defaults:`** — this section is the fallback source of truth and must be copied verbatim from `config/notebook_parameters_default.yml`. The comment `# N.B. Do not modify this section.` must appear above it.
- The merge order is: `defaults` → `all_notebooks` → `notebook_XX`. Only values in `all_notebooks` and `notebook_XX` sections are user-controlled.
- Null values (bare keys with no value, e.g. `n_jobs:`) in per-notebook sections mean "inherit from `all_notebooks`". Do not fill them in unless the user explicitly wants a notebook-level override.
- `use_extracted_data` must always be `true` — the Zenodo `data_prepared/` fallback is not used in this workflow.
- When `mantle.enabled: true`, `data_dir` must be set to a valid mantle simulation subdirectory (e.g. `zahirovic2022`, `zahirovic2022_cratonfix`).
- If `use_provided_plate_model: true`, do not set `plate_model_name` to anything other than `alfonso2024_default`.
- Valid deposit filenames are CSV files available in `data_source/deposits/`. Run `ls data_source/deposits/` to see current options.

## Instructions

1. **Determine run identity.** Ask (or infer from context):
   - `run_name` — becomes the output subdirectory name; use kebab-case (e.g. `zahirovic-porphyry`)
   - Plate model: `use_provided_plate_model: true` (→ `alfonso2024_default`) or `false` + a `plate_model_name`
   - Timespan range in Ma (default 0–170 for Alfonso model; up to 400 for zahirovic2022)
   - `deposits_filename` — a CSV from `data_source/deposits/`
   - Which `feature_sets` to enable: `subduction`, `crustal`, `mantle`, `erodep`

   Verify: if `mantle.enabled: true`, confirm a `data_dir` value is known before proceeding.

2. **Copy the boilerplate per-notebook block verbatim.** The per-notebook sections (`notebook_00a` through `notebook_08`) are identical across all configs — copy them exactly from `config/notebook_parameters_default.yml`. Only override a key inside a `notebook_XX` block when the user explicitly requests a notebook-level difference (e.g. `grid_resolution`, `num_unlabelled`, `create_svm`, `create_regional_models`, `outlier_contamination`).

3. **Copy the `defaults:` section verbatim** from `config/notebook_parameters_default.yml` lines 130–169. Add the comment `# N.B. Do not modify this section.` directly above it. This section must not be customised.

4. **Write the file to `config/`** using a descriptive name matching the `run_name`. The section order must be:
   1. `all_notebooks:`
   2. per-notebook blocks (`notebook_00a:` … `notebook_08:`)
   3. `defaults:`

   Verify: `run_name` in the file matches the filename stem.

5. **Validate** by running:
   ```bash
   python run_notebooks.py --config config/notebook_parameters_default.yml --setup
   ```
   This sets up directory structure and validates the config without running notebooks. Fix any printed errors before handing the config to the user.

## Examples

**User says:** "Create a config for a zahirovic2022 run with Porphyry deposits, mantle features on, timespan 0–350 Ma"

**Actions taken:**
- `run_name: zahirovic-porphyry`
- `plate_model_name: zahirovic2022`, `use_provided_plate_model: false`
- Timespan: min 0, max 350
- Deposit file: a Porphyry CSV from `data_source/deposits/`
- `feature_sets.mantle.enabled: true`, `data_dir: zahirovic2022`
- All other feature sets disabled
- Per-notebook blocks and `defaults:` copied verbatim

**Result config written to `config/`:**
```yaml
all_notebooks:
  run_name: zahirovic-porphyry       # used as output directory name
  plate_model:
    use_provided_plate_model: false
    plate_model_name: zahirovic2022
  timespan:
    min: 0
    max: 350

  use_extracted_data: true

  deposits_filename: "<deposit-csv-filename>"
  regions_filename: "regions.geojson"

  reference_feature: trenches
  study_zone_buffer: 6.0 # degrees

  feature_sets:
    subduction:
      enabled: false
    crustal:
      enabled: false
    mantle:
      enabled: true
      data_dir: zahirovic2022
    erodep:
      enabled: false

  n_jobs: 8
  verbose: true
  random_seed: 1234
  overwrite_output: false


notebook_00a:
  overwrite_output: false
  n_jobs:
  timespan:
  cleanup: false

notebook_00b:
  n_jobs:
  verbose:
  timespan:
  random_seed:
  overwrite_output:
  extracted_data_dir:
  regions_filename:
  deposits_filename:
  num_unlabelled: 200

notebook_00c:
  n_jobs:
  verbose:
  timespan:
  regions_filename:
  overwrite_output: false
  grid_resolution: 0.5

notebook_00d:
  n_jobs:
  verbose:
  timespan:
  regions_filename:
  overwrite_output:

notebook_01:
  output_dir:
  n_jobs:
  verbose:
  random_seed:
  use_extracted_data:
  overwrite_output:
  use_same_features: true
  automatic_feature_selection: false
  create_svm: true
  create_regional_models: true

notebook_02:
  output_dir:
  n_jobs:
  verbose:
  use_extracted_data:
  overwrite_output:

notebook_03:
  output_dir:
  n_jobs:
  timespan:
  verbose:
  use_extracted_data:
  overwrite_output:

notebook_04:
  output_dir:
  n_jobs:
  timespan:
  verbose:
  random_seed:
  use_extracted_data:
  outlier_contamination: 0.06

notebook_05:
  output_dir:
  n_jobs:
  verbose:
  use_extracted_data:

notebook_06:
  output_dir:
  n_jobs:
  verbose:
  use_extracted_data:

notebook_07:
  output_dir:
  n_jobs:
  random_seed:
  use_extracted_data:
  overwrite_output:
  grid_resolution: 100

notebook_08:
  output_dir:
  use_extracted_data:

defaults:
  # N.B. Do not modify this section.
  run_name: default
  plate_model:
    use_provided_plate_model: true
    plate_model_name: alfonso2024_default
  timespan:
    min: 0
    max: 170

  use_extracted_data: true

  deposits_filename: "<deposit-csv-filename>"
  regions_filename: "regions.geojson"

  reference_feature: trenches
  study_zone_buffer: 6.0 # degrees

  feature_sets:
    subduction:
      enabled: false
    crustal:
      enabled: false
    mantle:
      enabled: false
      data_dir: zahirovic2022
    erodep:
      enabled: false

  n_jobs: 8
  verbose: true
  random_seed: 1234
  overwrite_output: false

  use_same_features: true
  automatic_feature_selection: false
  create_svm: true
  create_regional_models: true
```

Then validate:
```bash
python run_notebooks.py --config config/notebook_parameters_default.yml --setup
```

## Common Issues

- **`KeyError: 'defaults'`** when running the pipeline: The `defaults:` section is missing from the file. Every config must end with this block copied verbatim from `config/notebook_parameters_default.yml`.

- **`FileNotFoundError` for deposits CSV**: The value of `deposits_filename` does not match a file in `data_source/deposits/`. Check exact spelling — filenames are case-sensitive. Run `ls data_source/deposits/` to verify available files.

- **`--setup` prints `mantle data_dir not found`**: `feature_sets.mantle.enabled: true` but `data_dir` names a directory that does not exist under `data_source/mantle/` (or equivalent). Either disable mantle features or correct `data_dir` to a valid subdirectory.

- **`plate_model_name` not recognised by PlateModelManager**: The name must exactly match a PMM-registered model string (e.g. `zahirovic2022`, `clennett2020`). Run `python -c "from gplately import PlateModelManager; print(PlateModelManager().get_available_models())"` to list valid names.

- **Outputs written to `output/default/` instead of expected directory**: `run_name` is missing from `all_notebooks:` and fell back to the `defaults:` value. Ensure `run_name` is set explicitly under `all_notebooks:`.
