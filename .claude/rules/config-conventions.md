---
paths:
  - config/**
---

# Config YAML Conventions

## Structure
- Three sections: `defaults` (base), `all_notebooks` (run-level overrides), `notebook_XX` (per-notebook overrides)
- Merge order: `defaults` → `all_notebooks` → `notebook_XX` (later wins)
- `null` values in `notebook_XX` sections mean "inherit from above" — do not set to `false` unless intentional

## Key parameters
- `run_name`: maps to `output/{run_name}/`; use descriptive names like `mantle_test`, `thesis_final`
- `plate_model.use_provided_plate_model: true` → uses `alfonso2024_default`
- `plate_model.plate_model_name`: e.g. `zahirovic2022`, `clennett2020`
- `feature_sets.mantle.data_dir`: must match a subdirectory in the mantle data path
- `feature_sets` supports nesting (e.g. `feature_sets.subduction.carbonates.enabled`); a child set is active only if its parent is also enabled
- `deposits_filename`: must be a CSV present in `data_source/deposits/`
- `regions_filename`: must be a GeoJSON present in `data_source/regions/`
- `overwrite_output: false` is the safe default — set `true` only when re-running intentionally

## Active config
- `config/.run_config.yml` is written before each run (gitignored)
- Do not commit `config/.run_config.yml`
