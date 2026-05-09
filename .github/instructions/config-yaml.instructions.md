---
applyTo: "config/*.yml"
---

# Config YAML Conventions

## Merge Hierarchy (lib/load_params.py)
`defaults.all_notebooks` → `all_notebooks` → `defaults.notebook_XX` → `notebook_XX`

Null value (`~`) in per-notebook section inherits from `all_notebooks`.

## Required Keys
- `run_name` — output: `outputs/{run_name}/`
- `plate_model.plate_model_name` — e.g. `zahirovic2022`, `alfonso2024_default`
- `plate_model.use_provided_plate_model` — `true` = bundled Alfonso2024
- `timespan.min` / `timespan.max` — integer Ma
- `deposits_filename` — filename in `data_source/deposits/`
- `regions_filename` — filename in `data_source/regions/`
- `reference_feature: trenches`
- `study_zone_buffer: 6.0`
- `n_jobs`, `random_seed`, `overwrite_output`

## Feature Sets Block
```yaml
feature_sets:
  subduction: {enabled: true}
  crustal:    {enabled: false}
  mantle:     {enabled: true, data_dir: zahirovic2022}
  erodep:     {enabled: false}
```
