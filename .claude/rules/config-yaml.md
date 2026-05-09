---
paths:
  - config/*.yml
  - config/*.yaml
---

# Config YAML Structure

All configs follow the 3-level merge in `lib/load_params.py:get_params()`:

```
defaults.all_notebooks  →  all_notebooks  →  defaults.notebook_XX  →  notebook_XX
```

## Required Keys (all_notebooks / defaults)
- `run_name` — output directory name: `outputs/{run_name}/`
- `plate_model.plate_model_name` — PMM model name (e.g. `zahirovic2022`)
- `plate_model.use_provided_plate_model` — `true` uses bundled Alfonso2024
- `timespan.min` / `timespan.max` — integer Ma
- `deposits_filename` — CSV in `data_source/deposits/` (e.g. `"deposits-Etherington.csv"`)
- `regions_filename` — GeoJSON in `data_source/regions/` (e.g. `"regions.geojson"`)
- `reference_feature` — `trenches` (geometry for study area)
- `study_zone_buffer` — float degrees (default `6.0`)
- `n_jobs` — int parallelism
- `random_seed` — int for reproducibility
- `overwrite_output` — bool

## Feature Sets Block
```yaml
feature_sets:
  subduction:     # distance to trench, slab age, convergence
    enabled: true
  crustal:        # crustal thickness rasters
    enabled: false
  mantle:         # temperature, flow velocity
    enabled: true
    data_dir: zahirovic2022
  erodep:         # cumulative erosion/deposition
    enabled: false
```

## Per-Notebook Overrides
- Null value (`~` or empty) means inherit from `all_notebooks`
- Only set keys that differ from global config
- `notebook_00c.grid_resolution` — degrees (default `0.5`)
- `notebook_04.outlier_contamination` — float (default `0.06`)
