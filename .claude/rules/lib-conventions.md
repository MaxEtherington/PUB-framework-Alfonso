---
paths:
  - lib/**/*.py
---

# lib/ Conventions

## Path management
- Never hardcode paths — use `PathConfigManager` from `lib/paths.py`
- Access paths via attributes: `p.OUTPUT_DIR`, `p.TRAINING_DATA_PATH`, `p.GRID_DATA_PATH`, `p.MANTLE_DATA_DIR`, `p.PLATE_MODEL_DIR`
- Construct once per run: `p = PathConfigManager(config_path, notebook='01a')`
- `p.use_features('mantle')` or `p.use_features('subduction.carbonate')` checks enabled state; nested dot-notation keys reflect nested `feature_sets` config structure

## Parallelism
- Use `joblib.Parallel` + `joblib.delayed` for multi-process work; `n_jobs` comes from config
- Thread-safe random generation: use `np.random.SeedSequence` + `seq.spawn(n)` for parallel RNGs

## Registry pattern (`grid_features.py`, `mantle_variables.py`)
- Register features on module-level singleton `features` (not inside functions)
- Use `@features.register(name, coords=snap_to_mantle)` for single-column features
- Use `@features.register_batch(declares=[...])` for multi-column features
- Coordinate resolvers: `snap_to_mantle`, `snap_to_plate_model`, or omit for `reconstructed`
- Do not manually cache — the registry caches in a `DataFrame` automatically
- `point_data` set on registry must contain columns: `lon`, `lat`, `age (Ma)`, `present_lon`, `present_lat`

## Data loading
- Use `lib.misc.load_data(data)` which accepts `str | DataFrame` — prefer this over bare `pd.read_csv`
- Filter by `age (Ma)` not by index; time columns use float (`np.float64`)

## Warnings
- Suppress `UserWarning` from `gplately`/`pygplates` imports with `warnings.catch_warnings()` + `simplefilter("ignore")`
