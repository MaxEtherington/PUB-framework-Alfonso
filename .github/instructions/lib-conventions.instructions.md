---
applyTo: "lib/**/*.py"
---

# lib/ Module Conventions

## Parallelism
- `joblib.Parallel(n_jobs, verbose=int(verbose))` + `delayed()` — never `multiprocessing`
- Pre-split: `np.array_split(times, nprocs)`
- Single-job guard before `Parallel` block

## Logging
- `verbose: bool` param on every pipeline function
- `print("...", file=stderr)` from `sys.stderr`; guarded by `if verbose:`

## Spatial Joins
- `NearestNeighbors(metric="haversine")` — inputs in **radians**, **lat first**
- Convert: `np.deg2rad(np.hstack([lats.reshape(-1,1), lons.reshape(-1,1)]))`

## Data Conventions
- Labels: `"positive"`, `"negative"`, `"unlabelled"` in `"label"` column
- Coords: `"lon"`, `"lat"`, `"age (Ma)"` — exact strings everywhere
- Use `lib/misc.py:load_data()` for all CSV/DataFrame inputs
- Use `lib/paths.py:PathConfigManager` for all path resolution

## Plate Reconstruction
- `lib/plate_models.py:get_plate_reconstruction(model_name, model_dir)` — PMM + local fallback
- Local fallback reads `.metadata.json` in `model_dir/model_name/`
