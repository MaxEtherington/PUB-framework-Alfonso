---
paths:
  - lib/**/*.py
  - lib/**/*.ipynb
---

# lib/ Conventions

## Parallelism
- Always `joblib.Parallel(n_jobs, verbose=int(verbose))` + `delayed()` — never `multiprocessing`
- Split work with `np.array_split(times, nprocs)` before parallel dispatch
- Guard single-job path: `if n_jobs <= 1: result = [func(...)]` before `Parallel` block

## Verbose / Logging
- Every pipeline function takes `verbose: bool` param
- Print to `sys.stderr`: `print("...", file=stderr)`
- Guard with `if verbose:`

## PathConfigManager
- Import from `lib/paths.py`: `from .paths import PathConfigManager`
- All output dirs created via `p.create_directories()`; never `os.makedirs` ad-hoc
- Resolve paths: `Path(data_dir).resolve()`

## Data Loading Pattern
- Use `lib/misc.py:load_data()` — accepts `str | os.PathLike | pd.DataFrame`
- Always pass `copy=False` when you own the data

## Spatial Joins (haversine NN)
- `sklearn.neighbors.NearestNeighbors(metric="haversine")` — inputs in radians
- Convert: `coords = np.deg2rad(np.hstack([lats.reshape(-1,1), lons.reshape(-1,1)]))`
- Lat first, lon second for haversine

## Plate Reconstruction
- `lib/plate_models.py:get_plate_reconstruction(model_name, model_dir)` — handles PMM + local fallback
- PMM fallback uses `.metadata.json` in `model_dir/model_name/`
- `filter_topologies=True` writes a temp file; hold the returned `_TemporaryFileWrapper`

## Label Values
- Positive: `"positive"` · Negative: `"negative"` · Unlabelled: `"unlabelled"` (also accept `"unlabeled"`)
- Column always named `"label"`
- Coordinate columns: `"lon"`, `"lat"`, `"age (Ma)"` — exact strings
