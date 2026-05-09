---
name: add-lib-module
description: Adds a new feature extraction or processing module to lib/ following the joblib/verbose/PathConfigManager patterns from lib/coregister_combined_point_data.py, lib/calculate_convergence.py, and lib/grid_features.py. Use when user says 'add new feature', 'extract X data', 'new coregistration', 'register grid feature', or adds files to lib/extract_data/. Ensures coordinate columns (lon, lat, age (Ma)) and label conventions are preserved. Do NOT use for changes to notebooks, config files, or the PathConfigManager itself.
paths:
  - lib/*.py
  - lib/extract_data/*.py
---
# Add lib/ Module

## Critical

- **Never rename or drop** `lon`, `lat`, `age (Ma)` columns — these are the universal join keys across the entire pipeline. Any new feature columns must be appended; existing columns must survive.
- **Never hardcode paths.** Paths come from `PathConfigManager` attributes (e.g. `paths.MANTLE_DATA_DIR`, `paths.POINTS_DATA_DIR`). Do not construct path strings manually.
- Wrap all `gplately` / `gplates` / `ptt` imports in `with warnings.catch_warnings(): warnings.simplefilter("ignore", UserWarning)` — these libraries emit noisy deprecation warnings.
- All public entry-point functions must accept `verbose: bool = False` and `n_jobs: int = 1`; log to `stderr` via `print(..., file=stderr)` only when `verbose=True`.
- When writing output, create the directory with `os.makedirs(output_dir, exist_ok=True)` before writing — never assume it exists.

---

## Instructions

### 1. Create the module file

Create `lib/<module_name>.py`. The filename must be lowercase with underscores and describe the operation, e.g. `extract_raster_features.py`, `coregister_grid_points.py`.

Required boilerplate at the top of every new module:

```python
"""One-line description of what this module computes."""
import os
import warnings
from sys import stderr
from typing import Optional

import numpy as np
import pandas as pd
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from gplately import PlateReconstruction, EARTH_RADIUS
from joblib import Parallel, delayed

from .misc import (
    _PathLike,
    _PathOrDataFrame,
)
```

Only import what is needed. Add `pygplates`, `geopandas`, `xarray`, or `sklearn` imports inside the same warning-suppression block if they emit warnings on import.

Verify the module is importable before proceeding: `python -c "from lib import <module_name>"`.

### 2. Write the public entry-point function

Every module exposes one `run_<operation>` function following this signature pattern (from `lib/coregister_combined_point_data.py:23` and `lib/calculate_convergence.py:30`):

```python
def run_<operation>(
    point_data: _PathOrDataFrame,          # str path or DataFrame
    <other_inputs>,
    output_filename: Optional[_PathLike] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """Short docstring.

    Parameters
    ----------
    point_data : str or DataFrame
    output_filename : str, optional
        If provided, write the result to a CSV file.
    n_jobs : int
        Number of processes to use.
    verbose : bool, default: False
        Print log to stderr.

    Returns
    -------
    DataFrame
    """
    # Load from path if given a string
    if isinstance(point_data, (str, os.PathLike)):
        if verbose:
            print(f"Loading point data from file: {point_data}", file=stderr)
        point_data = pd.read_csv(point_data)
    else:
        point_data = pd.DataFrame(point_data)

    # ... processing ...

    if output_filename is not None:
        output_dir = os.path.dirname(os.path.abspath(output_filename))
        if not os.path.isdir(output_dir):
            if verbose:
                print("Output directory does not exist; creating now: " + output_dir, file=stderr)
            os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print("Writing output to file: " + os.path.basename(output_filename), file=stderr)
        out.to_csv(output_filename, index=False)
    return out
```

Verify: call `run_<operation>` with a tiny test DataFrame containing `lon`, `lat`, `age (Ma)` columns and confirm it returns a DataFrame with those columns intact.

### 3. Write per-timestep parallel helpers

When processing is per-timestep (the common case), split into a `run_` wrapper that dispatches per time and a `_<operation>_timestep` private function:

```python
# In the run_ function body:
times = point_data["age (Ma)"].unique()

if n_jobs == 1:
    out = [_<operation>_timestep(point_data[point_data["age (Ma)"] == t], ...) for t in times]
else:
    with Parallel(n_jobs, verbose=10 if verbose else 0) as parallel:
        out = parallel(
            delayed(_<operation>_timestep)(point_data[point_data["age (Ma)"] == t], ...)
            for t in times
        )

out = pd.concat(out, ignore_index=True)
out = out.drop(columns="index", errors="ignore")
```

Note: `verbose=10 if verbose else 0` matches the pattern in `lib/calculate_convergence.py:95` — use `10`, not `True`.

Verify: run with `n_jobs=2` on mock data; confirm row count matches `n_jobs=1` output.

### 4. Register a grid feature (if extracting a new feature for the grid)

If the new feature should be available via the `GridFeatureRegistry` in `lib/grid_features.py`, add a registration at the bottom of that file (after line 340, before the `# === Feature definitions ===` section, or alongside existing `@features.register` calls).

For a **single-output feature**:

```python
@features.register("my_feature_name (units)", coords=reconstructed)  # or snap_to_mantle / snap_to_plate_model
def _my_feature(
    lons: np.ndarray,
    lats: np.ndarray,
    times: np.ndarray,
) -> pd.Series:
    ...
    return pd.Series(result, name="my_feature_name (units)")
```

For a **batch / multi-output feature**:

```python
@features.register_batch(declares=["col_a", "col_b"], coords=snap_to_plate_model)
def _my_batch_feature(
    lons: np.ndarray,
    lats: np.ndarray,
    times: np.ndarray,
) -> pd.DataFrame:
    ...
    return pd.DataFrame({"col_a": a_vals, "col_b": b_vals})
```

Available coordinate resolvers (defined in `lib/grid_features.py:392–435`):
- `reconstructed` — uses `lon`, `lat`, `age (Ma)` from point_data (reconstructed birth position, default)
- `present_day` — uses `present_lon`, `present_lat`, `age (Ma)`
- `snap_to_plate_model` — snaps birth times to 1 Myr plate model steps
- `snap_to_mantle` — snaps birth times to ~10 Myr mantle output steps, reconstructs position

Verify: `features.available` contains the new feature name after import.

### 5. Call the new module from a notebook

In the target notebook (e.g. `00b-extract_training_data.ipynb` or `00c-extract_grid_data.ipynb`), add an import cell:

```python
from lib.<module_name> import run_<operation>
```

Call it using paths from `PathConfigManager`:

```python
from lib.paths import PathConfigManager
from lib.load_params import get_params

paths = PathConfigManager(config_path=RUN_CONFIG_PATH, notebook="00b")

result = run_<operation>(
    point_data=paths.TRAINING_DATA_PATH,   # or pass a DataFrame directly
    output_filename=paths.POINTS_DATA_DIR / "<output_name>.csv",
    n_jobs=params["n_jobs"],
    verbose=True,
)
```

Verify: `"lon" in result.columns and "lat" in result.columns and "age (Ma)" in result.columns`.

---

## Examples

**User says:** "Add a module that coregisters grid points to raster data at each timestep."

**Actions taken:**
1. Create `lib/coregister_grid_to_raster.py` with the standard module header.
2. Implement `run_coregister_grid_to_raster(point_data, raster_dir, output_filename=None, n_jobs=1, verbose=False)`.
3. Per-timestep helper `_coregister_timestep(points_at_time, raster_path)` reads the raster, samples at each `(lon, lat)`, appends a new column without touching `lon`, `lat`, `age (Ma)`.
4. In `00c-extract_grid_data.ipynb`, add:
   ```python
   from lib.coregister_grid_to_raster import run_coregister_grid_to_raster
   grid_data = run_coregister_grid_to_raster(
       point_data=grid_df,
       raster_dir=paths.RASTER_DATA_DIR,
       output_filename=paths.GRID_DATA_PATH,
       n_jobs=params["n_jobs"],
       verbose=True,
   )
   ```

**Result:** `grid_data` has all original columns plus new raster feature columns; `lon`, `lat`, `age (Ma)` are intact; CSV written to `paths.GRID_DATA_PATH`.

---

## Common Issues

**`KeyError: 'lon'` or `KeyError: 'age (Ma)'` after concat**
The per-timestep function filtered or renamed a required column. Confirm `_timestep` returns a copy of the input slice with columns *appended*, not replaced. Use `points = points.copy()` at the start of every `_timestep` function.

**`RuntimeError: point_data has not been set on the feature registry`**
You called `features.get(name)` before setting `features.point_data`. Either call `features.extract(point_data, ...)` (which sets it automatically) or set `features.point_data = df` explicitly first. See `lib/grid_features.py:259`.

**`UserWarning: Overwriting existing point_data`**
`features.point_data` was already set and you're assigning a different DataFrame. Call `features.reset()` first, then set `features.point_data`.

**`parallel` hangs or produces fewer rows than expected with `n_jobs > 1`**
`joblib` with `loky` backend can deadlock when forking pygplates objects. Pass serialisable inputs only (filenames or plain numpy arrays) to the delayed function. Reconstruct `PlateReconstruction` inside the worker if needed — see `lib/coregister_combined_point_data.py:385`.

**`FileNotFoundError` on output write**
The output directory was not created before `to_csv`. The boilerplate in Step 2 includes `os.makedirs(output_dir, exist_ok=True)` — ensure it runs *before* `out.to_csv(...)`.

**`TypeError: Either topology_filenames and rotation_filenames or plate_reconstruction must be specified`**
The `run_` function received `None` for both the filenames and the `PlateReconstruction` object. When calling from a notebook always pass `plate_reconstruction=model` explicitly — see `lib/calculate_convergence.py:71`.