---
name: add-grid-feature
description: Registers a new geospatial feature sampler in `lib/grid_features.py` using the `@features.register` or `@features.register_batch` decorator pattern. Use when user says 'add a feature', 'register a new feature', 'add to grid registry', or modifies `lib/grid_features.py`. Handles coordinate resolver selection (`snap_to_mantle`, `snap_to_plate_model`, `reconstructed`), return type (Series vs DataFrame), and batch declarations. Do NOT use for modifying existing registered features or for adding mantle variables to `lib/mantle_variables.py`.
paths:
  - lib/grid_features.py
---
# Add Grid Feature

## Critical

- **Never edit notebook files directly.** Always edit `nb_scripts/*.py` — the `PostToolUse` hook in `.claude/settings.json` automatically runs `jupytext --sync` after each edit.
- All feature sampler functions **must** accept exactly `(lons, lats, times)` as positional arrays (`np.ndarray`) and return `pd.Series` or `pd.DataFrame`.
- Features registered with `@features.register` must return a **`pd.Series`** (single column). Features returning multiple columns **must** use `@features.register_batch`.
- The module-level singleton `features` (imported from `lib/grid_features.py`) is the only registry — never instantiate a new `GridFeatureRegistry`.
- Do not cache results manually; the registry's `get()` method handles caching automatically.
- Run `ruff check lib/ --fix` after editing `lib/grid_features.py`.

## Instructions

### Step 1 — Choose the coordinate resolver

Select based on what the feature samples:

| Resolver | Coordinates returned | Use when |
|---|---|---|
| `snap_to_mantle` | Reconstructed lon/lat at birth time, snapped to ~10 Myr mantle timesteps | Sampling `features.mantle_dataset` via `variables.get(...)` |
| `snap_to_plate_model` | Present-day lon/lat, birth times snapped to 1 Myr plate reconstruction steps | Sampling plate kinematics via `gpl.Points` or `features.plate_reconstruction` |
| `reconstructed` | Reconstructed lon/lat at exact birth time (default if `coords` omitted) | Using pre-reconstructed coordinates from `point_data["lon", "lat"]` |
| `present_day` | Present-day lon/lat, exact birth times | Sampling present-day fields |

Verify your resolver is already imported at the top of `lib/grid_features.py` — all four above are defined in that file.

### Step 2 — Register a single-output feature (`pd.Series`)

Add after the existing feature definitions block (after line ~619 in `lib/grid_features.py`):

```python
@features.register("my_feature_name", coords=snap_to_mantle)
def _my_feature(
    lons: np.ndarray,
    lats: np.ndarray,
    times: np.ndarray,
) -> pd.Series:
    da = variables.get("Temperature_Deviation_CG", features.mantle_dataset)
    result = sample_mantle_var(da=da, lons=lons, lats=lats, times=times)
    return result.iloc[:, 0].rename("my_feature_name")
```

Verify: calling `features.available` in a notebook cell lists `"my_feature_name"`.

### Step 3 — Register a multi-output feature (`pd.DataFrame`)

Use `@features.register_batch` with explicit `declares` when probing at `(0.0, 0.0, 0.0)` would fail (e.g., needs a live dataset):

```python
@features.register_batch(
    declares=["feature_col_a", "feature_col_b"],
    coords=snap_to_plate_model,
    probe=False,  # Required when sampler accesses features.plate_reconstruction or features.mantle_dataset
)
def _my_batch_feature(
    lons: np.ndarray,
    lats: np.ndarray,
    times: np.ndarray,
) -> pd.DataFrame:
    ...
    return pd.DataFrame({"feature_col_a": col_a, "feature_col_b": col_b})
```

If the function **can** be probed with dummy data (no registry access), omit `probe=False` and `declares` — the registry will infer column names from the probe call.

Verify: column names in `declares` exactly match the `pd.DataFrame` column names returned by the function.

### Step 4 — Register parameterised variants in a loop

For features registered at multiple parameter values (e.g., depths), use `functools.partial`:

```python
from functools import partial

def _my_parameterised_feature(
    lons: np.ndarray,
    lats: np.ndarray,
    times: np.ndarray,
    offset_km: float = 0,
) -> pd.DataFrame:
    ...
    return pd.DataFrame({f"my_feature_{offset_km}km": values})

for offset in [0, 40, 80, 120]:
    features.register_batch(
        declares=[f"my_feature_{offset}km"],
        coords=snap_to_mantle,
    )(partial(_my_parameterised_feature, offset_km=offset))
```

Verify: each loop iteration registers a distinct name — check `features.available` includes all variants.

### Step 5 — Access other registered features inside a sampler

Call `features.get("other_feature_name")` to access a cached result from another feature. This triggers sampling of that feature if not yet cached:

```python
def _my_derived_feature(lons, lats, times):
    v_east = features.get("east_plate_velocity (cm/yr)").to_numpy()
    v_north = features.get("north_plate_velocity (cm/yr)").to_numpy()
    speed = np.linalg.norm(np.column_stack([v_east, v_north]), axis=1)
    return pd.Series(speed, name="derived_speed")
```

Verify: the feature you depend on is registered *before* yours in the file (or is a `register_batch` feature that will be computed on demand).

### Step 6 — Lint

```bash
ruff check lib/ --fix
```

The `PostToolUse` hook handles jupytext sync automatically — no manual sync needed after editing `lib/grid_features.py`.

## Examples

**User says:** "Add a feature for plate speed (magnitude of east + north velocity) using the plate model resolver."

**Actions taken:**
1. Determine resolver: uses `snap_to_plate_model` (plate kinematics).
2. Returns a single column → use `@features.register`.
3. Depends on `east_plate_velocity (cm/yr)` and `north_plate_velocity (cm/yr)` → call `features.get()`.
4. Added to `lib/grid_features.py` after the existing `_plate_acceleration` function:

```python
@features.register("plate_speed (cm/yr)", coords=snap_to_plate_model)
def _plate_speed(
    present_lons: np.ndarray,
    present_lats: np.ndarray,
    times: np.ndarray,
) -> pd.Series:
    v_east = features.get("east_plate_velocity (cm/yr)").to_numpy()
    v_north = features.get("north_plate_velocity (cm/yr)").to_numpy()
    speed = np.linalg.norm(np.column_stack([v_east, v_north]), axis=1)
    return pd.Series(speed, name="plate_speed (cm/yr)")
```

**Result:** `"plate_speed (cm/yr)"` appears in `features.available` and is cached on first call to `features.get("plate_speed (cm/yr)")`.

---

**User says:** "Add LAB temperature sampled at 0, 50, 100 km below the LAB."

**Actions taken:**
1. Resolver: `snap_to_mantle` (samples mantle dataset).
2. Returns multiple columns → `@features.register_batch` with `probe=False` (needs live dataset).
3. Use `sample_LAB_depths` from `lib.mantle_variables`.

```python
def _lab_temperature(
    lons: np.ndarray,
    lats: np.ndarray,
    times: np.ndarray,
    offset_km: float = 0,
) -> pd.DataFrame:
    da = variables.get("FullTemperature_CG", features.mantle_dataset)
    result = sample_LAB_depths(
        ds=features.mantle_dataset,
        da=da,
        lons=lons, lats=lats, times=times,
        offset_km=offset_km,
    )
    result.columns = [f"LAB_Temperature_{offset_km}km"]
    return result

for offset in [0, 50, 100]:
    features.register_batch(
        declares=[f"LAB_Temperature_{offset}km"],
        coords=snap_to_mantle,
        probe=False,
    )(partial(_lab_temperature, offset_km=offset))
```

## Common Issues

**`RuntimeError: point_data has not been set on the feature registry`**
The sampler called `features.point_data` before `features.extract(...)` was called in the notebook. Features are only callable after `features.extract(point_data, mantle_data_dir, plate_reconstruction)` is invoked.

**`KeyError: Unknown feature: 'my_feature_name'. Available: [...]`**
The feature was not registered. Check: (a) the `@features.register(...)` decorator is applied, (b) the file was saved and re-imported (kernel restart may be needed).

**`TypeError: Batch producer 'fn' returned Series; expected pandas.DataFrame`**
`@features.register_batch` requires the function to return a `pd.DataFrame`. Wrap the Series: `return series.to_frame()`.

**`declares` names don't match actual DataFrame columns**
The registry maps each name in `declares` to the same sampler. If `get("feature_col_a")` is called but the DataFrame has column `"Feature_Col_A"`, you'll get a `KeyError`. Ensure `declares` strings exactly match the DataFrame column names.

**`RuntimeError: mantle_data_dir has not been set`**
Accessing `features.mantle_dataset` inside a sampler before the registry is initialised. Either (a) set `probe=False` so the function isn't probed at import time, or (b) ensure the feature is only called after `features.extract(...)` in the notebook.

**Ruff error `F811: redefinition of unused name`**
Using `def _` repeatedly in a loop (as seen in `mantle_variables.py`). Name each function uniquely or use `partial` with a named outer function instead.
