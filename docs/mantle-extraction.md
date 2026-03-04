# Mantle Feature Extraction

## Overview

G-ADOPT geodynamic model outputs (Zahirovic2022 reconstruction) provide 3D volumetric fields
(lon × lat × depth) at each timestep. This pipeline extracts mantle properties at the
reconstructed paleolocations of deposits and unlabelled points at the time of their formation,
adding these as additional columns in the training dataset.

The extraction is the primary original scientific contribution of this project.

## G-ADOPT Output Format

Location: `data/input_data/zahirovic2022/mantle_outputs/{gadopt_run_name}/`

- **Format**: netCDF, post-processed by `mantle-processing/untar-interp.py`
- **Dimensions**: `lon(360) × lat(181) × depth(146)` — depth in normalised units
- **Depth convention**: `depth_km = (2.208 − normalised_depth) × 2900`
  - Surface: `normalised_depth ≈ 1.208` (depth_km ≈ 0)
  - CMB: `normalised_depth ≈ 2.208` (depth_km ≈ 2900)
- **Variables present**:
  - `Temperature_Deviation_CG` — temperature deviation from the adiabat (main signal)
  - `Radial_Velocity` — radial (vertical) velocity
- **Timestep cadence**: one file per Ma, named `zahirovic2022_gadopt_{NNN}Ma.nc`

## Proof-of-Concept Location

`mantle-processing/point-sampling.ipynb`

This notebook demonstrates:
- `to_km(depth)`: normalised depth → km
- `reconstruct_points(model, lons, lats, times, plate_ids)`: wraps `gplately.Points.reconstruct()`
  over a time array
- `sample_mantle(ds, var, lons, lats, times, depths)`: samples mantle variable at (lon, lat, time)
  tuples + multiple depth levels using `xr.DataArray.interp()`; returns shape `(n_points, n_depths)`

**The notebook applies this only to a single hardcoded point (Canberra).** It must be generalised
into `lib/extract_mantle_features.py` before it is useful.

## Bugs in Existing Mantle-Processing Notebooks

Fix these before using the notebooks for scientific exploration:

| Notebook | Bug | Fix |
|---|---|---|
| `depth-slicing.ipynb` | `maxd` referenced in `ax.set_title()` but never defined → `NameError` | Define `maxd = ds.depth.max().item()` before the plotting loop |
| `time-resampling.ipynb` | Cell 2 calls `print(*[f"{p.stem}\n" for p in data_filenames])` but `data_filenames` not defined until Cell 3 | Move the `data_filenames` assignment to Cell 2, or swap cell order |
| `clustering.ipynb` | Despite the name, no clustering algorithm is implemented — only histograms | Not a bug per se; document as exploratory only |

## Feature Specification (PENDING — Phase 3 decision)

The specific mantle features to extract have not yet been decided. This must be resolved in
Phase 3 through scientific exploration before writing any extraction code.

Questions to answer:
1. **Which variables?** `Temperature_Deviation_CG`, `Radial_Velocity`, or derived products
   (e.g. `|Radial_Velocity| × Temperature_Deviation_CG` as a slab/plume discriminator,
   explored in `clustering.ipynb`)
2. **Which depths?** Fixed depth slices (e.g. 100 km, 200 km, 410 km), summary statistics
   across a depth range, or both
3. **Are features extraction at deposit formation age only, or across a time window?**
   Alfonso's `07-Input-to-PU-learn.ipynb` uses a 10 Ma temporal buffer for seafloor anomaly
   features — consider whether a similar buffer is geologically appropriate for mantle features

Record the decision as a comment block at the top of `00d-extract_mantle_features.ipynb`
before writing any extraction code.

## Library Function to Write — `lib/extract_mantle_features.py`

```python
def extract_mantle_features(
    points: pd.DataFrame,          # columns: lon, lat, age (Ma)
    mantle_data_dir: str,           # path to {gadopt_run_name}/ netCDF directory
    variables: list[str],           # e.g. ["Temperature_Deviation_CG", "Radial_Velocity"]
    depths_km: list[float],         # depth levels OR depth range for summary stats
    plate_reconstruction,           # gplately.PlateReconstruction object
    n_jobs: int = 4,
) -> pd.DataFrame:
    """
    For each point, reconstruct its paleoposition at age (Ma),
    sample the specified mantle variables at specified depths,
    and return the input DataFrame with appended mantle feature columns.

    Column naming: f"{variable}_at_{depth_km:.0f}km" for point values,
                   f"{variable}_{stat}_{min_km}_{max_km}km" for summary statistics.
    """
```

The function signature mirrors `lib/coregister_combined_point_data.py`:
- Input DataFrame, not a list of coordinates
- Returns DataFrame with appended columns
- Parallelised across timesteps using `joblib`

## Notebook — `00d-extract_mantle_features.ipynb`

**Not yet written.** Planned role:

1. Load `training_data_global.csv` (output of `00b`)
2. Load `grid_data.csv` (output of `00c`)
3. Call `extract_mantle_features()` on both
4. Write `training_data_global_with_mantle.csv` and `grid_data_with_mantle.csv`
5. Reads `gadopt_run_name` and `mantle_data_dir` from config

`mantle_data_dir` is constructed by `main.py` as:
`data/input_data/zahirovic2022/mantle_outputs/{gadopt_run_name}/`

This means `v1_pre_craton_fix` and `v2_post_craton_fix` outputs are automatically separated
by config without ad-hoc file naming.

## G-ADOPT Run Variants

| `gadopt_run_name` | Description | Status |
|---|---|---|
| `v1_pre_craton_fix` | Zahirovic2022, current G-ADOPT run (cratons as cold dripping lithosphere) | Available |
| `v2_post_craton_fix` | Zahirovic2022, corrected craton representation | Pending (in development) |

Comparing classifier outputs between v1 and v2 is a stated thesis objective — the config-keyed
file naming makes this comparison structurally clean.

## Integration with `01-create_classifiers.ipynb`

Add a `use_mantle_features` boolean to the config. In `01`:

```python
if params.get("use_mantle_features", False):
    training_file = "training_data_global_with_mantle.csv"
else:
    training_file = "training_data_global.csv"
```

No other changes to `01` are needed. The `CORRELATED_COLUMNS` set in `lib/pu.py` will not
include mantle features by default — they will pass through feature selection unchanged until
empirically evaluated. The Spearman clustering in `lib/feature_selection.py` (`01` with
`automatic_feature_selection: True`) will handle downstream correlation pruning.
