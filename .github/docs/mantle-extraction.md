# Mantle Feature Extraction

## Overview

G-ADOPT geodynamic model outputs provide 3D volumetric fields (lon × lat × depth) at each timestep. This pipeline extracts mantle properties at the reconstructed paleolocations of deposits and unlabelled points at formation time, appending them as feature columns to the training dataset. This is the primary original scientific contribution of the project.

## G-ADOPT Output Format

Location: `data/{plate_model_name}/mantle_outputs/{gadopt_run_name}/`

- **Format**: netCDF, post-processed by `mantle-processing/untar-interp.py`
- **Dimensions**: `lon(360) × lat(181) × depth(146)` — depth in normalised units
- **Depth convention**: `depth_km = (2.208 − normalised_depth) × 2900`
- **Variables**: `Temperature_Deviation_CG` (deviation from adiabat), `Radial_Velocity` (vertical)
- **Cadence**: one file per Ma — `{plate_model_name}_gadopt_{NNN}Ma.nc`

## G-ADOPT Run Variants

| `gadopt_run_name` | Description | Status |
|---|---|---|
| `v1_pre_craton_fix` | Zahirovic2022; cratons as cold dripping lithosphere | Available |
| `v2_post_craton_fix` | Zahirovic2022; corrected craton representation | Pending |

Comparing classifier outputs between v1 and v2 is a stated thesis objective. The config-keyed naming keeps runs cleanly isolated.

## Proof-of-Concept

`mantle-processing/point-sampling.ipynb` demonstrates the core operations (single hardcoded point only — must be generalised):

- `to_km(depth)`: normalised depth → km
- `reconstruct_points(model, lons, lats, times, plate_ids)`: wraps `gplately.Points.reconstruct()` over a time array
- `sample_mantle(ds, var, lons, lats, times, depths)`: samples variable at (lon, lat, time) + depth levels via `xr.DataArray.interp()`; returns shape `(n_points, n_depths)`

Known bugs in `mantle-processing/` notebooks are tracked in [#21](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/21).

## Feature Specification (PENDING — Phase 3 decision)

Resolve through scientific exploration in Phase 3 before writing any extraction code. Record the final decision as a comment block at the top of `00d-extract_mantle_features.ipynb`.

Key questions:
1. **Which variables?** `Temperature_Deviation_CG`, `Radial_Velocity`, or derived products
2. **Which depths?** Fixed slices (e.g. 100 km, 410 km), summary statistics over a range, or both
3. **Time window?** Formation age only, or a temporal buffer (cf. Mather's 10 Ma buffer for seafloor anomaly features)

## Target Library Function — `lib/extract_mantle_features.py`

Mirrors the interface of existing `lib/` coregistration modules (input DataFrame in, DataFrame with appended columns out):

```python
def extract_mantle_features(
    points: pd.DataFrame,        # columns: lon, lat, age (Ma)
    mantle_data_dir: str,         # path to {gadopt_run_name}/ directory
    variables: list[str],
    depths_km: list[float],
    plate_reconstruction,         # gplately.PlateReconstruction
    n_jobs: int = 4,
) -> pd.DataFrame:
```

Column naming: `{variable}_at_{depth_km:.0f}km` for point values; `{variable}_{stat}_{min}_{max}km` for summary statistics. Time interpolation between available timesteps and graceful NaN for out-of-domain points are both required.

## Integration with `01-create_classifiers.ipynb`

Controlled by the `use_mantle_features` boolean config key. When `True`, `01` reads `training_data_global_with_mantle.csv` instead of `training_data_global.csv`. Spearman clustering in `lib/feature_selection.py` handles downstream correlation pruning — no other changes to `01` are needed.
