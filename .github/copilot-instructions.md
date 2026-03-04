# Copilot Instructions — 4D PUB Classifier Project

## Project Goal

This project extends Alfonso et al.'s (2024) spatiotemporal Positive-Unlabelled Bagging (PUB)
classifier for porphyry copper deposit prospectivity by adding mantle dynamic features extracted
from G-ADOPT geodynamic model outputs (Zahirovic2022 reconstruction). The primary scientific
question is whether mantle properties hold predictive power for mineral deposit formation, and
whether improvements to the G-ADOPT model (e.g. craton representation) measurably affect
prospectivity estimates.

The pipeline must be **reconstruction-agnostic** — all paths are driven by config so runs under
different plate models (Zahirovic2022, Cao2024, Clennett2020 baseline) are isolated and
reproducible.

---

## Workspace Layout

All repositories live under:
`/Users/glados/Documents/Not Useless/Documents/University/2026/Honours/Data & Code/`

| Repository | Role |
|---|---|
| `PUB-framework-Alfonso/` | **Primary working repo.** Pure Alfonso 2024 fork. All new code goes here. |
| `PUB-framework-Ehsan/` | **Reference only.** NSW 2D prospectivity pipeline. Consult for: BayesSearchCV integration, deposit size-class weighting, two-stage PUB→RF design, `lib_mpm.py` feature extractors. |
| `Mather2025-SeafloorAnomalies/` | **Reference only.** Ben Mather's modified Alfonso pipeline. Consult for: seafloor anomaly feature augmentation (`seafloor-anomalies/07-Input-to-PU-learn.ipynb`), temporal buffer logic, `stellar-data-mining/lib/` for cross-comparison. |
| `mantle-processing/` | **Proof-of-concept.** Contains G-ADOPT netCDF processing scripts and the `point-samping.ipynb` prototype for mantle feature extraction. This is the basis for the new `lib/extract_mantle_features.py`. Has known bugs — see `docs/mantle-extraction.md`. |
| `cu-deposits-preprocessing/` | **Deposit database.** Source of `GlobalUnifiedDeposits.csv` and the preprocessing pipeline that assigns `Plate_ID`, `Recon_Age_Ma`, and corrects plate-age inconsistencies. Deposit schema documented in `docs/data-layout.md`. |
| `GPlates data/` | Plate model files. Zahirovic2022 reconstruction files used by this project live here or under `data/input_data/zahirovic2022/`. |

---

## This Repo — Structure

### Notebooks (run in order)
| Notebook | Purpose |
|---|---|
| `00a-generate_data.ipynb` | Generate all input rasters from scratch (seafloor age, sediment, carbonate, crustal thickness, CO2, erosion). Requires submodules. Skip by downloading from Zenodo 14010839. |
| `00b-extract_training_data.ipynb` | Extract kinematic + raster features at deposit and unlabelled point locations → `training_data_global.csv` |
| `00c-extract_grid_data.ipynb` | Same extraction on a regular prediction grid → `grid_data.csv` |
| `00d-extract_mantle_features.ipynb` | **NEW (not yet written).** Appends G-ADOPT mantle features to training and grid data. |
| `01-create_classifiers.ipynb` | Train PUB classifier + SVM. Feature selection, cross-validation. |
| `02-create_probability_maps.ipynb` | Apply trained classifier to grid data → probability netCDF maps. |
| `03–08` | Animations, erosion, preservation, partial dependence, time series. |

### `lib/` — Module Summary
| Module | Purpose |
|---|---|
| `load_params.py` | `get_params(filename, notebook)` — loads and merges YAML config. The entry point for all config reads. |
| `plate_models.py` | `get_plate_reconstruction()`, `get_plot_topologies()` — reconstruction-agnostic model loading via local path or `plate-model-manager`. |
| `calculate_convergence.py` | Subduction zone kinematics via PlateTectonicTools. |
| `coregister_ocean_rasters.py` | Joins seafloor rasters to subduction trench points, masked to subducting plate. |
| `create_study_area_polygons.py` | Buffers reference features (currently subduction zones) to define the study domain per timestep. |
| `generate_unlabelled_points.py` | Sphere-uniform random points filtered to study area polygons. |
| `combine_point_data.py` | Merges deposit + unlabelled point data; reconstructs paleocoordinates. |
| `coregister_combined_point_data.py` | Haversine nearest-neighbour join of points to subduction zone rows. |
| `coregister_crustal_thickness.py` | Radius-search extraction of crustal thickness statistics per point. |
| `pu.py` | `BaggingPuClassifier` creation, `get_xy()`, `generate_grid_points()`, `create_probability_grids()`, constants (`CORRELATED_COLUMNS`, `PU_PARAMS`). |
| `cv.py` | `perform_cv()` — `RepeatedStratifiedKFold` cross-validation with per-region evaluation. |
| `feature_selection.py` | Spearman correlation clustering, `select_features()`. |
| `misc.py` | `reconstruct_by_topologies()`, `calculate_slab_flux()`, `filter_topological_features()`. |
| `water.py` | Subducted water budget (6 components). |
| `slab_dip.py` | Slab dip prediction via `Slab-Dip` package. |
| `assign_regions.py` | Spatial join to `regions.geojson`. **Has a known broken default path — see Known Bugs.** |
| `erodep/` | Cumulative erosion/deposition extraction. |
| `extract_data/` | Raster generation sub-package (used by `00a` only). |
| `check_files.py` | Downloads plate model and prepared data from Zenodo if missing. |
| `visualisation.py`, `animation.py`, `partial_dependence.py`, `feature_importance.py` | Plotting and output utilities. |

### Config System
All parameters are read from `notebook_parameters_default.yml` via `lib.load_params.get_params()`.
Per-run overrides live in `config/` and are merged by `main.py` before notebook execution.

**Current state (Phase 0 not yet complete):**
- `main.py` does not yet exist — edit `notebook_parameters_default.yml` directly for now
- Several planned config keys do not yet exist in the YAML: `reconstruction.name`,
  `reconstruction.plate_model_dir`, `raster_data_dir`, `reference_feature`, `buffer_km`,
  `use_mantle_features`, `gadopt_run_name` — these are added in Phase 0.3
- `plate_model_dir` is currently hardcoded as a literal in `00b`, `00c`, `01` — Phase 0.4 moves it to config

Existing keys: `output_dir`, `extracted_data_dir`, `deposits_filename`, `timespan.min/max`,
`n_jobs`, `random_seed`, `plate_model.use_provided_plate_model`, `plate_model.plate_model_name`.

---

## Navigation — Which Files to Consult

| Task | Files to read first |
|---|---|
| Config or path logic | `notebook_parameters_default.yml`, `lib/load_params.py`, `docs/data-layout.md` |
| Plate reconstruction, model loading | `lib/plate_models.py`, `lib/misc.py`, `docs/reconstruction.md` |
| Raster generation (`00a`) | `00a-generate_data.ipynb`, `lib/extract_data/`, `docs/reconstruction.md` |
| Training data extraction (`00b/c`) | `00b-extract_training_data.ipynb`, `lib/calculate_convergence.py`, `lib/coregister_ocean_rasters.py`, `lib/combine_point_data.py` |
| Mantle feature extraction (`00d`) | `mantle-processing/point-samping.ipynb`, `docs/mantle-extraction.md` |
| PU classifier, training | `lib/pu.py`, `01-create_classifiers.ipynb` |
| Cross-validation | `lib/cv.py`, `01-create_classifiers.ipynb` |
| Feature selection / correlation | `lib/feature_selection.py`, `01-create_classifiers.ipynb` |
| Study area polygons, unlabelled points | `lib/create_study_area_polygons.py`, `lib/generate_unlabelled_points.py` |
| Deposit database, schema | `cu-deposits-preprocessing/src/deposit_data_preprocessing/`, `docs/data-layout.md` |
| Bayesian hyperparameter tuning (future) | `PUB-framework-Ehsan/MPM_Porphyry_NSW.ipynb`, `PUB-framework-Ehsan/lib_mpm.py` |
| Mather seafloor anomaly features (reference) | `Mather2025-SeafloorAnomalies/seafloor-anomalies/07-Input-to-PU-learn.ipynb` |
| Probability map generation | `02-create_probability_maps.ipynb`, `lib/pu.py` (`create_probability_grids`) |

---

## Known Critical Bugs (fix before any other work)

1. **`lib/assign_regions.py`** — `DEFAULT_REGIONS_FILE = "../source_data/regions.shp"` references a
   non-existent path. Change to `Path(__file__).parent.parent / "regions.geojson"`.

2. **`submodules/` is uninitialised** — `CarbonateSedimentThickness/` and
   `predicting-sediment-thickness/` are empty. Run `git submodule update --init --recursive`
   before attempting `00a-generate_data.ipynb`.

3. **`00a-generate_data.ipynb`** — `overwrite` and `cleanup` are hardcoded to `False` in a cell
   immediately after reading them from params, silently overriding the YAML config.

4. **`run_notebooks.py`** — the `ALL_NOTEBOOKS` list contains stale filenames that do not match
   actual notebook names. Do not rely on it for batch execution; pass filenames explicitly.

---

## Docs Reference

| Condition | File |
|---|---|
| Before planning or implementing any pipeline stage, or reviewing task status | [`docs/pipeline.md`](docs/pipeline.md) |
| Before reading or writing any file path, working on config/path logic, or understanding the data hierarchy | [`docs/data-layout.md`](docs/data-layout.md) |
| Before working on plate reconstruction, raster generation, model file setup, or adding a new reconstruction | [`docs/reconstruction.md`](docs/reconstruction.md) |
| Before working on mantle feature extraction, G-ADOPT outputs, or `00d-extract_mantle_features.ipynb` | [`docs/mantle-extraction.md`](docs/mantle-extraction.md) |

## Ending Sessions
At the end of any session involving a significant decision, architectural change, or resolved ambiguity, propose updates to the relevant docs/ file and to this file if needed. Do not update without confirmation.