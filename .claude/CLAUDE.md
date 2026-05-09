# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Honours thesis research codebase for predicting copper mineral prospectivity in the American Cordillera using Positive-Unlabelled (PU) machine learning applied to plate tectonic reconstructions. The pipeline extracts features from plate models and mantle simulation outputs, trains PU classifiers, and produces time-dependent prospectivity maps.

## Environment setup

```bash
conda env create --file environment.yml   # create environment (named 'prospectivity')
conda activate prospectivity
```

Python 3.13. Key dependencies: `gplately`, `pygplates`, `pulearn`, `xarray`, `scikit-learn`, `papermill`, `geopandas`, `cartopy`.

## Running the pipeline

Notebooks are run via `run_notebooks.py` using `papermill`. Each run requires a config YAML file:

```bash
# Run specific notebooks
python run_notebooks.py --config config/notebook_parameters_default.yml --notebooks 00b 00c 01

# Overwrite notebook outputs in-place (saves to the .ipynb itself)
python run_notebooks.py --config config/notebook_parameters_default.yml --notebooks 01 -o

# List available notebooks
python run_notebooks.py --list-defaults

# Set up directory structure / validate config without running notebooks
python run_notebooks.py --config config/notebook_parameters_default.yml --setup

# Pre-cache remote plate model data (for HPC use without internet)
python run_notebooks.py --config config/myconfig.yml --cache-remote
```

Notebooks can also be run interactively in Jupyter. The `.ipynb` files and `nb_scripts/*.py` are kept in sync automatically via jupytext (format: `percent` py scripts).

## Notebook pipeline sequence

| Notebook | Purpose |
|----------|---------|
| `00a` | Generate/download plate model data |
| `00b` | Extract training data (deposit points) |
| `00c` | Extract grid data (regular grid for prospectivity maps) |
| `01` | Train PU classifiers (PU + SVM, regional models) |
| `02` | Create prospectivity probability maps |
| `03` | Create probability animations |
| `04` | Erosion/preservation distribution analysis |
| `05` | Create preservation maps |
| `06` | Create preservation animations |
| `07` | Partial dependence plots |
| `08` | Time series analysis |

The Zenodo download path ([zenodo.org/record/14010839](https://zenodo.org/record/14010839)) and `data_prepared/` fallback are holdovers from the upstream Alfonso et al. paper; this workflow always uses `use_extracted_data: true` and runs `00a`–`00c` to generate its own data.

## Configuration system

Config YAML files live in `config/`. The active run config is written to `config/.run_config.yml` (gitignored) before each run, and a snapshot is saved to `output/{run_name}/config_snapshot.yml`.

Config merging order (later overrides earlier):
1. `defaults` section in the YAML
2. `all_notebooks` section
3. `notebook_XX` section for the specific notebook

Key config parameters:
- `run_name`: output subdirectory under `output/`
- `plate_model`: choose `use_provided_plate_model: true` (uses `alfonso2024_default`) or specify `plate_model_name` (e.g. `zahirovic2022`)
- `timespan`: `min`/`max` in Ma
- `feature_sets`: enable/disable `subduction`, `crustal`, `mantle`, `erodep` feature groups
- `use_extracted_data`: `true` = use data from `data_extracted/`, `false` = use pre-prepared data from `data_prepared/`
- `deposits_filename`: CSV from `data_source/deposits/`

## Architecture

### `lib/` — core library

- **`paths.py` (`PathConfigManager`)**: Single source of truth for all file paths. Constructed once per run from a config YAML; exposes `OUTPUT_DIR`, `TRAINING_DATA_PATH`, `GRID_DATA_PATH`, `MANTLE_DATA_DIR`, etc. as attributes. Passed between notebooks to maintain consistent paths.

- **`load_params.py` (`get_params`)**: Merges the layered config YAML into a flat dict for a given notebook. Called internally by `PathConfigManager`.

- **`grid_features.py` (`GridFeatureRegistry`)**: Decorator-based registry for geospatial feature samplers. Features are registered with `@features.register(name)` or `@features.register_batch(...)`. Each feature has an optional `coordinate_resolver` (e.g. `snap_to_mantle`, `snap_to_plate_model`) that controls how point coordinates are transformed before sampling. Results are cached in a DataFrame to avoid redundant computation. The module-level singleton `features` is imported and used by notebooks.

- **`mantle_variables.py` (`MantleVariableRegistry`)**: Registry for mantle dataset variables, including base variables loaded directly from netCDF and derived variables computed from them (e.g. `LAB_Depth`, `Sublithospheric_Cold_Anomaly_Thickness`, depth-averaged temperature deviations with rolling means). The module-level singleton `variables` is used by `grid_features.py` to fetch `xr.DataArray` slices from the mantle dataset.

- **`pu.py`**: PU classifier creation (`create_classifier`), training data preparation (`get_xy`), grid point generation (`generate_grid_points`), probability calculation (`calculate_probabilities`), and raster grid output (`create_grids`). Uses `BaggingPuClassifier` from `pulearn` wrapping sklearn base estimators.

- **`plate_models.py`**: Plate model loading utilities using `plate-model-manager`.

- **`erodep/`**: Submodule for erosion/deposition data extraction and ML.

- **`extract_data/`**: Utilities for extracting paleotopography, paleobathymetry, crustal thickness, and LIP data.

### Data directories

- `data_source/`: Raw inputs — deposit CSVs, plate model files, mantle simulation outputs (netCDF), regions GeoJSON.
- `data_extracted/`: Data extracted from plate models by notebooks `00a`–`00c`; organised by plate model name.
- `data_prepared/`: Pre-prepared fallback data from Zenodo.
- `output/{run_name}/`: All outputs (classifier joblib files, probability grids, figures, config snapshot).

### Submodules

- `submodules/CarbonateSedimentThickness`: Carbonate sediment thickness computation.
- `submodules/predicting-sediment-thickness`: Ocean sediment thickness prediction.

### Notebook ↔ script sync

`jupytext.toml` keeps `*.ipynb` and `nb_scripts/*.py` (percent format) in sync. Always edit the `nb_scripts/*.py` files — never the `.ipynb` files directly, as their JSON structure is complex and not suited for programmatic editing. After any edit, sync back to the notebook:

```bash
jupytext --sync nb_scripts/<notebook>.py
```

## Commit conventions

Use conventional commits: `<type>(<scope>): <description>`

Types: `feat`, `fix`, `refactor`, `docs`, `chore`, `test`
Scope (optional): `lib`, `config`, `notebooks`, `data`

Examples:
- `feat(lib): add LAB-relative velocity feature to grid registry`
- `fix(config): correct mantle data path for zahirovic2022`
- `refactor(lib): simplify coordinate resolver caching logic`
- `docs: update CLAUDE.md with commit conventions`

Use `/commit` to generate a compliant commit message from staged changes.
Use `/commit-push-pr` to create a branch, commit, push, and open a PR.

## Adding new grid features

Register a feature on the module-level `features` registry in `lib/grid_features.py`:

```python
@features.register("my_feature_name", coords=snap_to_mantle)
def _my_feature(lons, lats, times):
    # return pd.Series or pd.DataFrame
    ...
```

For features that return multiple columns, use `@features.register_batch(declares=[...])`. The `coords` argument selects the coordinate resolver; omit it to use `reconstructed` (default). Results are cached automatically.
