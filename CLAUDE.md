# CLAUDE.md

Guidance for Claude Code on this repository.

## Project overview

Honours thesis: copper mineral prospectivity in the American Cordillera using Positive-Unlabelled (PU) ML on plate tectonic reconstructions. Pipeline extracts features from plate models and mantle simulations, trains `BaggingPuClassifier` models, and produces time-dependent prospectivity maps.

## Environment

```bash
conda env create --file environment.yml   # env named 'prospectivity'
conda activate prospectivity
```

Python 3.13. Key deps: `gplately`, `pygplates`, `pulearn`, `xarray`, `scikit-learn`, `papermill`, `geopandas`, `cartopy`, `jupytext`, `ruff`.

## Running the pipeline

All notebooks run via `run_notebooks.py` using `papermill`:

```bash
# Run specific notebooks
python run_notebooks.py --config config/notebook_parameters_default.yml --notebooks 00b 00c 01a 01b

# Overwrite notebook in-place (saves to .ipynb itself)
python run_notebooks.py --config config/notebook_parameters_default.yml --notebooks 01b -o

# List available notebook codes
python run_notebooks.py --list-defaults

# Validate config + set up dirs without running
python run_notebooks.py --config config/notebook_parameters_default.yml --setup
```

## Figures and plotting

Read `config/thesis.mplstyle` before creating or modifying any plot. All style decisions, figure sizes, and per-context guidelines are documented there.

## Linting

```bash
ruff check lib/          # lint lib/ (excludes nb_scripts/, submodules/)
ruff check lib/ --fix    # auto-fix
```

Config in `ruff.toml`: Python 3.13, line-length 120. Ignores `E501` (line length), `E741` (ambiguous names), `F401` (unused imports). `nb_scripts/`, `submodules/`, `playground/` are excluded. The `.ruff_cache/` directory (containing `CACHEDIR.TAG`) is auto-generated and not committed to version control.

A `PostToolUse` hook automatically runs `ruff check` after any `Edit` or `Write` to `lib/*.py` files.

## Notebook pipeline sequence

| Notebook | Purpose |
|----------|---------|
| `00a` | Generate/download plate model data |
| `00b` | Extract training data (deposit points) |
| `00c` | Extract grid data (regular grid for maps) |
| `01a` | Select features for classifier |
| `01b` | Train and validate PU classifiers (PU + SVM, regional models) |
| `02` | Create prospectivity probability maps |
| `03` | Create probability animations |
| `04` | Erosion/preservation distribution analysis |
| `05` | Create preservation maps |
| `06` | Create preservation animations |
| `07` | Partial dependence plots |
| `08` | Time series analysis |

This workflow always uses `use_extracted_data: true` and runs `00a`–`00c` to generate data. The Zenodo fallback (`data_prepared/`) and `zenodo.org/record/14010839` are upstream holdovers.

## Notebook ↔ script sync

`jupytext.toml` keeps `*.ipynb` and `nb_scripts/*.py` (percent format) in sync.

**Always edit `nb_scripts/*.py` — never `.ipynb` files directly.**

Syncing is automatic: a `FileChanged` hook in `.claude/settings.json` runs `jupytext --sync` when `nb_scripts/*.py` files are saved. A `PreToolUse` guard in `.claude/settings.json` blocks direct `.ipynb` editing. No manual sync step is needed when Claude edits these files.

To sync manually (e.g. after editing outside Claude Code):
```bash
jupytext --sync nb_scripts/<notebook>.py
```

## Configuration system

Config YAMLs live in `config/`. Active config written to `config/.run_config.yml` (gitignored); snapshot saved to `output/{run_name}/config_snapshot.yml`.

**Merge order** (later overrides earlier):
1. `defaults` section
2. `all_notebooks` section
3. `notebook_XX` section

Key parameters:
- `run_name`: output subdirectory under `output/`
- `plate_model.use_provided_plate_model: true` → uses `alfonso2024_default`; or set `plate_model_name` (e.g. `zahirovic2022`)
- `timespan.min`/`max`: in Ma
- `feature_sets`: enable/disable `subduction`, `crustal`, `mantle`, `erodep`; supports nesting (e.g. `subduction.carbonate`)
- `use_extracted_data: true` → reads from `data_extracted/`; `false` → reads from `data_prepared/`
- `deposits_filename`: CSV from `data_source/deposits/` (e.g. `deposits-Etherington.csv`, `Porphyry-deposits.csv`, `IOCG-deposits.csv`, `VMS-deposits.csv`, `SedCu-deposits.csv`)
- `regions_filename`: GeoJSON from `data_source/regions/` (default: `regions.geojson`)
- `grid_resolution`: degrees (notebook_00c, default `0.5`)
- `n_jobs`: parallelism via `joblib`

Existing configs: `config/notebook_parameters_default.yml`, `config/mantle_test.yml`, `config/cache_test.yml`, `config/zahirovic_baseline.yml`, `config/mantle_only.yml`.

## Architecture

**Entry**: `run_notebooks.py` → `papermill` · **Config**: `lib/load_params.py` (`get_params`) → `lib/paths.py` (`PathConfigManager`)

### `lib/` — core library

- **`paths.py` (`PathConfigManager`)**: single source of truth for all file paths. Constructed from a config YAML; exposes `OUTPUT_DIR`, `TRAINING_DATA_PATH`, `GRID_DATA_PATH`, `MANTLE_DATA_DIR`, `PLATE_MODEL_DIR`, etc. `active_feature_sets` is a set of dot-notation keys (e.g. `subduction`, `subduction.carbonate`) derived recursively from `feature_sets` config; `use_features()` accepts the same. Passed between notebooks.
- **`load_params.py` (`get_params`)**: merges layered config YAML into flat dict for a given notebook. Called internally by `PathConfigManager`.
- **`grid_features.py` (`GridFeatureRegistry`)**: decorator-based registry for geospatial feature samplers. Register with `@features.register(name)` or `@features.register_batch(declares=[...])`. Each feature has an optional `coordinate_resolver` (`snap_to_mantle`, `snap_to_plate_model`, or default `reconstructed`). Results cached in a `DataFrame`.
- **`mantle_variables.py` (`MantleVariableRegistry`)**: registry for mantle dataset variables — base variables from netCDF and derived variables (e.g. `LAB_Depth`, `Sublithospheric_Cold_Anomaly_Thickness`, depth-averaged temperature deviations). Module-level singleton `variables` used by `grid_features.py`.
- **`pu.py`**: PU classifier creation (`create_classifier`), training data prep (`get_xy`), grid point generation (`generate_grid_points`), probability calculation (`calculate_probabilities`), raster output (`create_grids`). Uses `BaggingPuClassifier` from `pulearn`.
- **`plate_models.py`**: plate model loading via `plate-model-manager` (`PlateModelManager`). Key functions: `get_plate_reconstruction`, `get_plot_topologies`, `cache_plate_model`.
- **`calculate_convergence.py`**: subduction convergence extraction wrapping `ptt.subduction_convergence`.
- **`combine_point_data.py`**: combines deposit and unlabelled point data.
- **`generate_unlabelled_points.py`**: generates random unlabelled points within study area polygons.
- **`coregister_combined_point_data.py`**: joins point data to subduction zone kinematics via haversine nearest-neighbour.
- **`coregister_crustal_thickness.py`**: joins points to time-dependent crustal thickness rasters.
- **`coregister_ocean_rasters.py`**: joins subduction zone kinematic data to time-dependent ocean plate rasters (seafloor age, spreading rate, sediment thickness, carbonate thickness, crustal CO2) via plate-ID-masked haversine nearest-neighbour sampling.
- **`create_plate_maps.py`**: rasterises topological plate models to netCDF.
- **`assign_regions.py`**: spatial joins using `data_source/regions/regions.geojson`.
- **`misc.py`**: shared utilities — `reconstruct_by_topologies`, `filter_topological_features`, `load_data`, `calculate_slab_flux`.
- **`erodep/`**: erosion/deposition data extraction (`_extract_erodep.py`) and ML (`_ml.py`).
- **`extract_data/`**: paleotopography (`paleotopography/`), paleobathymetry (`paleobathymetry.py`), crustal thickness (`crustal_thickness.py`), LIP data (`lip_reconstruction.py`).
- **`cv.py`**: cross-validation with region-separated scoring.
- **`feature_importance.py`**, **`feature_selection.py`**, **`partial_dependence.py`**: ML analysis utilities.
- **`visualisation.py`**, **`animation.py`**: plotting and video output.
- **`slab_dip.py`**: slab dip prediction via `slabdip.predictor.SlabDipper`.

### Data directories

- `data_source/`: raw inputs — deposit CSVs in `data_source/deposits/`, plate model files in `data_source/plate_models/`, mantle netCDFs, `data_source/regions/regions.geojson`
- `data_extracted/`: extracted data from `00a`–`00c`; organised by plate model name (e.g. `data_extracted/clennett2020/`)
- `data_prepared/`: pre-prepared Zenodo fallback data
- `output/{run_name}/`: classifier `.joblib` files, probability grids, figures, `config_snapshot.yml`

### Submodules

- `submodules/CarbonateSedimentThickness`: carbonate sediment thickness
- `submodules/predicting-sediment-thickness`: ocean sediment thickness

### CI / tooling directories

- **`.github/`**: GitHub Actions workflow definitions for automated CI runs.
- **`.ruff_cache/`**: Auto-generated ruff linting cache (contains `CACHEDIR.TAG`); not tracked in version control.

## Adding new grid features

Register on the module-level `features` singleton in `lib/grid_features.py`:

```python
@features.register("my_feature_name", coords=snap_to_mantle)
def _my_feature(lons, lats, times):
    # return pd.Series or pd.DataFrame
    ...
```

For multi-column features use `@features.register_batch(declares=["col_a", "col_b"])`. Omit `coords` to use `reconstructed` (default). Results are cached automatically — do not manually cache.

## Adding new mantle variables

Register on the module-level `variables` singleton in `lib/mantle_variables.py`. Base variables load from netCDF; derived variables compute from registered base variables.

## Commit conventions

Conventional commits: `<type>(<scope>): <description>`

Types: `feat`, `fix`, `refactor`, `docs`, `chore`, `test` · Scopes: `lib`, `config`, `notebooks`, `data`

Examples:
- `feat(lib): add LAB-relative velocity feature to grid registry`
- `fix(config): correct mantle data path for zahirovic2022`
- `refactor(lib): simplify coordinate resolver caching logic`

Use `/commit` to generate a compliant message. Use `/commit-push-pr` to branch, commit, push, and open a PR.

## Before Committing

**IMPORTANT:** Before every git commit, you MUST ensure Caliber syncs agent configs with the latest code changes.

First, check if the pre-commit hook is already installed:
```bash
grep -q "caliber" .git/hooks/pre-commit 2>/dev/null && echo "hook-active" || echo "no-hook"
```

- If **hook-active**: the hook handles sync automatically — just commit normally. Tell the user: "Caliber will sync your agent configs automatically via the pre-commit hook."
- If **no-hook**: run Caliber manually before committing:
  1. Tell the user: "Caliber: Syncing agent configs with your latest changes..."
  2. Run: `caliber refresh && git add CALIBER_LEARNINGS.md CLAUDE.md .claude/ 2>/dev/null`
  3. After it completes, briefly tell the user what Caliber updated. Then proceed with the commit.

**Valid `caliber refresh` options:** `--quiet` (suppress output) and `--dry-run` (preview without writing). Do not pass any other flags — options like `--auto-approve`, `--debug`, or `--force` do not exist and will cause errors.

**`caliber config`** takes no flags — it runs an interactive provider setup. Do not pass `--provider`, `--api-key`, or `--endpoint`.

If `caliber` is not found, tell the user: "This project uses Caliber for agent config sync. Run /setup-caliber to get set up."
## Session Learnings

Read `CALIBER_LEARNINGS.md` for patterns and anti-patterns learned from previous sessions.
## Model Configuration

Recommended default: `claude-sonnet-4-6` with high effort (stronger reasoning; higher cost and latency than smaller models).
Smaller/faster models trade quality for speed and cost — pick what fits the task.
Pin your choice (`/model` in Claude Code, or `CALIBER_MODEL` when using Caliber with an API provider) so upstream default changes do not silently change behavior.

## Context Sync

This project uses [Caliber](https://github.com/caliber-ai-org/ai-setup) to keep AI agent configs in sync across Claude Code, Cursor, Copilot, and Codex.
Configs update automatically before each commit via `caliber refresh`.
If the pre-commit hook is not set up, run `/setup-caliber` to configure everything automatically.
