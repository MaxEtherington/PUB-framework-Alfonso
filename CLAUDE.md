# PUB Framework — Alfonso Honours Thesis

Spatio-temporal copper prospectivity prediction using positive-unlabelled (PU) machine learning on plate tectonic reconstructions.

@./.github/docs/pipeline.md
@./.github/docs/data-layout.md
@./.github/docs/reconstruction.md

## Setup

```bash
conda env create --file environment.yml
conda activate prospectivity
```

## Pipeline Execution

Always use `run_notebooks.py` (not raw Jupyter) for reproducible runs:

```bash
# Run full pipeline with a config
python run_notebooks.py --config config/mantle_test.yml --notebooks 00a 00b 00c 01 02

# Setup dirs only (no notebook execution)
python run_notebooks.py --config config/zahirovic_baseline.yml --setup

# Pre-cache remote data for HPC (no internet required later)
python run_notebooks.py --config config/mantle_test.yml --cache-remote

# Overwrite notebook files in-place with execution output
python run_notebooks.py --config config/notebook_parameters_default.yml --notebooks 01 -o
```

### Notebook Sequence

| Code | File | Purpose |
|------|------|---------|
| `00a` | `00a-generate_data.ipynb` | Download plate model + source data |
| `00b` | `00b-extract_training_data.ipynb` | Extract features for deposit points |
| `00c` | `00c-extract_grid_data.ipynb` | Extract features for 0.5° grid |
| `01` | `01-create_classifiers.ipynb` | Train PU + SVM classifiers |
| `02` | `02-create_probability_maps.ipynb` | Prospectivity maps |
| `03` | `03-create_probability_animations.ipynb` | Time-animated maps |
| `04` | `04-create_erosion_distribution.ipynb` | Erosion/deposition analysis |
| `05` | `05-create_preservation_maps.ipynb` | Preservation maps |
| `06` | `06-create_preservation_animations.ipynb` | Preservation animations |
| `07` | `07-partial_dependence.ipynb` | PDP plots via `lib/partial_dependence.py` |
| `08` | `08-time_series.ipynb` | Time series analysis |

## Architecture

**Config flow**: `config/*.yml` → `lib/load_params.py:get_params()` (merges `defaults` → `all_notebooks` → per-notebook) → `lib/paths.py:PathConfigManager` (all derived paths)

**Entry**: `run_notebooks.py` → `PathConfigManager` → `papermill.execute_notebook()`

**Key lib modules** (`lib/`):
- `plate_models.py` — `get_plate_reconstruction()`, `get_plot_topologies()` via `gplately.PlateModelManager`; falls back to local `.metadata.json` scan
- `paths.py` — `PathConfigManager`: single source for all output/input paths
- `load_params.py` — `get_params(config_path, notebook)`: ruamel.yaml loader with 3-level merge
- `generate_unlabelled_points.py` — uniform sphere sampling within study area GeoJSONs, joblib parallel
- `combine_point_data.py` — merge deposit CSVs + unlabelled points, assigns `overriding_plate_id` via `PlotTopologies.get_all_topologies()` spatial join
- `calculate_convergence.py` — wraps `ptt.subduction_convergence_over_time`, renames columns, joblib parallel
- `coregister_combined_point_data.py` — haversine `NearestNeighbors` (sklearn) join of points → subduction zones
- `coregister_crustal_thickness.py` — radius-based join to `crustal_thickness_{t}Ma.nc` rasters via xarray
- `coregister_ocean_rasters.py` — ocean raster coregistration
- `cv.py` — region-aware `StratifiedKFold` CV with roc_auc/average_precision/balanced_accuracy/f1
- `pu.py` — `get_xy()`, column constants (`CORRELATED_COLUMNS`, `COLUMNS_TO_DROP`, `PRESERVATION_COLUMNS`)
- `feature_selection.py` — Spearman correlation dendrogram (`scipy.cluster.hierarchy.ward`)
- `feature_importance.py` — Gini importance + Kendall tau ranking correlations
- `partial_dependence.py` — `sklearn.inspection.PartialDependenceDisplay` wrappers
- `visualisation.py` — figure utilities; uses `config/thesis.mplstyle`
- `animation.py` — moviepy + ffmpeg; auto-detects `hevc_videotoolbox` on Apple silicon
- `create_plate_maps.py` — rasterize topologies → netCDF via `rasterio.features.rasterize`
- `check_files.py` — Zenodo download (record/14010839); `check_prepared_data()`, `check_plate_model()`
- `cache_remote_data.py` — HPC pre-cache for PMM plate models + Zenodo bundles
- `assign_regions.py` — spatial join to `data_source/regions/regions.geojson`
- `misc.py` — `reconstruct_by_topologies()`, `filter_topological_features()`, `load_data()`
- `slab_dip.py` — `SlabDipper` wrapper; adds `slab_dip (degrees)` + `arc_trench_distance (km)`
- `erodep/` — `_extract_erodep.py`, `_ml.py`, `_visualisation.py`
- `extract_data/` — `paleobathymetry.py`, `crustal_thickness.py`, `crustal_co2.py`, `lip_reconstruction.py`, `paleotopography/` submodule
- `grid_features.py` — grid-based feature extraction
- `mantle_variables.py` — mantle variable handling

**Data sources** (`data_source/`):
- `deposits/` — `deposits.csv`, `deposits-Etherington.csv`, `VMS-deposits.csv`, `IOCG-deposits.csv`, `Porphyry-deposits.csv`, `SedCu-deposits.csv`
- `regions/regions.geojson` — study region polygons for `assign_regions()`
- `plate_models/zahirovic2022/` — zahirovic2022 plate model layers (Rotations, Topologies, StaticPolygons, Coastlines, ContinentalPolygons)

**Config files** (`config/`):
- `notebook_parameters_default.yml` — defaults: Alfonso2024 model, 0–170 Ma
- `zahirovic_baseline.yml` — zahirovic2022, 0–400 Ma
- `mantle_test.yml` — mantle feature test run
- `thesis.mplstyle` — matplotlib style for all figures

## Conventions

- **Parallelism**: `joblib.Parallel(n_jobs, verbose=int(verbose))` + `delayed()`; never `multiprocessing` directly
- **Verbose**: `bool` param, always print to `sys.stderr` not stdout
- **Paths**: use `PathConfigManager` — never hardcode output paths
- **Config params**: add to `config/notebook_parameters_default.yml` `defaults` section first
- **Data loading**: `lib/misc.py:load_data()` accepts both file path `str` and `DataFrame`
- **Labels**: `"positive"`, `"negative"`, `"unlabelled"` — exact strings, no variation
- **Coordinate columns**: `"lon"`, `"lat"`, `"age (Ma)"` — exact column names everywhere
- **Plate model**: `lib/plate_models.py:get_plate_reconstruction()` handles PMM fetch + local fallback
- **Jupytext**: notebooks sync to `nb_scripts/` as `.py:percent` (see `jupytext.toml`)
- **Submodules**: `submodules/CarbonateSedimentThickness`, `submodules/predicting-sediment-thickness`

@./.github/docs/contributing.md
@./.github/docs/agent-routing.md
@./.github/docs/mantle-extraction.md

<!-- caliber:managed:pre-commit -->
## Before Committing

**IMPORTANT:** Before every git commit, you MUST ensure Caliber syncs agent configs with the latest code changes.

First, check if the pre-commit hook is already installed:
```bash
grep -q "caliber" .git/hooks/pre-commit 2>/dev/null && echo "hook-active" || echo "no-hook"
```

- If **hook-active**: the hook handles sync automatically — just commit normally. Tell the user: "Caliber will sync your agent configs automatically via the pre-commit hook."
- If **no-hook**: run Caliber manually before committing:
  1. Tell the user: "Caliber: Syncing agent configs with your latest changes..."
  2. Run: `caliber refresh && git add CALIBER_LEARNINGS.md CLAUDE.md .claude/ .cursor/ .cursorrules .github/copilot-instructions.md .github/instructions/ 2>/dev/null`
  3. After it completes, briefly tell the user what Caliber updated. Then proceed with the commit.

**Valid `caliber refresh` options:** `--quiet` (suppress output) and `--dry-run` (preview without writing). Do not pass any other flags — options like `--auto-approve`, `--debug`, or `--force` do not exist and will cause errors.

**`caliber config`** takes no flags — it runs an interactive provider setup. Do not pass `--provider`, `--api-key`, or `--endpoint`.

If `caliber` is not found, tell the user: "This project uses Caliber for agent config sync. Run /setup-caliber to get set up."
<!-- /caliber:managed:pre-commit -->

<!-- caliber:managed:learnings -->
## Session Learnings

Read `CALIBER_LEARNINGS.md` for patterns and anti-patterns learned from previous sessions.
These are auto-extracted from real tool usage — treat them as project-specific rules.
<!-- /caliber:managed:learnings -->

<!-- caliber:managed:model-config -->
## Model Configuration

Recommended default: `claude-sonnet-4-6` with high effort (stronger reasoning; higher cost and latency than smaller models).
Smaller/faster models trade quality for speed and cost — pick what fits the task.
Pin your choice (`/model` in Claude Code, or `CALIBER_MODEL` when using Caliber with an API provider) so upstream default changes do not silently change behavior.

<!-- /caliber:managed:model-config -->

<!-- caliber:managed:sync -->
## Context Sync

This project uses [Caliber](https://github.com/caliber-ai-org/ai-setup) to keep AI agent configs in sync across Claude Code, Cursor, Copilot, and Codex.
Configs update automatically before each commit via `caliber refresh`.
If the pre-commit hook is not set up, run `/setup-caliber` to configure everything automatically.
<!-- /caliber:managed:sync -->
