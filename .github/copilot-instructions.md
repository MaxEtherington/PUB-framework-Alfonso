# PUB Framework — Copper Prospectivity ML Pipeline

Honours thesis pipeline predicting spatio-temporal copper deposit prospectivity using positive-unlabelled (PU) machine learning and plate tectonic reconstructions.

## Environment

```bash
conda env create --file environment.yml  # env name: prospectivity
conda activate prospectivity
```

## Pipeline Execution

```bash
# Run notebooks with a config (preferred over raw Jupyter)
python run_notebooks.py --config config/mantle_test.yml --notebooks 00b 00c 01
python run_notebooks.py --config config/zahirovic_baseline.yml --setup
python run_notebooks.py --config config/notebook_parameters_default.yml --cache-remote
```

Notebook sequence: `00a` → `00b` → `00c` → `01` → `02` → `03` → `04` → `05` → `06` → `07` → `08`

## Architecture

**Config**: `config/*.yml` parsed by `lib/load_params.py:get_params()` with 3-level merge · `lib/paths.py:PathConfigManager` exposes all derived paths

**Key modules** (`lib/`):
- `plate_models.py` — `get_plate_reconstruction()` via `gplately.PlateModelManager`; local fallback
- `generate_unlabelled_points.py` — uniform sphere sampling, joblib parallel
- `combine_point_data.py` — merge `data_source/deposits/*.csv` + unlabelled points
- `calculate_convergence.py` — `ptt.subduction_convergence_over_time` wrapper
- `coregister_combined_point_data.py` — haversine `NearestNeighbors` join to subduction zones
- `coregister_crustal_thickness.py` — radius join to `crustal_thickness_{t}Ma.nc` via xarray
- `cv.py` — region-aware `StratifiedKFold` with roc_auc/f1/balanced_accuracy metrics
- `pu.py` — `get_xy()`, `CORRELATED_COLUMNS`, `PRESERVATION_COLUMNS` constants
- `feature_selection.py` — Spearman correlation dendrogram
- `partial_dependence.py` — `PartialDependenceDisplay` wrappers
- `check_files.py` — Zenodo download (record/14010839)
- `assign_regions.py` — spatial join to `data_source/regions/regions.geojson`
- `misc.py` — `reconstruct_by_topologies()`, `load_data()`, `filter_topological_features()`
- `erodep/` — `_extract_erodep.py`, `_ml.py`
- `extract_data/` — `paleobathymetry.py`, `crustal_thickness.py`, `crustal_co2.py`, `paleotopography/`
- `grid_features.py`, `mantle_variables.py`

**Config files** (`config/`):
- `notebook_parameters_default.yml` — Alfonso2024 model, 0–170 Ma defaults
- `zahirovic_baseline.yml` — zahirovic2022, 0–400 Ma
- `mantle_test.yml` — mantle feature test run
- `thesis.mplstyle` — matplotlib style

**Data** (`data_source/`):
- `deposits/` — `deposits.csv`, `deposits-Etherington.csv`, `VMS-deposits.csv`, `IOCG-deposits.csv`, `Porphyry-deposits.csv`, `SedCu-deposits.csv`
- `regions/regions.geojson` — region polygons
- `plate_models/zahirovic2022/` — plate model layers (Rotations, Topologies, StaticPolygons, Coastlines)

## Conventions

- **Parallelism**: `joblib.Parallel(n_jobs, verbose=int(verbose))` + `delayed()` always
- **Verbose**: print to `sys.stderr`, never stdout; guard with `if verbose:`
- **Paths**: always `PathConfigManager` from `lib/paths.py` — never hardcode
- **Labels**: `"positive"`, `"negative"`, `"unlabelled"` in `"label"` column — exact strings
- **Coordinates**: `"lon"`, `"lat"`, `"age (Ma)"` — exact column names
- **Data loading**: `lib/misc.py:load_data()` accepts `str | Path | DataFrame`
- **Plate models**: `lib/plate_models.py:get_plate_reconstruction()` — PMM + local fallback
- **Jupytext**: notebooks sync to `nb_scripts/` as `.py:percent` (`jupytext.toml`)

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
  2. Run: `caliber refresh && git add CLAUDE.md .claude/ .cursor/ .cursorrules .github/copilot-instructions.md .github/instructions/ AGENTS.md CALIBER_LEARNINGS.md .agents/ .opencode/ 2>/dev/null`
  3. After it completes, briefly tell the user what Caliber updated. Then proceed with the commit.

**Valid `caliber refresh` options:** `--quiet` (suppress output) and `--dry-run` (preview without writing). Do not pass any other flags — options like `--auto-approve`, `--debug`, or `--force` do not exist and will cause errors.

**`caliber config`** takes no flags — it runs an interactive provider setup. Do not pass `--provider`, `--api-key`, or `--endpoint`.

If `caliber` is not found, tell the developer to set up Caliber by running `/setup-caliber` in Claude Code or Cursor. Alternatively, they can run these commands in their terminal:
```
npx @rely-ai/caliber hooks --install
npx @rely-ai/caliber refresh
```
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
If the pre-commit hook is not set up, the developer should run `/setup-caliber` in Claude Code or Cursor for automated setup. Alternatively, run in terminal:
```bash
npx @rely-ai/caliber hooks --install
npx @rely-ai/caliber refresh
git add CLAUDE.md .claude/ .cursor/ .cursorrules .github/copilot-instructions.md .github/instructions/ AGENTS.md CALIBER_LEARNINGS.md .agents/ .opencode/ 2>/dev/null
```
<!-- /caliber:managed:sync -->
