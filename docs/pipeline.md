# Pipeline — Stages, Status, and Task List

## Current Status

The repository is a clean fork of Alfonso et al. (2024). No project-specific modifications have
been made yet. All phases below are pending.

---

## Phase 0 — Repository Foundations

**Goal:** Make the config system reconstruction-agnostic and add CLI-driven run isolation.
All items here are prerequisites for everything else.

- [ ] **0.1** Fix `lib/assign_regions.py` broken default path.
  `DEFAULT_REGIONS_FILE` → `Path(__file__).parent.parent / "regions.geojson"`

- [ ] **0.2** Run `git submodule update --init --recursive`.
  Required before attempting `00a-generate_data.ipynb`.

- [ ] **0.3** Add new keys to `notebook_parameters_default.yml`:
  - `reconstruction.name` (e.g. `"zahirovic2022"`)
  - `reconstruction.plate_model_dir`
  - `raster_data_dir` (where `00a` writes grids; constructed from `reconstruction.name`)
  - `reference_feature` (e.g. `"trenches"`)
  - `buffer_km` (e.g. `600`)
  - `use_mantle_features` (bool, controls whether `01` reads mantle-augmented data)
  - `gadopt_run_name` (e.g. `"v1_pre_craton_fix"`)

- [ ] **0.4** Move `plate_model_dir = "plate_model"` literal out of `00b`, `00c`, `01` notebooks
  and read from `params["reconstruction"]["plate_model_dir"]` instead.

- [ ] **0.5** Update `00b` and `00c` to read raster grids from `raster_data_dir`
  (not `extracted_data_dir`). Currently grids are sourced as
  `os.path.join(extracted_data_dir, "SeafloorAge")` etc. — ~8 path references per notebook.

- [ ] **0.6** Write `main.py`:
  - CLI: `python main.py --config config/run_mantle_v1.yml [--notebooks 00b 00c 01]`
  - Merges run config over `notebook_parameters_default.yml`
  - Constructs derived paths: `raster_data_dir`, `extracted_data_dir`, `output_dir`
    from `reconstruction.name`, `reference_feature`, `buffer_km`, `run_name`
  - Resolves all paths to absolute (paths in config are relative to workspace root)
  - Writes merged/resolved YAML to a **temp file** that replaces `notebook_parameters_default.yml`
    as the config path read by notebooks
  - **Critical constraint**: `run_notebooks.py` passes `parameters=None` to papermill — there is
    no parameter injection mechanism. The only way to pass config to notebooks is by writing a
    modified YAML file and having notebooks read it. `main.py` must write the resolved config
    to a temp path and update the `config_file` variable inside each notebook, OR write it to
    `notebook_parameters_default.yml` directly (simpler but destructive).
    Recommended: write to a well-known temp path (e.g. `config/.resolved_run_config.yml`) and
    update the `config_file = ...` line in each notebook to read from that path.
  - Writes `config_snapshot.yml` to `outputs/{run_name}/` at start of run

- [ ] **0.7** Create data directory skeleton (see `docs/data-layout.md`).

- [ ] **0.8** Write `config/run_baseline_clennett.yml` — Clennett2020, no mantle features,
  uses Zenodo pre-extracted data, validates pipeline is intact.

---

## Phase 1A — Zahirovic2022 Plate Model Setup

- [ ] **1A.1** Obtain Zahirovic2022 plate model files (`.rot`, topology `.gpml`, static polygons).
  Check whether `plate-model-manager` has `"zahirovic2022"` registered.
  Place under `data/input_data/zahirovic2022/plate_model/`.

- [ ] **1A.2** Identify the correct static polygons file for Zahirovic2022.
  Currently hardcoded as `StaticGeometries/StaticPolygons/Clennett_2020_StaticPolygons.gpml`
  in several places. Update all references.

- [ ] **1A.3** Write `config/run_baseline_zahirovic2022.yml`:
  ```yaml
  reconstruction:
    name: zahirovic2022
    plate_model_dir: data/input_data/zahirovic2022/plate_model
  reference_feature: trenches
  buffer_km: 600
  run_name: baseline_zahirovic2022
  use_mantle_features: false
  ```

---

## Phase 1B — Raster Generation Under Zahirovic2022 (via `00a`)

**High uncertainty phase.** Work through each raster type sequentially; blockers are likely.

- [ ] **1B.1** Seafloor age and spreading rate grids.
  `gplately.SeafloorGrid` with Zahirovic2022 topology files → `data/input_data/zahirovic2022/grids/SeafloorAge/`
  and `SpreadingRate/`.

- [ ] **1B.2** Sediment thickness — **requires scientific decision first.**
  The polynomial model coefficients in `00a` (9 hardcoded floats) were fitted to Clennett2020
  seafloor ages. Determine whether they must be re-derived for Zahirovic2022. The
  `predicting-sediment-thickness` submodule contains the fitting code — check after 0.2.

- [ ] **1B.3** Fix COB file paths for Zahirovic2022.
  `plate_model/North_America_COBs.gpml` and
  `plate_model/StaticGeometries/AgeGridInput/Global_EarthByte_GeeK07_COB_Terranes.gpml`
  are hardcoded Clennett2020 paths. Assess whether GeeK07 COB file is
  reconstruction-independent (likely yes — consult Alfonso 2024 supplementary).

- [ ] **1B.4** Paleotopography download.
  Requires access to `https://www.earthbyte.org/webdav/ftp/earthbyte/Paleotopography/`.
  Matthews2016 paleogeography is geometrically independent of the plate model.

- [ ] **1B.5** Erosion/deposition data — **requires scientific decision.**
  Currently downloaded from Zenodo 14010839. Check Alfonso 2024 data methods: is this
  dataset reconstruction-specific (computed under Clennett2020) or independent? If
  reconstruction-specific and no Zahirovic2022 equivalent exists, this is either a known
  limitation to document or a blocker requiring re-computation.

- [ ] **1B.6** Run `00a` end-to-end targeting `data/input_data/zahirovic2022/grids/`.
  Validate output netCDFs by plotting a sample depth slice.

---

## Phase 2 — Baseline Validation (Gate)

**Do not proceed to Phase 3/4 until this passes.**

- [ ] **2.1** Run `00b` with `run_baseline_zahirovic2022.yml` → verify `training_data_global.csv`.
- [ ] **2.2** Run `00c` → verify `grid_data.csv`.
- [ ] **2.3** Run `01` → verify a probability map is produced and known deposits score higher
  probability than random background. Qualitative comparison to Alfonso 2024 published figures.

---

## Phase 3 — Mantle Feature Exploration

- [ ] **3.1** Fix bugs in mantle-processing notebooks (see `docs/mantle-extraction.md`).
- [ ] **3.2** Explore G-ADOPT variables at known deposit paleolocations vs. random arc points.
  Determine: which variables (`Temperature_Deviation_CG`, `Radial_Velocity`, derived products),
  which depth levels or ranges, point values vs. summary statistics.
- [ ] **3.3** Record the feature specification decision as a comment block at the top of
  `00d-extract_mantle_features.ipynb` before writing any extraction code.

---

## Phase 4 — Mantle Extraction Pipeline

- [ ] **4.1** Write `lib/extract_mantle_features.py`.
  Converts `mantle-processing/point-samping.ipynb` into a proper library function.
  Signature mirrors other coregistration functions in `lib/`:
  input DataFrame (lon, lat, age) + netCDF dir + variables + depth spec → DataFrame with appended columns.
  Handles: time interpolation between available timesteps, graceful NaN for out-of-domain points.

- [ ] **4.2** Write `00d-extract_mantle_features.ipynb`.
  Reads `training_data_global.csv` and `grid_data.csv`, calls `extract_mantle_features()`,
  writes `training_data_global_with_mantle.csv` and `grid_data_with_mantle.csv`.
  Reads `gadopt_run_name` and `mantle_data_dir` from config.

- [ ] **4.3** Update `01-create_classifiers.ipynb` to select between mantle-augmented and
  standard training data based on `use_mantle_features` config key.

---

## Phase 5 — Comparison Runs

- [ ] **5.1** Write `config/run_mantle_v1.yml` (Zahirovic2022 rasters + mantle, pre-craton fix).
- [ ] **5.2** Write `config/run_mantle_v2.yml` (post-craton fix G-ADOPT outputs).
  **Cannot execute until G-ADOPT craton fix outputs exist**, but write the config now.
- [ ] **5.3** Execute baseline and v1 runs. Compare `feature_importance.csv` outputs.
  First scientific result: do mantle features appear in the importance rankings?

---

## Phase 6 — Aspirational (design only, do not implement)

- **Craton reference polygon**: different reference geometry + different feature set.
  Document the abstraction point in `lib/create_study_area_polygons.py` only.
- **Alternative reconstruction (Cao2024)**: the config/path system from Phase 0 already
  accommodates this. Requires only a new `data/input_data/cao2024/` and a new config file.
- **BayesSearchCV integration**: consult `PUB-framework-Ehsan/MPM_Porphyry_NSW.ipynb` and
  `PUB-framework-Ehsan/lib_mpm.py`. Revisit after a validated baseline result exists.
- **Continuous tonnage weighting** (`Cu_Tonnage_Mt`): already present in the deposit database.
  `CONST_WEIGHTS_COLUMN = "Cu (Mt)"` exists in `lib/pu.py` but is never activated.
  One-line change when ready.

---

## Critical Path

```
Phase 0 → Phase 1A → Phase 1B (highest uncertainty) → Phase 2 (gate)
    → Phase 3 (exploration, parallel with Phase 4 setup)
        → Phase 4 → Phase 5
```

Phases 1B.2 (sediment polynomial) and 1B.5 (erosion reconstruction-dependence) are the most
likely scientific blockers. Resolve both before committing to the full Zahirovic2022 raster
generation approach.
