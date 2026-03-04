# Plate Reconstruction — Setup, Flexibility, and Known Issues

## Design Principle

The pipeline is reconstruction-agnostic. All plate-model-dependent paths flow from
`reconstruction.name` and `reconstruction.plate_model_dir` in the config. Adding a new
reconstruction requires: (1) populating `data/input_data/{name}/`, (2) writing a config file,
(3) running `00a` to generate rasters. No code changes are needed.

## How Reconstruction Loading Works

`lib/plate_models.py` provides two functions used throughout the notebooks:

- `get_plate_reconstruction(model_name, model_dir, ...)`: if `model_name=None`, scans `model_dir`
  for `.gpml`/`.rot` files; otherwise fetches via `PlateModelManager`. Returns a
  `pygplates.PlateReconstruction` object.
- `get_plot_topologies(...)`: same, also returns a `gplately.PlotTopologies` object.

Both functions handle topology filtering (flat slabs, inactive deforming networks) via
`lib/misc.py::filter_topological_features()`.

**Current notebook behaviour (before Phase 0 is complete):** `plate_model_dir = "plate_model"` is
hardcoded as a literal in `00b`, `00c`, and `01`. Phase 0.4 moves this into the config.

## Reconstruction-Specific Files Required

For each reconstruction, the following files are needed under `data/input_data/{name}/plate_model/`:

| File type | Used by | Notes |
|---|---|---|
| `.rot` rotation files | All notebooks | One or more; all files in the directory are loaded |
| `.gpml` topology/feature files | All notebooks | All files in the directory are loaded |
| Static polygons `.gpml` | `combine_point_data.py`, `00a` | Currently hardcoded as `Clennett_2020_StaticPolygons.gpml` — must be updated per reconstruction |
| COB terranes `.gpml` | `00a` sediment/carbonate | `Global_EarthByte_GeeK07_COB_Terranes.gpml` — likely reconstruction-independent (geological observation), verify |
| North America COBs `.gpml` | `00a` | Hardcoded as `plate_model/North_America_COBs.gpml` — Clennett2020-specific, needs equivalent for other models |

## Zahirovic2022 — Specific Setup Notes

1. **Check `plate-model-manager` availability.** Run `PlateModelManager().list_models()` to see
   whether `"zahirovic2022"` is registered. If yes, `check_plate_model()` in `lib/check_files.py`
   can auto-download it. If not, files must be placed manually.

2. **Static polygons.** Find the Zahirovic2022 static polygons file (`.gpml`) and update all
   references to `Clennett_2020_StaticPolygons.gpml`. There are instances in `00a` and in
   `lib/combine_point_data.py::_prepare_deposit_data()`.

3. **Sediment polynomial coefficients (UNRESOLVED — scientific decision required).**
   `00a` contains 9 hardcoded floats that parameterise the sediment thickness prediction model.
   These are derived from a polynomial fit to ocean proximity and seafloor age. If the coefficients
   were fitted using Clennett2020 age grids, they are invalid under Zahirovic2022 and must be
   re-derived. The `predicting-sediment-thickness` submodule (after `git submodule update --init`)
   contains the fitting code. Check the Alfonso 2024 paper supplementary for clarification before
   proceeding.

4. **Erosion/deposition data (UNRESOLVED — scientific decision required).**
   The `ErosionDeposition/` rasters are downloaded from Zenodo 14010839. Check Alfonso 2024
   data methods: were these computed under Clennett2020? If yes, using them in a Zahirovic2022
   pipeline is a methodological inconsistency. Options:
   - Re-compute under Zahirovic2022 (significant effort)
   - Use Clennett2020 erosion data with documented caveat
   - Exclude erosion as a feature for Zahirovic2022 runs

5. **Paleotopography.** Requires EarthByte webdav access:
   `https://www.earthbyte.org/webdav/ftp/earthbyte/Paleotopography/paleotopography-data.tgz`
   Uses Matthews2016 paleogeography — this is a separate model from either reconstruction and
   is likely geometrically compatible with both.

## Hardcoded Paths to Update in `00a` for Zahirovic2022

These are all in `00a-generate_data.ipynb` and must be generalised or made config-driven:

| Hardcoded path | Location in notebook | Required action |
|---|---|---|
| `plate_model/StaticGeometries/AgeGridInput/CombinedTerranes.gpml` | Coastlines for `gplot` | Move to config or `plate_models.py` |
| `plate_model/North_America_COBs.gpml` | Sediment thickness | Move to config |
| `plate_model/StaticGeometries/AgeGridInput/Global_EarthByte_GeeK07_COB_Terranes.gpml` | Sediment thickness | Assess if reconstruction-independent; if so, store outside `plate_model/` |
| Zenodo erosion download URL | Erosion cell | Conditioned on reconstruction choice |

## Adding a New Reconstruction (Future)

1. Populate `data/input_data/{new_name}/plate_model/` with `.rot` and `.gpml` files.
2. Add the static polygons filename to the config.
3. Run `00a` targeting `data/input_data/{new_name}/grids/` as output.
4. Write `config/run_baseline_{new_name}.yml`.
5. No changes to `lib/` are needed if `get_plate_reconstruction()` receives the correct `model_dir`.
