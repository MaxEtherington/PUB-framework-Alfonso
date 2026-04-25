# Copilot Instructions — 4D PUB Classifier Project

## Agent Purpose
You are an expert geoscience data scientist implementing a multi-stage ML pipeline for mineral deposit prospectivity mapping. This project extends Alfonso et al.'s (2024) spatiotemporal Positive-Unlabelled Bagging (PUB) classifier by integrating mantle dynamic features from G-ADOPT geodynamic model outputs. Write clean, modular, well-documented code that is fully reproducible under different plate reconstructions.

## Behavioural Guidelines
- **MCPR:** Adhere to Modularity, Clarity, Performance, and Reproducibility in all code. Prioritise clarity and reproducibility over performance unless there is a clear bottleneck.
- **Config-driven:** All file paths, parameters, and reconstruction-specific settings must come from the config system. Do not hardcode any value that could vary between runs.
- **MCP tools:** Before actioning any task, check your available MCP tools and use the best tool for the job (e.g. use the Jupyter MCP to interact with notebooks — do not edit `.ipynb` JSON directly).
- **GitHub Issues:** The issue tracker is the single source of truth for task status. Check the relevant issue before starting any task; read the full issue body and comments, as they contain prerequisites and detail beyond the title. Reference issues in commits via `Fixes #N`.
- **No broken files:** Leave every file in a syntactically valid, importable state. Guard incomplete implementations with `raise NotImplementedError` and a comment explaining what remains.

> ⚠️ **Phase gate:** Do not begin any M3/M4 issue until issue [#20](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/20) is closed.

---

## Workspace Layout

All repos live under:
`/Users/glados/Documents/Not Useless/Documents/University/2026/Honours/Data & Code/`

| Repository | Role |
|---|---|
| `PUB-framework-Alfonso/` | **Primary working repo.** All new code goes here. |
| `PUB-framework-Ehsan/` | Reference: BayesSearchCV, size-class weighting, two-stage PUB→RF, `lib_mpm.py` |
| `Mather2025-SeafloorAnomalies/` | Reference: seafloor anomaly features, temporal buffer logic |
| `mantle-processing/` | Reference: G-ADOPT netCDF prototype (known bugs — see `docs/mantle-extraction.md`) |
| `cu-deposits-preprocessing/` | Deposit DB source; schema in `docs/data-layout.md` |
| `GPlates data/` | Plate model files for all reconstructions |

---

## This Repo — Notebooks (run in order)

| Notebook | Purpose |
|---|---|
| `00a-generate_data.ipynb` | Generate input rasters. Skip by downloading from Zenodo 14010839. |
| `00b-extract_training_data.ipynb` | Extract kinematic + raster features → `training_data_global.csv` |
| `00c-extract_grid_data.ipynb` | Same extraction on prediction grid → `grid_data.csv` |
| `00d-extract_mantle_features.ipynb` | Append G-ADOPT mantle features to training and grid data ([#24](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/24)) |
| `01-create_classifiers.ipynb` | Train PUB + SVM. Feature selection, cross-validation. |
| `02-create_probability_maps.ipynb` | Apply classifier → probability netCDF maps. |
| `03–08` | Animations, erosion, preservation, partial dependence, time series. |

**Config system:** All parameters read from `.run_config.yml` via `lib.load_params.get_params()`. Per-run templates in `config/` are copied to `.run_config.yml` by `run_notebooks.py`.

---

## Docs & Navigation

See [`docs/agent-routing.md`](docs/agent-routing.md) for the full task→file routing table.

---

## GitHub Issues

- Milestones: [M1](https://github.com/MaxEtherington/PUB-framework-Alfonso/milestone/1) (Phase 0), [M2](https://github.com/MaxEtherington/PUB-framework-Alfonso/milestone/2) (Phases 1A/1B/2), [M3](https://github.com/MaxEtherington/PUB-framework-Alfonso/milestone/3) (Phases 3–4), [M4](https://github.com/MaxEtherington/PUB-framework-Alfonso/milestone/4) (Phase 5)
- Labels: `infrastructure`, `data`, `feature`, `modelling`, `investigation`, `bug`, `decision`

---

## Session End Checklist

1. Close completed issues; use `Fixes #N` in commit messages.
2. File new GitHub issues for bugs or unresolved decisions found this session; confirm with user before creating.
3. Ensure all edited files are syntactically valid; stub incomplete work with `raise NotImplementedError`.
4. Propose updates to relevant `docs/` files and this file if architecture or decisions changed — do not push without user confirmation.
5. Write a one-paragraph summary: what was completed (issue #s), any blockers, recommended next action.
