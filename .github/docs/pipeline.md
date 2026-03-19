# Pipeline — Design Context and Phase Guide

For task status, assignments, and progress: **[GitHub issue tracker](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues)**

## Milestones

| Milestone | Phase | Issues |
|---|---|---|
| [M1 — Repository Foundations](https://github.com/MaxEtherington/PUB-framework-Alfonso/milestone/1) | Phase 0 | #1–#9 |
| [M2 — Zahirovic2022 Baseline](https://github.com/MaxEtherington/PUB-framework-Alfonso/milestone/2) | Phases 1A, 1B, 2 | #10–#20 |
| [M3 — Mantle Feature Integration](https://github.com/MaxEtherington/PUB-framework-Alfonso/milestone/3) | Phases 3–4 | #21–#25 |
| [M4 — Comparison & Analysis](https://github.com/MaxEtherington/PUB-framework-Alfonso/milestone/4) | Phase 5 | #26–#27 |

---

## Architectural Constraint — Config Handoff to Notebooks

`run_notebooks.py` passes `parameters=None` to papermill; there is no parameter injection mechanism. Config is passed via a resolved YAML file written to disk by `run_notebooks.py::prepare_run()` (or `lib/setup_run.py::run_setup()` when using `--setup`).

The resolved config is written to `config/.run_config.yml` and a snapshot to `output/{run_name}/config_snapshot.yml`. Each notebook reads from `config/.run_config.yml` via `lib/load_params.get_params()`. Do not point notebooks at `notebook_parameters_default.yml` directly.

---

## Critical Path

```
Phase 0 → Phase 1A → Phase 1B (highest uncertainty) → Phase 2 (gate #20)
    → Phase 3 (#22 gates #23)
        → Phase 4 → Phase 5
```

---

## Phase 6 — Aspirational (design only, do not implement)

- **Craton reference polygon:** different reference geometry + different feature set. Abstraction point is `lib/create_study_area_polygons.py`.
- **Alternative reconstruction (Cao2024):** requires only a new `data/cao2024/` directory and a new config file — no code changes needed.
- **BayesSearchCV integration:** consult `PUB-framework-Ehsan/MPM_Porphyry_NSW.ipynb` and `lib_mpm.py`. Revisit after a validated baseline exists.
- **Continuous tonnage weighting:** `CONST_WEIGHTS_COLUMN = "Cu (Mt)"` exists in `lib/pu.py` but is not activated. One-line change when ready.
