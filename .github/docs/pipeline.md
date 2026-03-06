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

## Phase 0 — Repository Foundations

**Milestone:** [M1](https://github.com/MaxEtherington/PUB-framework-Alfonso/milestone/1) | **Issues:** [#1](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/1)–[#9](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/9)

**Goal:** Make the config system reconstruction-agnostic and add CLI-driven run isolation. All Phase 0 items are prerequisites for everything else.

### Architectural Constraint — Config Handoff to Notebooks

`run_notebooks.py` passes `parameters=None` to papermill; there is no parameter injection mechanism. Config must be passed via a resolved YAML file written to disk.

`main.py` ([#6](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/6)) writes the merged config to `config/.resolved_run_config.yml` and a reproducibility snapshot to `outputs/{run_name}/config_snapshot.yml`. Each notebook's `config_file` variable must point to the resolved path, not to `notebook_parameters_default.yml` directly.

---

## Phase 1A — Zahirovic2022 Plate Model Setup

**Milestone:** [M2](https://github.com/MaxEtherington/PUB-framework-Alfonso/milestone/2) | **Issues:** [#11](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/11)–[#13](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/13)

Obtain model files, identify correct static polygons, and write the baseline config. Static polygons are Zahirovic2022-specific (see [#12](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/12)); COB terranes may be reconstruction-independent — verify in [#16](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/16).

---

## Phase 1B — Raster Generation Under Zahirovic2022 (via `00a`)

**Milestone:** [M2](https://github.com/MaxEtherington/PUB-framework-Alfonso/milestone/2) | **Issues:** [#10](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/10), [#14](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/14)–[#19](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/19)

**High-uncertainty phase.** Work through each raster type sequentially; blockers are expected.

### Scientific Decision Point — Sediment Polynomial Coefficients ([#10](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/10))

`00a` contains 9 hardcoded polynomial coefficients for the silicic sediment thickness model. If these were fitted using Clennett2020 seafloor ages, they cannot be transferred to Zahirovic2022 without re-fitting (using `submodules/predicting-sediment-thickness/`). Resolve [#10](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/10) before committing to the full raster generation approach — it gates [#15](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/15).

### Resolved — Erosion/Deposition Excluded

Alfonso 2024 does not include erosion as a training feature (confirmed by feature importance tables and Methods). This project follows the published model. Document as a stated caveat in the thesis if surface process feedbacks are discussed.

---

## Phase 2 — Baseline Validation Gate

**Milestone:** [M2](https://github.com/MaxEtherington/PUB-framework-Alfonso/milestone/2) | **Issue:** [#20](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/20)

**Do not proceed to Phase 3/4 until [#20](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/20) is closed.** Acceptance criteria (output file validation, deposit vs. background probability check, qualitative comparison to Alfonso 2024 Fig. 1–2) are defined on the issue.

---

## Phase 3 — Mantle Feature Exploration

**Milestone:** [M3](https://github.com/MaxEtherington/PUB-framework-Alfonso/milestone/3) | **Issues:** [#21](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/21)–[#22](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/22)

Fix mantle-processing notebook bugs ([#21](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/21)) then run the deposit-vs-background exploratory analysis. The chosen feature specification (variable × depth spec × aggregation) must be recorded as a comment block at the top of `00d-extract_mantle_features.ipynb` before any extraction code is written. [#23](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/23) is blocked on [#22](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/22).

---

## Phase 4 — Mantle Extraction Pipeline

**Milestone:** [M3](https://github.com/MaxEtherington/PUB-framework-Alfonso/milestone/3) | **Issues:** [#23](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/23)–[#25](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/25)

`lib/extract_mantle_features.py` ([#23](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/23)) mirrors the interface of existing `lib/` coregistration modules: input DataFrame (lon, lat, age) + netCDF dir + variables + depth spec → DataFrame with appended columns. Time interpolation between available timesteps and graceful NaN for out-of-domain points are both required. `01` ([#25](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/25)) switches between standard and mantle-augmented input data via the `use_mantle_features` config key.

---

## Phase 5 — Comparison Runs

**Milestone:** [M4](https://github.com/MaxEtherington/PUB-framework-Alfonso/milestone/4) | **Issues:** [#26](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/26)–[#27](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/27)

[#27](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/27) (mantle_v2 config) can be written immediately but **cannot be executed** until post-craton-fix G-ADOPT outputs are delivered. Do not block the thesis timeline on this; the baseline vs. mantle_v1 comparison ([#26](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/26)) is the primary scientific result.

---

## Phase 6 — Aspirational (design only, do not implement)

- **Craton reference polygon:** different reference geometry + different feature set. Abstraction point is `lib/create_study_area_polygons.py`.
- **Alternative reconstruction (Cao2024):** the config/path system from Phase 0 already accommodates this — requires only a new `data/input_data/cao2024/` and a new config file.
- **BayesSearchCV integration:** consult `PUB-framework-Ehsan/MPM_Porphyry_NSW.ipynb` and `PUB-framework-Ehsan/lib_mpm.py`. Revisit after a validated baseline result exists.
- **Continuous tonnage weighting:** `CONST_WEIGHTS_COLUMN = "Cu (Mt)"` exists in `lib/pu.py` but is never activated. One-line change when ready.

---

## Critical Path

```
Phase 0 → Phase 1A → Phase 1B (highest uncertainty) → Phase 2 (gate #20)
    → Phase 3 (#22 gates #23)
        → Phase 4 → Phase 5
```

[#10](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/10) (sediment polynomial) and [#16](https://github.com/MaxEtherington/PUB-framework-Alfonso/issues/16) (COB file assessment) are the most likely scientific blockers. Resolve both before committing to the full Zahirovic2022 raster generation approach.
