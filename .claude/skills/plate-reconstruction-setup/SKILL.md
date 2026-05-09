---
name: plate-reconstruction-setup
description: Sets up gplately PlateReconstruction and PlotTopologies objects using lib/plate_models.py. Use when user says 'plate reconstruction', 'load plate model', 'PlateReconstruction', 'PlotTopologies', 'zahirovic2022 model', or works in 00a/00b/00c notebooks or lib/plate_models.py. Covers PMM fetch, local file fallback via .metadata.json, filter_topologies for inactive networks, and the topology temp-file lifetime rule. Do NOT use for raster generation, feature extraction, or classifier training.
paths:
  - lib/plate_models.py
  - lib/paths.py
  - 00a-generate_data.ipynb
  - 00b-extract_training_data.ipynb
  - 00c-extract_grid_data.ipynb
---
# Plate Reconstruction Setup

## Critical

- **Never** call `get_plate_reconstruction()` or `get_plot_topologies()` with `model_name` derived from the config when `use_provided_plate_model: True`. When using provided local files, pass `model_name=None`.
- **When `filter_topologies=True`**, the function returns a **tuple** `(plate_reconstruction, tf)`. You **must** bind `tf` to a variable that lives for the entire notebook scope — if it is garbage-collected, the underlying temp `.gpml` file is deleted and the reconstruction breaks silently.
- All paths come from `PathConfigManager` in `lib/paths.py`. Never hardcode paths. Read `paths.PLATE_MODEL_DIR` for `model_dir`.
- `model_dir` must be passed as `str(paths.PLATE_MODEL_DIR)` — `pathlib.Path` objects are accepted but `str` is safer across gplately versions.
- Notebooks read config via `lib/load_params.get_params(config_path, notebook)`. Do not point notebooks at `notebook_parameters_default.yml` directly — always read `config/.run_config.yml`.

## Instructions

### Step 1 — Load config and derive paths

```python
from lib.paths import PathConfigManager

CONFIG_PATH = "config/.run_config.yml"  # always this path in notebooks
NOTEBOOK = "00b"  # replace with the current notebook code

paths = PathConfigManager(config_path=CONFIG_PATH, notebook=NOTEBOOK)
config = paths.config
```

Verify before proceeding: `paths.PLATE_MODEL_DIR` exists on disk (created by `run_notebooks.py --setup`).

### Step 2 — Determine model_name from config

```python
# use_provided_plate_model: True  → local files, pass model_name=None
# use_provided_plate_model: False → PMM fetch, pass plate_model_name string
if paths.use_provided_plate_model:
    model_name = None
else:
    model_name = config['plate_model']['plate_model_name']  # e.g. 'zahirovic2022'

model_dir = str(paths.PLATE_MODEL_DIR)
```

Verify: when `model_name` is not `None`, confirm `data/{plate_model_name}/plate_model/` is populated or internet is available.

### Step 3 — Build PlateReconstruction

**Without topology filtering (most notebooks):**

```python
from lib.plate_models import get_plate_reconstruction

plate_reconstruction = get_plate_reconstruction(
    model_name=model_name,
    model_dir=model_dir,
    anchor_plate_id=0,
    filter_topologies=False,
)
```

**With topology filtering (subduction convergence notebooks that need clean topologies):**

```python
from lib.plate_models import get_plate_reconstruction

plate_reconstruction, topology_tf = get_plate_reconstruction(
    model_name=model_name,
    model_dir=model_dir,
    anchor_plate_id=0,
    filter_topologies=True,
)
# topology_tf must stay in scope — do not del it
```

Verify: no `FileNotFoundError` raised — if it is, see Common Issues #1.

### Step 4 — Build PlotTopologies (if needed)

Reuse the `plate_reconstruction` built in Step 3 to avoid re-fetching model files:

```python
from lib.plate_models import get_plot_topologies

plot_topologies = get_plot_topologies(
    model_name=model_name,
    model_dir=model_dir,
    anchor_plate_id=0,
    time=0,                             # set to desired reconstruction age (Ma)
    plate_reconstruction=plate_reconstruction,  # reuse from Step 3
    filter_topologies=False,            # match the flag used in Step 3
)
```

If `filter_topologies=True` was used in Step 3, pass the same flag here — `get_plot_topologies` handles the temp-file internally and attaches it to the returned object via `plot_topologies._topology_tf`.

Verify: `plot_topologies.plate_reconstruction` is not `None`.

### Step 5 — Inject into GridFeatureRegistry (notebook 00c only)

When working in `00c-extract_grid_data.ipynb`:

```python
# registry is constructed earlier in the notebook
registry.plate_reconstruction = plate_reconstruction
```

Verify: accessing `registry.plate_reconstruction` does not raise `RuntimeError`.

## Examples

**User says:** "Load the Zahirovic2022 plate model for subduction feature extraction."

**Config (`zahirovic_baseline.yml`):**
```yaml
all_notebooks:
  plate_model:
    use_provided_plate_model: False
    plate_model_name: zahirovic2022
```

**Actions taken:**

```python
from lib.paths import PathConfigManager
from lib.plate_models import get_plate_reconstruction, get_plot_topologies

paths = PathConfigManager(config_path="config/.run_config.yml", notebook="00b")
config = paths.config

model_name = config['plate_model']['plate_model_name']  # 'zahirovic2022'
model_dir  = str(paths.PLATE_MODEL_DIR)                 # '.../data_source/plate_models/zahirovic2022'

plate_reconstruction = get_plate_reconstruction(
    model_name=model_name,
    model_dir=model_dir,
    anchor_plate_id=0,
    filter_topologies=False,
)

plot_topologies = get_plot_topologies(
    model_name=model_name,
    model_dir=model_dir,
    plate_reconstruction=plate_reconstruction,
    time=100,
)
```

**Result:** `plate_reconstruction` (gplately `PlateReconstruction`) and `plot_topologies` (gplately `PlotTopologies`) ready for subduction convergence calculations and spatial joins.

---

**User says:** "Load the provided Alfonso2024 reconstruction with topology filtering."

**Config:**
```yaml
all_notebooks:
  plate_model:
    use_provided_plate_model: True
    plate_model_name: alfonso2024_default
```

**Actions taken:**

```python
model_name = None  # use_provided_plate_model: True
model_dir  = str(paths.PLATE_MODEL_DIR)

plate_reconstruction, topology_tf = get_plate_reconstruction(
    model_name=model_name,
    model_dir=model_dir,
    filter_topologies=True,
)
# topology_tf stays in scope until notebook finishes
```

## Common Issues

**`FileNotFoundError: Missing required plate model files. Found in ... Rotations: None`**
- PMM fetch failed and no local fallback found. Fix:
  1. Check internet access or run `python run_notebooks.py --cache-remote --config <your_config.yml>` on a machine with internet first.
  2. Verify `data_source/plate_models/{plate_model_name}/` exists and contains `.rot` + `.gpml`/`.gpmlz` files.
  3. If local files exist but PMM fetch still fails, confirm `data_source/plate_models/{plate_model_name}/.metadata.json` is present (created by PMM on first successful fetch).

**`RuntimeError: plate_reconstruction has not been set on the feature registry`** (notebook 00c)
- You called a registry method before injecting the reconstruction. Fix: add `registry.plate_reconstruction = plate_reconstruction` before the first call that uses it.

**Topology temp file deleted mid-run (silent wrong results)**
- Caused by not binding the second return value when `filter_topologies=True`. Always unpack as `plate_reconstruction, topology_tf = get_plate_reconstruction(..., filter_topologies=True)` and never `del topology_tf`.

**`get_plot_topologies` re-fetches model files even though `plate_reconstruction` already exists**
- You forgot to pass `plate_reconstruction=plate_reconstruction` to `get_plot_topologies`. When omitted, it fetches from scratch. Always pass the already-built object.

**Wrong `model_dir` passed (e.g. repo root instead of plate_model subdirectory)**
- `paths.PLATE_MODEL_DIR` resolves to `{repo_root}/data_source/plate_models/{plate_model_name}`. Pass `str(paths.PLATE_MODEL_DIR)` — not `str(paths.SOURCE_DATA_DIR)` or a hand-constructed string.