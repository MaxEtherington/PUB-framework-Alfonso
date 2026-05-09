---
paths:
  - nb_scripts/**
  - "*.ipynb"
---

# Notebook / Script Conventions

## CRITICAL: always edit nb_scripts/, never .ipynb directly
- All notebook logic lives in `nb_scripts/<notebook>.py` (jupytext percent format)
- `.ipynb` files are derived — never edit them programmatically
- **Syncing is automatic**: a `PostToolUse` hook in `.claude/settings.json` runs `jupytext --sync` immediately after any `Edit` or `Write` to a file matching `nb_scripts/*.py`. No manual sync step is needed.
- To sync manually (e.g. after editing outside Claude Code): `jupytext --sync nb_scripts/<notebook>.py`

## Running notebooks
- Via `run_notebooks.py`: `python run_notebooks.py --config config/notebook_parameters_default.yml --notebooks 01`
- `-o` flag overwrites the `.ipynb` in-place
- `--setup` validates config and creates directories without running

## Config access in notebooks
- Import `PathConfigManager` from `lib.paths`; construct with the config path injected by `papermill`
- Use `p.use_features('mantle')` to check feature_sets flags before computing
- Config snapshot auto-saved to `output/{run_name}/config_snapshot.yml`
