---
name: pu-learning-workflow
description: Guides positive-unlabelled (PU) classifier training using pulearn with the project's label conventions (positive/unlabelled/negative). Covers lib/pu.py get_xy() and create_classifier(), the Pipeline+BaggingPuClassifier pattern from 01-create_classifiers.ipynb, lib/cv.py perform_cv() region-aware cross-validation, scipy linkage-based feature selection, and lib/partial_dependence.py. Use when user says 'train classifier', 'cross-validate', 'feature selection', 'PU model', 'feature importance'. Do NOT use for data extraction (notebooks 00a–00c) or prospectivity map generation (notebook 02).
paths:
  - 01-create_classifiers.ipynb
  - lib/pu.py
  - lib/cv.py
  - lib/feature_selection.py
  - lib/partial_dependence.py
---
# PU Learning Workflow

## Critical

- **Label convention is non-negotiable.** Training data must have a `label` column with exactly these string values: `"positive"` (known deposits), `"unlabelled"` (background), `"negative"` (confirmed absence, rare). `get_xy()` encodes `positive→1`, everything else→0. Never pass integer labels directly to `BaggingPuClassifier`.
- **Never read `config/notebook_parameters_default.yml` directly.** All config flows through `lib/paths.py:PathConfigManager` which reads `config/.run_config.yml`.
- **PU training uses only `{positive, unlabelled}` rows.** CV test folds use only `{positive, negative}` rows. The split is explicit and intentional — do not mix these.
- **Drop correlated/metadata columns before fitting.** Use `COLUMNS_TO_DROP`, `PRESERVATION_COLUMNS`, and any project-specific `component_columns` before constructing `X`. `get_xy()` handles the standard sets automatically.
- **Deduplicate training data** before any fitting. Duplicates arise from the one-to-many spatial join in `combine_point_data.py`. Always call `data.drop_duplicates(subset=["lon", "lat", "age (Ma)"], keep="first")`.
- **Skip regions with < 50 deposits.** Regional models below this threshold are unreliable.

## Instructions

### Step 1 — Load config and paths

```python
from lib.paths import PathConfigManager
pcm = PathConfigManager("config/.run_config.yml", notebook="01")
pcm.create_directories()

training_data_filepath = pcm.TRAINING_DATA_PATH
output_dir = pcm.OUTPUT_DIR
random_seed = pcm.config["random_seed"]
n_jobs = pcm.config["n_jobs"]
```

Verify `pcm.TRAINING_DATA_PATH` exists before proceeding.

### Step 2 — Load and clean training data

```python
import pandas as pd
from lib.pu import COLUMNS_TO_DROP, PRESERVATION_COLUMNS

data = pd.read_csv(training_data_filepath)
data = data.drop_duplicates(subset=["lon", "lat", "age (Ma)"], keep="first")

# Skip under-represented regions
regions_to_skip = {
    region for region, subset in data.groupby("region")
    if (subset["label"] == "positive").sum() < 50
}
data = data[~data["region"].isin(regions_to_skip)]

# Drop metadata + preservation columns (+ any project-specific component_columns)
cleaned = data.drop(
    columns=list(COLUMNS_TO_DROP | PRESERVATION_COLUMNS | component_columns),
    errors="ignore",
)
```

Verify label distribution: `data.groupby(["region", "label"]).size()`.

### Step 3 — Build preprocessing pipeline and PU classifier

```python
from pulearn.bagging import BaggingPuClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import RobustScaler
from lib.pu import PU_PARAMS

rf_params = {"random_state": random_seed, "n_estimators": 50, "n_jobs": 1}
pu_params = {
    **PU_PARAMS,          # n_estimators=50, max_samples=1.0
    "n_jobs": n_jobs,
    "random_state": random_seed,
    "balanced_subsample": True,
}
preprocessing = make_pipeline(
    IterativeImputer(random_state=random_seed, add_indicator=False),
    RobustScaler(),
    VarianceThreshold(),
)
pu_model = BaggingPuClassifier(RandomForestClassifier(**rf_params), **pu_params)
pu_pipeline = Pipeline([("preprocessing", preprocessing), ("classifier", pu_model)])
pu_pipeline.set_output(transform="pandas")
```

### Step 4 — Feature selection via Spearman correlation dendrogram

Use Spearman's |r| ≥ 0.8 as the correlation cutoff (distance threshold = 0.2).

```python
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.stats import spearmanr
from sklearn.base import clone
from lib.misc import format_feature_name

cutoff_value = 0.8
cluster_threshold = 1 - cutoff_value

# PU training slice: positive + unlabelled only
train_pu = cleaned[cleaned["label"].isin({"positive", "unlabelled"})]
x_pu = train_pu.drop(columns="label")
y_pu = train_pu["label"].replace({"positive": 1, "unlabelled": 0}).astype("int")

Z = linkage(
    clone(preprocessing).fit_transform(x_pu).T,
    method="single",
    metric=lambda x, y: 1.0 - np.abs(spearmanr(x, y).statistic),
)
cluster_ids = fcluster(Z, cluster_threshold, criterion="distance")
clusters = {
    cluster_id: list(x_pu.columns[cluster_ids == cluster_id])
    for cluster_id in np.unique(cluster_ids)
}
# Select one representative per cluster (first member)
selected_features = [clusters[cid][0] for cid in sorted(np.unique(cluster_ids))]

# Save selection
from pathlib import Path
pd.DataFrame({"selected_features": selected_features}).to_csv(
    output_dir / "selected_features.csv", index=False
)
```

Visualize with:
```python
figures_dir = output_dir / "feature_selection"
figures_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, max(4, len(x_pu.columns) * 0.25)))
dendro = dendrogram(
    Z, orientation="right",
    labels=[format_feature_name(i, bold=True) for i in x_pu.columns],
    ax=ax, color_threshold=cluster_threshold,
)
ax.axvline(cluster_threshold, linestyle="dashed", color="red",
           label=f"Cutoff ({cutoff_value}) → {len(clusters)} features")
ax.set_xticks(ax.get_xticks(), [f"{1-i:.1f}" for i in ax.get_xticks()])
ax.set_xlabel(r"Spearman's $|r|$", fontsize=14)
fig.savefig(figures_dir / "dendrogram.pdf", dpi=350, bbox_inches="tight")
```

Verify: `len(selected_features)` should be substantially fewer than the original column count.

### Step 5 — Train and save global model

```python
import joblib

pu_dir = output_dir / "PU"
pu_dir.mkdir(parents=True, exist_ok=True)

pu_pipeline.fit(x_pu[selected_features], y_pu)
joblib.dump(pu_pipeline, pu_dir / "classifier.joblib", compress=True)
```

This step uses `x_pu` and `selected_features` from Steps 2–4.

### Step 6 — Region-aware cross-validation with `lib/cv.py`

```python
from lib.cv import perform_cv
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=random_seed)

# data must contain 'region' column; CV trains on {positive, unlabelled},
# tests on {positive, negative}.
cv_results = perform_cv(
    clf=clone(pu_pipeline),   # untrained clone
    data=cleaned[cleaned.columns],  # full cleaned df with 'label' and 'region'
    cv=cv,
    thresh=0.5,
    pu=True,               # train filter: {positive, unlabelled}
    separate_regions=True, # report metrics per-region
    random_state=random_seed,
    verbose=True,
)
print(cv_results.groupby("region")[["roc_auc", "average_precision", "balanced_accuracy"]].mean())
```

`perform_cv` returns a DataFrame with columns: `roc_auc`, `average_precision`, `balanced_accuracy`, `accuracy`, `f1`, `n_train`, `n_test`, `time_fit`, `time_predict`, `region`.

### Step 7 — Feature importance (Gini, cross-validated)

```python
model = joblib.load(pu_dir / "classifier.joblib")
feature_names = model.feature_names_in_
replace = {"positive": 1, "negative": 0, "unlabelled": 0}

importances = []
for train_idx, _ in cv.split(cleaned, cleaned["label"]):
    data_train = cleaned.iloc[train_idx]
    data_train = data_train[data_train["label"] != "negative"]
    X_train = data_train[feature_names]
    y_train = data_train["label"].replace(replace).infer_objects()
    clf = clone(model)
    clf.fit(X_train, y_train)
    importances.append(pd.DataFrame(
        np.vstack([e.feature_importances_ for e in clf["classifier"].estimators_]),
        columns=clf["preprocessing"].get_feature_names_out(),
    ))
importances = pd.concat(importances, ignore_index=True)
importances = importances[importances.median().sort_values(ascending=False).index]
importances.to_csv(pu_dir / "feature_importance" / "feature_importance.csv", index=False)
```

### Step 8 — Partial dependence plots (notebook 07)

```python
# In 07-partial_dependence.ipynb — do not call directly from 01
from lib.partial_dependence import make_plot
make_plot(
    classifier_filename=str(pu_dir / "classifier.joblib"),
    data_filename=str(training_data_filepath),
    output_basename=str(output_dir / "partial_dependence"),
    n_jobs=n_jobs,
    verbose=True,
)
```

## Examples

**User says:** "Train the PU classifier with mantle features and cross-validate by region."

**Actions taken:**
1. Load config via `PathConfigManager("config/.run_config.yml", notebook="01")`.
2. Read `pcm.TRAINING_DATA_PATH`, deduplicate on `["lon", "lat", "age (Ma)"]`, skip regions < 50 positives.
3. Drop `COLUMNS_TO_DROP | PRESERVATION_COLUMNS | component_columns`.
4. Build `Pipeline([preprocessing, BaggingPuClassifier(RandomForestClassifier(...))])`.
5. Slice `train_pu = cleaned[cleaned["label"].isin({"positive", "unlabelled"})]`, build `x_pu`, `y_pu`.
6. Run Spearman linkage on preprocessed `x_pu.T`, call `fcluster(Z, 0.2)`, pick `clusters[cid][0]` per cluster → `selected_features`.
7. Fit `pu_pipeline` on `x_pu[selected_features], y_pu`; dump to `output_dir/PU/classifier.joblib`.
8. Call `perform_cv(clf=clone(pu_pipeline), data=cleaned, separate_regions=True)`; print per-region AUC.

**Result:** `output_dir/PU/classifier.joblib`, `output_dir/selected_features.csv`, `output_dir/feature_selection/dendrogram.{pdf,png}`.

## Common Issues

**`KeyError: 'label'` in `get_xy()`:** The DataFrame passed has no `label` column. Check you are passing the raw training CSV, not the already-split `x_pu`.

**`ValueError: Input contains NaN` during `BaggingPuClassifier.fit`:** `IterativeImputer` sits before the classifier in the pipeline but was bypassed by calling `clf.fit(x, y)` directly on the base estimator. Always call `pipeline.fit(x, y)`, never `pipeline["classifier"].fit(x, y)`.

**`AttributeError: 'Pipeline' object has no attribute 'feature_importances_'`:** Feature importances live inside the nested estimator. Access them as `clf["classifier"].estimators_[i].feature_importances_` (see Step 7 pattern).

**Region missing from CV results:** That region had < 50 positive labels and was filtered in Step 2. Verify with `data.groupby(["region", "label"]).size()`.

**`perform_cv` raises `KeyError: 'region'`:** `data` passed to `perform_cv` must retain the `region` column from the training CSV. Do not strip it when building `cleaned`; `get_xy()` drops it internally via `COLUMNS_TO_DROP`.

**Linkage step very slow:** `x_pu` has too many rows. The linkage operates on features (columns), not samples — pass `x_pu.T` as the input matrix, not `x_pu`.

**`joblib.dump` produces 0-byte file:** `compress=True` requires the output directory to exist. Call `(pu_dir / "feature_importance").mkdir(parents=True, exist_ok=True)` before dumping.