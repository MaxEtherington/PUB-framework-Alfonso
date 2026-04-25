# Agent Routing — Task to File Reference

Use this table before starting any task to identify which files to consult. Always check the relevant GitHub issue first for prerequisites and blockers.

| Task | Files to read first |
|---|---|
| Config or path logic | `notebook_parameters_default.yml`, `lib/load_params.py`, `docs/data-layout.md` |
| Plate reconstruction, model loading | `lib/plate_models.py`, `lib/misc.py`, `docs/reconstruction.md` |
| Raster generation (`00a`) | `00a-generate_data.ipynb`, `lib/extract_data/`, `docs/reconstruction.md` |
| Training data extraction (`00b/c`) | `00b-extract_training_data.ipynb`, `lib/calculate_convergence.py`, `lib/coregister_ocean_rasters.py`, `lib/combine_point_data.py` |
| Mantle feature extraction (`00d`) | `mantle-processing/point-samping.ipynb`, `docs/mantle-extraction.md` |
| PU classifier, training | `lib/pu.py`, `01-create_classifiers.ipynb` |
| Cross-validation | `lib/cv.py`, `01-create_classifiers.ipynb` |
| Feature selection / correlation | `lib/feature_selection.py`, `01-create_classifiers.ipynb` |
| Study area polygons, unlabelled points | `lib/create_study_area_polygons.py`, `lib/generate_unlabelled_points.py` |
| Deposit database, schema | `cu-deposits-preprocessing/src/deposit_data_preprocessing/`, `docs/data-layout.md` |
| Bayesian hyperparameter tuning (future) | `PUB-framework-Ehsan/MPM_Porphyry_NSW.ipynb`, `PUB-framework-Ehsan/lib_mpm.py` |
| Seafloor anomaly features (reference) | `Mather2025-SeafloorAnomalies/seafloor-anomalies/07-Input-to-PU-learn.ipynb` |
| Probability map generation | `02-create_probability_maps.ipynb`, `lib/pu.py` (`create_probability_grids`) |
