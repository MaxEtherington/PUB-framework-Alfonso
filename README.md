## M Etherington Honours Project: Mantle-Coupled Porphyry Prospectivity Data Mining Scripts

This repository contains the Python scripts and notebooks required to extract the data, train the models, and produce the figures for Max Etherington's Research School of Earth Sciences ANU honours project "Reconstructing the 4-D geodynamic blueprint of copper porphyry formation". It creates a spatiotemporal positive-unlabelled bagging (PUB) classifier for Cu deposit prospectivity mapping. The repository is a heavily modified version of the workflow from Christopher Alfonso's 2024 paper "Spatio-temporal copper prospectivity in the American Cordillera predicted by positive-unlabelled machine learning".

Training data can be extracted using the `00b-extract_training_data.ipynb` and `00c-extract_grid_data.ipynb` notebooks.

`00b` does the following:
- Co-registers a deposit database with a plate reconstruction
- Randomly generates unlabelled points 
- Extracts features for positively labelled deposit points and randomly generated unlabelled points from the plate model, G-ADOPT mantle flow outputs, and other input datasets
- Assigns regions to these points using a 'regions' polygon file.

Some features require special grid data that is sourced externally. Notebook `00a-generate_data.ipynb` performs this function. Note that these externally sourced features are unlikely to be compatible with most plate reconstructions. 

The workflow is designed in a modular fashion. The parameters of any classifier training run can be configured using a config file in `/config`. This enables the user to determine which features are extracted by the notebooks, which plate reconstruction to use, which set of mantle outputs to use, as well as other more fine-grained controls for these workflows.

### To run the notebooks:

1. Create a `conda` environment using the `environment.yml` file: `conda env create --file environment.yml`
2. Run the following notebooks to download and extract training data from Zenodo/your local device(optional):
    - `00a-generate_data.ipynb`
    - `00b-extract_training_data.ipynb`
    - `00c-extract_grid_data.ipynb`
3. Run these notebooks to train a PU classifier and create prospectivity maps and other plots:
    - `01a-select_features.ipynb`
    - `01b-train_validate_classifiers.ipynb`
    - `02-create_probability_maps.ipynb`
    - `03-create_probability_animations.ipynb`
    - `04-create_erosion_distribution.ipynb`
    - `05-create_preservation_maps.ipynb`
    - `06-create_preservation_animations.ipynb`
    - `07-partial_dependence.ipynb`
    - `08-time_series.ipynb`
