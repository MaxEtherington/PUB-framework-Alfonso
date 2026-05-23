from pathlib import Path

from .load_params import get_params

class PathConfigManager():
    """Lightweight class to hold and manage project config and config-derived paths between notebooks."""

    # Invariant paths
    ROOT            = Path(__file__).resolve().parent.parent
    CONFIG_DIR      = ROOT / 'config'
    RUN_CONFIG_PATH = CONFIG_DIR / '.run_config.yml'

    def __init__(
        self,
        config_path,
        notebook: str = None
    ):
        self.CONFIG_PATH = Path(config_path).resolve()
        self.notebook = notebook

        try:
            self.update_paths(config_path=config_path, notebook=notebook)
        except KeyError as e:
            raise KeyError(f"Missing required config key: {e}") from e


    def update_paths(self, config_path=None, notebook=None):
        """Update exposed attributes (paths, config) based on the config file."""
        # Validate
        if config_path is not None:
            self.CONFIG_PATH = Path(config_path).resolve()
        if notebook is not None:
            self.notebook = notebook

        # Load config
        try:
            config = get_params(self.CONFIG_PATH, self.notebook)
        except Exception as e:
            message_notebook = f" for notebook '{self.notebook}'" if self.notebook else ""
            raise Exception(f"Error loading config from {self.CONFIG_PATH}{message_notebook}: {e}") from e
        self.config = config

        # Expose important config values as attributes
        self.plate_model_name = (
            "alfonso2024_default" if config['plate_model']['use_provided_plate_model']
            else config['plate_model']['plate_model_name'] or "custom_plate_model"
        )
        self.use_provided_plate_model = config['plate_model']['use_provided_plate_model']
        self.use_extracted_data = config['use_extracted_data']

        # Derive paths from config
        self.PREPARED_DATA_DIR = self.ROOT / 'data_prepared'

        self.SOURCE_DATA_DIR = self.ROOT / 'data_source'
        self.PLATE_MODEL_DIR = self.SOURCE_DATA_DIR / 'plate_models' / self.plate_model_name
        self.DEPOSITS_PATH = self.SOURCE_DATA_DIR / 'deposits' / config['deposits_filename']
        self.REGIONS_PATH = self.SOURCE_DATA_DIR / 'regions' / config['regions_filename']
        self.MANTLE_DATA_DIR = self.SOURCE_DATA_DIR / 'mantle_outputs' / config['feature_sets']['mantle']['data_dir']

        self.EXTRACTED_DATA_DIR = self.ROOT / 'data_extracted' / self.plate_model_name
        self.RASTER_DATA_DIR = self.EXTRACTED_DATA_DIR / 'rasters'
        self.POINTS_DATA_DIR = self.EXTRACTED_DATA_DIR / 'polygons_points' / f"{config['reference_feature']}_{config['study_zone_buffer']}_deg_buffer"

        # Training data and associated filepaths
        active_source_dir = self.POINTS_DATA_DIR if self.use_extracted_data else self.PREPARED_DATA_DIR
        self.TRAINING_DATA_PATH = active_source_dir / 'training_data_global.csv'
        self.GRID_DATA_PATH = active_source_dir / 'grid_data.csv'
        self.FEATURES_MANIFEST_PATH = active_source_dir / 'features_manifest.json'

        # Output figure directories and data files
        self.OUTPUT_DIR = self.ROOT / 'output' / config['run_name']

        # Output subdirectories (flat hierarchy; PU dir dissolved)
        self.FEATURE_SELECTION_DIR   = self.OUTPUT_DIR / 'feature_selection'
        self.CROSS_VALIDATION_DIR    = self.OUTPUT_DIR / 'cross_validation'
        self.FEATURE_IMPORTANCE_DIR  = self.OUTPUT_DIR / 'feature_importance'
        self.AREA_RECALL_DIR         = self.OUTPUT_DIR / 'area_recall'
        self.PARTIAL_DEPENDENCE_DIR  = self.OUTPUT_DIR / 'partial_dependence'

        # Probability grid outputs grouping
        self.PROBABILITY_OUTPUT_DIR  = self.OUTPUT_DIR / 'probability_grid_outputs'
        self.GRID_PROBABILITIES_PATH = self.PROBABILITY_OUTPUT_DIR / 'grid_probabilities.csv'
        self.PROBABILITY_GRIDS_DIR   = self.PROBABILITY_OUTPUT_DIR / 'probability_grids'

        # Classifier base path; regional variants derived via .with_name(f"classifier_{r}.joblib")
        self.CLASSIFIER_PATH         = self.OUTPUT_DIR / 'classifier.joblib'

        # Selected features; regional variants via .with_name(f"selected_features_{r}.csv")
        self.SELECTED_FEATURES_PATH  = self.FEATURE_SELECTION_DIR / 'selected_features.csv'
        self.SELECTED_FEATURES_MANIFEST_PATH  = self.FEATURE_SELECTION_DIR / 'selected_features_manifest.json'

        # Comparison-data output paths (written during cross-validation and area-recall)
        self.CV_AP_SCORES_PATH       = self.CROSS_VALIDATION_DIR / 'cv_ap_scores.csv'
        self.AREA_RECALL_DATA_PATH   = self.AREA_RECALL_DIR / 'area_recall_data.csv'

        # Create active feature set list, paths, filenames
        # Feature sets group related features by source data; each can be enabled/disabled in the config
        # Some feature sets are nested, e.g. carbonate-related features rely on subduction kinematics
        def find_active_feature_sets(mapping, parent_key="", parent_enabled=True):
            for key, val in mapping.items():
                if not isinstance(val, dict):
                    continue
                full_key = f"{parent_key}.{key}" if parent_key else key
                enabled = parent_enabled and val.get('enabled', False)
                if enabled:
                    yield full_key
                yield from find_active_feature_sets(val, parent_key=full_key, parent_enabled=enabled)

        self.active_feature_sets = set(find_active_feature_sets(config['feature_sets']))


    def status(self) -> dict[str, bool]:
        """Print pipeline output status and return a dict of {label: present}."""
        checkpoints = [
            ("00a", "Plate model",        self.PLATE_MODEL_DIR),
            ("00b", "Training data",      self.TRAINING_DATA_PATH),
            ("00c", "Grid data",          self.GRID_DATA_PATH),
            ("01a", "Selected features",  self.SELECTED_FEATURES_PATH),
            ("01b", "Classifier",         self.CLASSIFIER_PATH),
            ("02a", "Grid probabilities", self.GRID_PROBABILITIES_PATH),
            ("02a", "Probability grids",  self.PROBABILITY_GRIDS_DIR),
            ("02b", "Area recall data",   self.AREA_RECALL_DATA_PATH),
            ("07",  "Partial dependence", self.PARTIAL_DEPENDENCE_DIR),
        ]
        results = {}
        print(f"Pipeline status — run: {self.config['run_name']}")
        for stage, label, path in checkpoints:
            present = path.is_file() or (path.is_dir() and any(path.iterdir()))
            results[label] = present
            mark = "✓" if present else "✗"
            print(f"  [{stage}] {label:<22} {mark}  {path.relative_to(self.ROOT)}")
        n = sum(results.values())
        print(f"  {n} / {len(results)} outputs present")
        return results

    def use_features(self, feature_set: str) -> bool:
        return feature_set in self.active_feature_sets


    def create_directories(self):
        """Create all necessary directories based on current config."""
        config = get_params(self.CONFIG_PATH)

        for path in [
            self.PLATE_MODEL_DIR,
            self.OUTPUT_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        if config['use_extracted_data']:
            for path in [self.EXTRACTED_DATA_DIR, self.RASTER_DATA_DIR, self.POINTS_DATA_DIR]:
                path.mkdir(parents=True, exist_ok=True)


    def validate_input_paths(self):
        """Raise an error if expected source data paths do not exist."""
        config = get_params(self.CONFIG_PATH)
        missing_paths = []

        if config['use_extracted_data']:
            for path in [self.DEPOSITS_PATH, self.REGIONS_PATH]:
                if not path.exists():
                    missing_paths.append(path)

        if self.use_features('mantle'):
            for path in [self.MANTLE_DATA_DIR]:
                if not path.exists():
                    missing_paths.append(path)

        if missing_paths:
            raise FileNotFoundError("Expected source files not found:" + "".join(f"\n  - {path.relative_to(self.ROOT)}" for path in missing_paths))
