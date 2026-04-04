from pathlib import Path

from .load_params import get_params

class PathConfigManager():
    """Lightweight class to hold and manage project config and config-derived paths."""
    
    # Invariant paths
    ROOT            = Path(__file__).resolve().parent.parent
    CONFIG_DIR      = ROOT / 'config'
    RUN_CONFIG_PATH = CONFIG_DIR / '.run_config.yml'
    
    def __init__(
        self, 
        config_path, 
        notebook: str = None
    ):
        # 
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
        self.TRAINING_DATA_PATH = self.EXTRACTED_DATA_DIR / 'training_data_global.csv'
        self.GRID_DATA_PATH = self.EXTRACTED_DATA_DIR / 'grid_data.csv'

        self.OUTPUT_DIR = self.ROOT / 'output' / config['run_name']
        
        # Active feature sets (e.g. subduction, crustal, mantle) are determined by config and exposed as attributes
        # Create active feature set list, paths, filenames
        self.active_feature_sets = [
            feature_set for feature_set, params in config['feature_sets'].items() if params['enabled']
        ]
        
        self.feature_filepaths = {
            name: {
                'training': self.POINTS_DATA_DIR / f"{name}_features_training.csv",
                'grid': self.POINTS_DATA_DIR / f"{name}_features_grid.csv",
            }
            for name in self.active_feature_sets
        }
        
        self.use_subduction_features = 'subduction' in self.active_feature_sets
        self.use_crustal_features = 'crustal' in self.active_feature_sets
        self.use_mantle_features = 'mantle' in self.active_feature_sets
        self.use_erodep = 'erodep' in self.active_feature_sets
    
    
    def create_directories(self):
        """Create all necessary directories based on current config."""
        config = get_params(self.CONFIG_PATH)
    
        for path in [self.PLATE_MODEL_DIR, self.OUTPUT_DIR]:
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
        
        if self.use_mantle_features:
            for path in [self.MANTLE_DATA_DIR]:
                if not path.exists():
                    missing_paths.append(path)
        
        if missing_paths:
            raise FileNotFoundError("Expected source files not found:" + "".join(f"\n  - {path.relative_to(self.ROOT)}" for path in missing_paths))