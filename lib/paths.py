from pathlib import Path
from load_params import get_params

class PathConfigManager():
    """Lightweight class to hold and manage project config and config-derived paths."""
    
    # Invariant paths
    ROOT            = Path(__file__).resolve().parent.parent
    CONFIG_DIR      = ROOT / 'config'
    RUN_CONFIG_PATH = CONFIG_DIR / '.run_config.yml'
    
    def __init__(self, config_path, notebook=None):    
        self.CONFIG_PATH = Path(config_path).resolve()
        self.notebook = notebook
        
        if not self.CONFIG_PATH.is_file():
            raise FileNotFoundError(f"Config file not found: {self.CONFIG_PATH}")
        
        try:
            self.update_paths()
        except KeyError as e:
            raise KeyError(f"Missing required config key: {e}") from e
    
    
    def update_paths(self):
        """Update all paths based on the current config file."""
        config = get_params(self.CONFIG_PATH, self.notebook)
        
        plate_model_name = (
            "alfonso2024_default" if config['plate_model']['use_provided_plate_model']
            else config['plate_model']['plate_model_name'] or "custom_plate_model"
        )
        
        self.PREPARED_DATA_DIR = self.ROOT / 'data_prepared'

        self.SOURCE_DATA_DIR = self.ROOT / 'data_source'
        self.PLATE_MODEL_DIR = self.SOURCE_DATA_DIR / 'plate_models' / plate_model_name
        self.DEPOSITS_PATH = self.SOURCE_DATA_DIR / 'deposits' / config['deposits_filename']
        self.REGIONS_PATH = self.SOURCE_DATA_DIR / 'regions' / config['regions_filename']
        self.MANTLE_DATA_DIR = self.SOURCE_DATA_DIR / 'mantle_outputs' / config['mantle']['mantle_dir']
    
            
        self.EXTRACTED_DATA_DIR = self.ROOT / 'data_extracted' / plate_model_name
        self.RASTER_DATA_DIR = self.EXTRACTED_DATA_DIR / 'rasters'
        self.POINTS_DATA_DIR = self.EXTRACTED_DATA_DIR / 'polygons_points' / f"{config['reference_feature']}_{config['study_zone_buffer']}_deg_buffer"

        self.OUTPUT_DIR = self.ROOT / 'output' / config['run_name']
        
        self.config = config
    
    
    def create_directories(self):
        """Create all necessary directories based on current config."""
        config = get_params(self.CONFIG_PATH)
    
        for path in [self.OUTPUT_DIR]:
            path.mkdir(parents=True, exist_ok=True)
            
        if config['use_extracted_data']:
            for path in [self.EXTRACTED_DATA_DIR, self.PLATE_MODEL_DIR, self.RASTER_DATA_DIR, self.POINTS_DATA_DIR]:
                path.mkdir(parents=True, exist_ok=True)
    
    
    def validate_input_paths(self):
        """Raise an error if any expected paths do not exist."""
        config = get_params(self.CONFIG_PATH)
        missing_paths = []

        if config['use_extracted_data']:
            for path in [self.DEPOSITS_PATH, self.REGIONS_PATH]:
                if not path.exists():
                    missing_paths.append(path)
        
        if config['mantle_features']['use_mantle_features']:
            for path in [self.MANTLE_DATA_DIR]:
                if not path.exists():
                    missing_paths.append(path)
        
        if missing_paths:
            raise FileNotFoundError(f"Expected files not found: {[f'\n  - {path}' for path in missing_paths]}")