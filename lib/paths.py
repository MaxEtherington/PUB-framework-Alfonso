from pathlib import Path
from load_params import get_params

# =============== Load config =============== 

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / 'config' / '.run_config.yml'
config = get_params(CONFIG_PATH)

# =============== Data paths ===============

PREPARED_DATA_DIR = ROOT / 'data_prepared'

SOURCE_DATA_DIR = ROOT / 'data_source'
MANTLE_DATA_DIR = SOURCE_DATA_DIR / 'mantle_outputs' / config['mantle_features']['mantle_dir']
DEPOSITS_PATH = SOURCE_DATA_DIR / "deposits" / config['deposits_filename']
REGIONS_PATH = SOURCE_DATA_DIR / "regions" / config['regions_filename']

EXTRACTED_DATA_DIR = ROOT / 'data_extracted' / config['plate_model']['plate_model_name']
PLATE_MODEL_DIR = EXTRACTED_DATA_DIR / 'plate_models' / config['plate_model']['plate_model_name']
RASTER_DATA_DIR = EXTRACTED_DATA_DIR / 'rasters'
POINTS_DATA_DIR = EXTRACTED_DATA_DIR / 'polygons_points' / f"{config['reference_feature']}_{config['study_zone_buffer']}_deg_buffer"

OUTPUT_DIR = ROOT / 'output' / config['run_name']

# =============== Initialise directories ===============

for path in [EXTRACTED_DATA_DIR, PLATE_MODEL_DIR, RASTER_DATA_DIR, POINTS_DATA_DIR, OUTPUT_DIR]:
    path.mkdir(parents=True, exist_ok=True)