import sys

from .paths import PathConfigManager
from .plate_models import cache_plate_model
from .check_files import (
    check_plate_model,
    check_prepared_data,
    check_erodep_data,
    check_paleotopography_data
)


def _cache_pmm_plate_model(p: PathConfigManager):
    """Cache a plate model from the EarthByte server using PlateModelManager."""
    cache_plate_model(
        model_name=p.config['plate_model']['plate_model_name'],
        model_dir =p.PLATE_MODEL_DIR
    )


def _cache_zenodo_plate_model(p: PathConfigManager):
    """Cache plate model from Alfonso 2024 et al. Zenodo repository."""
    check_plate_model(
        model_dir=p.PLATE_MODEL_DIR,
        verbose  =p.config['verbose'],
        force    =p.config['overwrite_output']
    )


def _cache_prepared_training_grid_data(p: PathConfigManager):
    """Cache prepared training and grid data from Alfonso 2024 et al. Zenodo repository."""
    check_prepared_data(
        data_dir=p.PREPARED_DATA_DIR,
        verbose =p.config['verbose'],
        force   =p.config['overwrite_output']
    )
    

def _cache_erodep_data(p: PathConfigManager):
    """Cache erosion and deposition data bundle from Alfonso 2024 et al. Zenodo repository."""
    check_erodep_data(
        data_dir=p.RASTER_DATA_DIR / "erodep_outputs",
        verbose =p.config['verbose'],
        force   =p.config['overwrite_output']
    )
    
    
def _cache_paleotopography_data(p: PathConfigManager):
    """Cache palaeotopography data from EarthByte server."""
    check_paleotopography_data(
        data_dir=p.RASTER_DATA_DIR / "crust_outputs" / "paleotopography",
        verbose =p.config['verbose'],
        force   =p.config['overwrite_output']
    )


def cache_remote_data(p: PathConfigManager):
    """Cache remotely-hosted data (e.g. plate models), for when 
    notebooks are run on a machine without internet access (e.g. HPC)."""
    
    # Cache prepared data bundle (training and grid data)
    if not p.use_extracted_data:
        _cache_prepared_training_grid_data(p)
        print("\nPrepared data:   prepared data bundle ready", file=sys.stderr)
    
    # Cache plate model
    if p.use_provided_plate_model:
        _cache_zenodo_plate_model(p)
        print("Plate model:     provided model ready\n", file=sys.stderr)
    elif p.plate_model_name == "custom_plate_model":
        print("Plate model:     custom plate model specified; skipping caching"
              f" (ensure model files are present in '{p.PLATE_MODEL_DIR.relative_to(p.ROOT)}')\n", file=sys.stderr)
    else:
        _cache_pmm_plate_model(p)
        print(f"Plate model:     PlateModel '{p.plate_model_name}' ready\n", file=sys.stderr)

    # Cache erodep data
    if p.use_erodep:
        _cache_erodep_data(p)
        print("Erodep data:     erosion/deposition data ready\n", file=sys.stderr)
    
    # Cache palaeotopography data
    if p.use_crustal_features:
        _cache_paleotopography_data(p)
        print("Palaeotopography data:     palaeotopography data ready\n", file=sys.stderr)

    print("Setup complete.\n", file=sys.stderr)