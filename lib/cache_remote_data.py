import sys

from .paths import PathConfigManager


def _cache_pmm_plate_model(p: PathConfigManager):
    """Cache a plate model from the EarthByte server using PlateModelManager."""
    from .plate_models import cache_plate_model
    cache_plate_model(
        model_name=p.config['plate_model']['plate_model_name'],
        model_dir=p.PLATE_MODEL_DIR
    )


def _cache_zenodo_plate_model(p: PathConfigManager):
    """Cache plate model from Alfonso 2024 et al. Zenodo repository."""
    from .check_files import check_plate_model
    check_plate_model(
        model_dir=p.PLATE_MODEL_DIR,
        verbose  =p.config['verbose'],
        force    =p.config['overwrite_output'],
    )


def _cache_prepared_training_grid_data(p: PathConfigManager):
    """Cache prepared training and grid data from Alfonso 2024 et al. Zenodo repository."""
    from .check_files import check_prepared_data
    check_prepared_data(
        data_dir=p.PREPARED_DATA_DIR,
        verbose=p.config['verbose'],
        force=p.config['overwrite_output']
    )
    

def _cache_erodep_data(p: PathConfigManager):
    """Cache erosion and deposition data bundle from Alfonso 2024 et al. Zenodo repository."""
    from .check_files import check_erodep_data
    check_erodep_data(
        data_dir=p.RASTER_DATA_DIR / "erodep_outputs",
        verbose=p.config['verbose'],
        force=p.config['overwrite_output']
    )
    
    
def _cache_palaeotopography_data(p: PathConfigManager):
    """Cache palaeotopography data from EarthByte server."""
    from .check_files import check_palaeotopography_data
    check_palaeotopography_data(
        data_dir=p.RASTER_DATA_DIR / "crust_outputs" / "paleotopography",
        verbose=p.config['verbose'],
        force=p.config['overwrite_output']
    )


def cache_remote_data(p: PathConfigManager):
    """Cache remotely-hosted data (e.g. plate models), for when 
    notebooks are run on a machine without internet access (e.g. HPC)."""
    
    config = p.config
    
    # Cache prepared data bundle (training and grid data)
    if not config['use_extracted_data']:
        _cache_prepared_training_grid_data(p)
        print("Prepared data:   prepared data bundle ready", file=sys.stderr)
    
    # Cache plate model
    if config['use_provided_plate_model']:
        _cache_zenodo_plate_model(p)
        print("Plate model:     provided model ready", file=sys.stderr)
    else:
        _cache_pmm_plate_model(p)
        print(f"Plate model:     PlateModel '{config['plate_model_name']}' ready", file=sys.stderr)

    # Cache erodep data
    if config['use_erodep']:
        _cache_erodep_data(p)
        print("Erodep data:     erosion/deposition data ready", file=sys.stderr)
    
    # Cache palaeotopography data
    if config['use_crustal_features']:
        _cache_palaeotopography_data(p)
        print("Palaeotopography data:     palaeotopography data ready", file=sys.stderr)

    print("Setup complete.", file=sys.stderr)
