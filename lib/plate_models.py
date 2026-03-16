import glob
import os
from pathlib import Path
from tempfile import NamedTemporaryFile, _TemporaryFileWrapper
from typing import Optional

from gplately import (
    PlateReconstruction,
    PlotTopologies,
    PlateModelManager,
    PlateModel
)

from .check_files import check_plate_model  # re-export  # noqa: F401
from .misc import filter_topological_features

def cache_plate_model(model_name: str, model_dir: str) -> None:
    """Use PlateModelManager to cache plate model files in provided directory"""
    pmm = PlateModelManager()
    model = pmm.get_model(model_name=model_name, data_dir=model_dir)
    layers = [
        'StaticPolygons', 
        'Coastlines', 
        'ContinentalPolygons', 
        'Topologies', 
        'COBs'
    ]
    for layer in layers:
        model.get_layer(layer_name=layer, return_none_if_not_exist=True)
    model.get_rotation_model()
    return

def _fetch_plate_model(model_name, model_dir):
    try:
        pmm = PlateModelManager()
        model = pmm.get_model(model_name=model_name, data_dir=model_dir)
    except Exception as exc:
        # If PlateModelManager fetch fails but local model files exist, fall back to local files.
        model = PlateModel(
                model_name=model_name,
                data_dir=model_dir,
                model_cfg=Path(model_dir) / ".metadata.json",
                readonly=True,
            )
        if not model.is_model_dir(Path(model_dir) / model_name):
            raise exc
    return model

def has_plate_model_files(model_dir: Path | str) -> bool:
    """Return True if the directory appears to contain a usable plate model."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        return False
    has_rotations = any(model_dir.rglob("*.rot"))
    has_features = any(model_dir.rglob("*.gpml")) or any(model_dir.rglob("*.gpmlz"))
    return has_rotations and has_features

def get_plate_reconstruction(
    model_name: Optional[str] = None,
    model_dir: str = "plate_model",
    anchor_plate_id: int = 0,
    filter_topologies: bool = False,
) -> PlateReconstruction | tuple[PlateReconstruction, _TemporaryFileWrapper]:
    """Get a `PlateReconstruction` object from a model name and directory.

    If `model_name` is `None` use local files in `model_dir`.

    Parameters
    ----------
    model_name : str, optional
        Name of the model to fetch from PMM.
    model_dir : str, default: 'plate_model'
        Directory to store files.
    anchor_plate_id : int, default: 0
        Anchor plate ID of the plate model.
    filter_topologies : bool, default: False
        Remove inactive deforming networks and flat slab topologies from
        the model. Requires writing to a temporary file.

    Returns
    -------
    PlateReconstruction
        The output plate model.
    tf : tempfile._TemporaryFileWrapper, optional
        If `filter_topologies` is True, a handle to the temporary
        file containing the topologies will also be returned, to prevent
        it from being cleaned up by the garbage collector and the file
        potentially being deleted.

    Raises
    ------
    Exception
        Propagates exceptions from PMM fetch when no valid local fallback
        model directory is available.
    """
    def _find_model_files():
        """Search local storage to retrieve custom plate model"""
        globs = ["*.gpml", "*.gpmlz"]
        rotation_files = []
        topology_files = []
        static_polygons = []
        for g in globs:
            all_filenames = glob.glob(os.path.join(model_dir, "**", g), recursive=True)
            topology_files.extend(all_filenames)
            # topology_files.extend(filenames)
            # rotation_files.extend(filenames)
            static_polygons.extend(
                [
                    i for i in all_filenames
                    if "static" in os.path.basename(i).lower()
                    and "polygon" in os.path.basename(i).lower()
                ]
            )
        rotation_files.extend(
            glob.glob(os.path.join(model_dir, "**", "*.rot"), recursive=True)
        )
        return rotation_files, topology_files, static_polygons
    
    model = None
    if model_name is None: # Alfonso2024 provided reconstruction
        rotation_files, topology_files, static_polygons = _find_model_files()
    else:
        model = _fetch_plate_model(model_name, model_dir)
        # Collect files
        rotation_files = model.get_rotation_model()
        topology_files = model.get_topologies()
        static_polygons = model.get_static_polygons()
    
    if filter_topologies:
        topology_features = filter_topological_features(topology_files)
        tf = NamedTemporaryFile(suffix=".gpml")
        topology_features.write(tf.name)
        topology_files = [tf.name]

    plate_reconstruction = PlateReconstruction(
        rotation_model=rotation_files,
        topology_features=topology_files,
        static_polygons=static_polygons,
        anchor_plate_id=anchor_plate_id,
        plate_model=model
    )
    if filter_topologies:
        return plate_reconstruction, tf
    return plate_reconstruction



def get_plot_topologies(
    model_name: Optional[str] = None,
    model_dir: str = "plate_model",
    anchor_plate_id: int = 0,
    time: int = 0,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    filter_topologies: bool = False,
) -> PlotTopologies:
    """Get a `PlotTopologies` object from a model name and directory.

    If `model_name` is `None`, use local files in `model_dir`.

    Parameters
    ----------
    model_name : str, optional
        Name of the model to fetch from PMM.
    model_dir : str, default: 'plate_model'
        Directory to store files.
    anchor_plate_id : int, default: 0
        Anchor plate ID of the plate model.
    time : int, optional
        If provided, set the `PlotTopologies` object to this time.
    plate_reconstruction: PlateReconstruction, optional
        If provided, use this `PlateReconstruction` instead of fetching
        all files from PMM.
    filter_topologies : bool, default: False
        Remove inactive deforming networks and flat slab topologies from
        the model. Requires writing to a temporary file.

    Returns
    -------
    `PlotTopologies`

    Raises
    ------
    Exception
        Propagates exceptions from plate reconstruction/model retrieval if
        model data cannot be resolved from PMM or local fallback.
    """
    def _find_plate_topologies():
        """Search local storage to retrieve custom plate model topologies"""
        file_exts = ["*.gpml", "*.gpmlz"]
        coastlines = []
        continent_candidates = []
        for g in file_exts:
            filenames = glob.glob(os.path.join(model_dir, "**", g), recursive=True)
            for filename in filenames:
                basename = os.path.basename(filename)
                if "coast" in basename.lower():
                    coastlines.append(filename)
                if (
                    "continent" in basename.lower()
                    or "terrane" in basename.lower()
                    or (
                        "static" in basename.lower()
                        and "polygon" in basename.lower()
                    )
                ):
                    continent_candidates.append(filename)
        continents = continent_candidates or coastlines
        return coastlines, continents, None # None for COBs
    
    # Main function
    topology_tf = None
    if plate_reconstruction is None:
        plate_reconstruction = get_plate_reconstruction(
            model_name=model_name,
            model_dir=model_dir,
            anchor_plate_id=anchor_plate_id,
            filter_topologies=filter_topologies,
        )
        if filter_topologies and isinstance(plate_reconstruction, tuple):
            plate_reconstruction, topology_tf = plate_reconstruction
    
    if model_name is None: # Alfonso2024 provided reconstruction
        coastlines, continents, COBs = _find_plate_topologies()
    else:
        model = _fetch_plate_model(model_name, model_dir)
        # Collect files
        coastlines = model.get_coastlines()
        continents = model.get_continental_polygons()
        COBs = model.get_COBs(return_none_if_not_exist=True)

    plot_topologies = PlotTopologies(
        plate_reconstruction=plate_reconstruction, 
        coastlines=coastlines, 
        continents=continents, 
        COBs=COBs,
        time=time,
        anchor_plate_id=anchor_plate_id
    )
    if topology_tf is not None:
        # Keep temporary filtered topology file alive for object lifetime.
        plot_topologies._topology_tf = topology_tf
    return plot_topologies
