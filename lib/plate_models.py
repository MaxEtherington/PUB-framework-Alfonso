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
from .misc import (
    filter_topological_features,
    _PathLike,
)
    

_LAYER_NAMES = (
    'Rotations', 'StaticPolygons', 'Coastlines',
    'ContinentalPolygons', 'Topologies', 'COBs',
)

def cache_plate_model(model_name: str, model_dir: _PathLike) -> None:
    """Cache plate model files to model_dir via PlateModelManager."""
    model = PlateModelManager().get_model(model_name=model_name, data_dir=model_dir)
    model.get_rotation_model()
    for layer in _LAYER_NAMES:
        model.get_layer(layer, return_none_if_not_exist=True)

def _fetch_plate_model(model_name: str, model_dir: _PathLike) -> PlateModel:
    """Fetch a plate model via PMM, falling back to local files if the fetch fails."""
    try:
        return PlateModelManager().get_model(model_name=model_name, data_dir=model_dir)
    except Exception:
        model = PlateModel(
            model_name=model_name,
            data_dir=model_dir,
            model_cfg=Path(model_dir) / ".metadata.json",
            readonly=True,
        )
        if not model.is_model_dir(Path(model_dir) / model_name):
            raise
        return model

def _scan_model_dir(
    model_dir: _PathLike,
) -> dict[str, Optional[list[str]]]:
    """Single-pass filesystem scan returning plate model files by category."""
    rotations, topologies, static_polygons = [], [], []
    coastlines, continent_candidates, cobs = [], [], []

    for path in glob.glob(os.path.join(model_dir, "**", "*.rot"), recursive=True):
        rotations.append(path)

    for ext in ("*.gpml", "*.gpmlz"):
        for path in glob.glob(os.path.join(model_dir, "**", ext), recursive=True):
            name = os.path.basename(path).lower()
            topologies.append(path)
            is_static_polygon = "static" in name and "polygon" in name
            if is_static_polygon:
                static_polygons.append(path)
            if "coast" in name:
                coastlines.append(path)
            if "continent" in name or "terrane" in name or is_static_polygon:
                continent_candidates.append(path)
            if "cob" in name:
                cobs.append(path)

    return {
        'Rotations':            rotations or None,
        'StaticPolygons':       static_polygons or None,
        'Coastlines':           coastlines or None,
        'ContinentalPolygons':  (continent_candidates or coastlines) or None,
        'Topologies':           topologies or None,
        'COBs':                 cobs or None,
    }


def get_model_filenames(
    plate_reconstruction: Optional[PlateReconstruction] = None,
    model_dir: Optional[str | os.PathLike] = None,
) -> dict[str, Optional[list[str]]] | None:
    """
    Return a dict mapping layer names to file lists (None if a layer is absent).

    Prefers plate_reconstruction.plate_model if available; falls back to a
    filesystem scan of model_dir. Returns None if neither source is usable.
    """
    if plate_reconstruction is not None:
        if (plate_model := plate_reconstruction.plate_model) is not None:
            filenames = {
                layer: plate_model.get_layer(layer, return_none_if_not_exist=True)
                for layer in _LAYER_NAMES
            }
            filenames['Rotations'] = plate_model.get_rotation_model()
            return filenames

    if model_dir is not None:
        return _scan_model_dir(model_dir)

    return None

def get_plate_reconstruction(
    model_name: Optional[str] = None,
    model_dir: _PathLike = "plate_model",
    anchor_plate_id: int = 0,
    filter_topologies: bool = False,
) -> PlateReconstruction | tuple[PlateReconstruction, _TemporaryFileWrapper]:
    """Get a `PlateReconstruction` object from a model name and directory.

    If `model_name` is `None` use local files in `model_dir`.

    Parameters
    ----------
    model_name : str, optional
        Name of the model to fetch from PMM.
    model_dir : _PathLike, default: 'plate_model'
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
    
    model = None
    if model_name is None: # Alfonso2024 provided reconstruction
        filenames = _scan_model_dir(model_dir)
        rotation_files  = filenames['Rotations']
        topology_files  = filenames['Topologies']
        static_polygons = filenames['StaticPolygons']
    else:
        model = _fetch_plate_model(model_name, model_dir)
        rotation_files  = model.get_rotation_model()
        topology_files  = model.get_topologies()
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
    model_dir: _PathLike = "plate_model",
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
    model_dir : _PathLike, default: 'plate_model'
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
    
    topology_tf = None
    if plate_reconstruction is None:
        result = get_plate_reconstruction(
            model_name=model_name,
            model_dir=model_dir,
            anchor_plate_id=anchor_plate_id,
            filter_topologies=filter_topologies,
        )
        if filter_topologies:
            plate_reconstruction, topology_tf = result
        else:
            plate_reconstruction = result

    if model_name is None:
        filenames  = _scan_model_dir(model_dir)
        coastlines = filenames['Coastlines']
        continents = filenames['ContinentalPolygons']
        cobs       = filenames['COBs']
    else:
        model      = plate_reconstruction.plate_model or _fetch_plate_model(model_name, model_dir)
        coastlines = model.get_coastlines()
        continents = model.get_continental_polygons()
        cobs       = model.get_COBs(return_none_if_not_exist=True)

    plot_topologies = PlotTopologies(
        plate_reconstruction=plate_reconstruction,
        coastlines=coastlines,
        continents=continents,
        COBs=cobs,
        time=time,
        anchor_plate_id=anchor_plate_id,
    )
    if topology_tf is not None:
        plot_topologies._topology_tf = topology_tf
    return plot_topologies
