"""Functions to sample mantle data from G-ADOPT
output grids and join to point data.
"""
import numpy as np
import pandas as pd
import xarray as xr
import pint_xarray # noqa: F401
import gplately as gplt
import warnings

from numpy.typing import ArrayLike
from .misc import _PathLike

# Non-dimensionalisation offset used by G-ADOPT: surface radius in Earth radii.
_GADOPT_SURFACE_R = 2.208
_MANTLE_THICKNESS = 2891.0  # km

DEFAULT_DEPTHS: np.ndarray = np.arange(100, 500, 100)  # km (upper mantle)
BASE_MANTLE_VARS: set[str, ...] = {
    "FullTemperature_CG",
    "Pressure",
    "Radial_Velocity",
    "Temperature_CG",
    "Temperature_Deviation_CG",
    "Velocity_x",
    "Velocity_y",
    "Velocity_z",
    "Viscosity_CG",
    "East_Velocity",
    "North_Velocity",
}

# ==================
# Feature registry
# ==================


class MantleVariableRegistry:
    def __init__(self):
        self._variables: dict[str, callable] = {}
        self._transforms: dict[str, callable] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, name: str):
        """Decorator to register a variable computation function."""
        def decorator(fn):
            self._variables[name] = fn
            return fn
        return decorator
    
    def register_base(self, *names: str):
        """Register one or more base variables that are loaded directly from the dataset."""
        for name in names:
            self._variables[name] = lambda ds, n=name: ds[n]

    def register_transform(self, name: str):
        """Register a named transformation (e.g. depth averaging)."""
        def decorator(fn):
            self._transforms[name] = fn
            return fn
        return decorator

    def register_derived(self, name: str, transform: str, **kwargs):
        """Register a new named variable by applying a transform to an existing one."""
        if transform not in self._transforms:
            raise KeyError(f"Unknown transform: '{transform}'. Available: {list(self._transforms)}")
        self._variables[name] = lambda ds: self._transforms[transform](ds, **kwargs)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get(self, name: str, ds: xr.Dataset) -> xr.DataArray:
        if name not in self._variables:
            raise KeyError(f"Unknown variable: '{name}'. Available: {list(self._variables)}")
        result = self._variables[name](ds)
        return result.rename(name) if isinstance(result, xr.DataArray) else result

    def available(self) -> list[str]:
        return list(self._variables)


variables = MantleVariableRegistry()

variables.register_base(*BASE_MANTLE_VARS)


# ==================
# Helper functions
# ==================


def _to_km(depths: ArrayLike) -> np.ndarray:
    """Convert G-ADOPT nondimensionalised depths to km below surface."""
    return (_GADOPT_SURFACE_R - np.asarray(depths, dtype=float)) * _MANTLE_THICKNESS


def _to_nondim(depths_km: ArrayLike) -> np.ndarray:
    """Convert depths in km below surface to G-ADOPT nondimensionalised depths."""
    return _GADOPT_SURFACE_R - (np.asarray(depths_km, dtype=float) / _MANTLE_THICKNESS)


# ==================
# Derived varaibles
# ==================

@variables.register("Cell_Volume")
def _cell_volume(ds: xr.Dataset) -> xr.DataArray:
    """Calculate spherical cell volumes from center coordinates."""

    def centers_to_edges(centers: xr.DataArray, dim: str) -> xr.DataArray:
        """Convert evenly-spaced cell centers to edges."""
        spacing = centers.diff(dim).mean().item()
        interior = (centers.values[:-1] + centers.values[1:]) / 2
        edges = np.concatenate([[centers.values[0] - spacing/2],
                                interior,
                                [centers.values[-1] + spacing/2]])
        return xr.DataArray(edges, dims=[dim], attrs=centers.attrs).sortby(dim)
    
    r      = centers_to_edges(ds.depth, 'depth')
    theta  = centers_to_edges(np.deg2rad(90 - ds.lat), 'lat')
    phi    = centers_to_edges(np.deg2rad(ds.lon), 'lon')

    da = (r**3).diff('depth') / 3 * (-np.cos(theta)).diff('lat') * phi.diff('lon')
    
    depth_units = ds.depth.attrs["units"]
    da.attrs = {
        "units": f"{depth_units}**3",
        "long_name": "cell volume",
    }
    
    return da.rename("Cell_Volume")


@variables.register("Speed")
def _speed(ds: xr.Dataset) -> xr.DataArray:
    """Calculate flow speed from velocity components."""
    ds = ds.pint.quantify()
    
    vx = ds["Velocity_x"]
    vy = ds["Velocity_y"]
    vz = ds["Velocity_z"]
    da = np.sqrt(vx**2 + vy**2 + vz**2)
    
    da.attrs = {"long_name": "speed"}
    da = da.pint.dequantify()
    return da.rename("Speed")


@variables.register("Tangential_Speed")
def _tangential_speed(ds: xr.Dataset) -> xr.DataArray:
    """Calculate tangential speed from velocity components."""
    ds = ds.pint.quantify()
    
    ve = ds["East_Velocity"]
    vn = ds["North_Velocity"]
    da = np.sqrt(ve**2 + vn**2)
    
    da.attrs = {"long_name": "tangential speed"}
    da = da.pint.dequantify()
    return da.rename("Tangential_Speed")


@variables.register("Radial_Tangential_Ratio")
def _radial_tangential_ratio(ds: xr.Dataset) -> xr.DataArray:
    """Calculate radial-to-tangential velocity ratio."""
    ds = ds.pint.quantify()
    
    vr = ds["Radial_Velocity"]
    vt = _tangential_speed(ds)
    da = np.abs(vr) / vt
    
    da.attrs = {"long_name": "radial-to-tangential velocity ratio"}
    da = da.pint.dequantify()
    return da.rename("Radial_Tangential_Ratio")


@variables.register("LAB_Depth")
def _LAB_depth(ds: xr.Dataset) -> xr.DataArray:
    if "Lithosphere_Indicator" not in ds:
        raise ValueError("Dataset must contain 'Lithosphere_Indicator' variable to calculate LAB depth.")
       
    da = xr.where(ds["Lithosphere_Indicator"] > 0.5, ds["depth"], 0, keep_attrs=True).max(dim="depth")
    
    da.attrs = {"long_name": "depth to lithosphere-asthenosphere boundary"}
    return da.rename("LAB_Depth")


@variables.register("Slab_Depth") #TODO
def _slab_depth(ds: xr.Dataset) -> xr.DataArray:
    pass


# Features requiring plate velocities (need to be extracted from plate model) (perhaps a different feature registry/module?) (TODO)


def _plate_velocity_delta(ds: xr.Dataset) -> xr.DataArray: #TODO
    pass


def _relative_tangential_velocity(ds: xr.Dataset) -> xr.DataArray: #TODO
    pass  


# ==================
# Transforms
# ==================

@variables.register_transform("average_over_depth") #TODO
def _average_over_depth(
    ds: xr.Dataset,
    var_name: str,
    depth_range: tuple[float, float] | None = None,
) -> xr.DataArray:
    """Average a variable over a depth range, defaulting to full extent of the mantle."""
    da = variables.get(var_name, ds)
    if "depth" not in da.dims:
        raise ValueError(f"'{var_name}' has no 'depth' dimension.")

    depth_range = depth_range or (float(da.depth.min()), float(da.depth.max()))
    d0, d1 = depth_range

    da = da.sel(depth=(da.depth >= d0) & (da.depth <= d1)).mean(dim="depth")
    return da.rename(f"{var_name}_avg_{int(d0)}-{int(d1)}km")


# ==================
# Feature extraction
# ==================


def _detect_lon_convention(lon_values: ArrayLike) -> str | None:
    """Detect longitude convention: '0-360', '-180-180', or None if unknown."""
    vals = np.asarray(lon_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None

    lon_min = float(np.min(vals))
    lon_max = float(np.max(vals))

    if lon_min >= 0.0 and lon_max > 180.0:
        return "0-360"
    if lon_min < 0.0 and lon_max <= 180.0:
        return "-180-180"
    if lon_min < 0.0 and lon_max > 180.0:
        raise ValueError(
            f"Cannot determine dataset longitude convention from range [{lon_min:.2f}, {lon_max:.2f}]."
        )
    return None


def _normalize_lons(lons: ArrayLike, convention: str | None) -> np.ndarray:
    """Normalize longitudes to a target convention."""
    lons_arr = np.asarray(lons, dtype=float)
    if convention == "0-360":
        return np.mod(lons_arr, 360.0)
    if convention == "-180-180":
        return ((lons_arr + 180.0) % 360.0) - 180.0
    return lons_arr


def _match_longitude_convention(da: xr.DataArray, lons: ArrayLike) -> np.ndarray:
    """Map longitudes to match the dataset longitude convention."""
    convention = _detect_lon_convention(da["lon"].values)
    return _normalize_lons(lons, convention)


def _wrap_longitude_seam(
    da: xr.DataArray,
) -> np.ndarray:
    """Make longitude cyclic so seam points interpolate using 359<->0 neighbours."""

    lon_vals = np.asarray(da["lon"].values, dtype=float)
    lon_vals = lon_vals[np.isfinite(lon_vals)]
    if lon_vals.size == 0:
        raise ValueError("Longitude coordinate is empty.")

    # Add wrapped slices on both sides of the longitude axis to interpolate
    # across the seam (e.g. between 359 and 0 degrees).
    left = da.isel(lon=-1).assign_coords(lon=da["lon"].isel(lon=-1) - 360.0)
    right = da.isel(lon=0).assign_coords(lon=da["lon"].isel(lon=0) + 360.0)
    da_cyclic = xr.concat([left, da, right], dim="lon").sortby("lon")
    
    return da_cyclic


def sample_mantle_var(
    da: xr.DataArray,
    lons: ArrayLike = None,
    lats: ArrayLike = None,
    times: ArrayLike = None,
    depths: ArrayLike = None,
    method: str = "linear",
) -> (pd.Series | pd.DataFrame):
    """Interpolate a mantle data array at requested coordinates.

    This function validates that every dimension in ``da`` has a matching
    coordinate argument. Only coordinates for dimensions present in ``da`` are
    passed to :meth:`xarray.DataArray.interp`.

    If longitude sampling is requested, longitudes are first normalized to the
    dataset convention, then the longitude seam is wrapped to support cyclic
    interpolation across 0/360.

    Depth handling is shape-based:
    - If ``depth`` is a dimension in ``da`` and ``depths`` is interpreted as an
      independent axis, depth is passed orthogonally (dimension ``("depth",)``), 
      and outputs will be broadcast for each depth value.
    - Otherwise, depth is treated as point-wise and passed with
      dimension ``("points",)``.

    Parameters
    ----------
    da : xr.DataArray
        Data array to sample.
    lons, lats, times, depths : array-like or None
        Candidate interpolation coordinates. Any coordinate corresponding to a
        dimension in ``da`` must be provided.
    method : {'linear', 'nearest'}, optional
        Interpolation method passed to ``xarray.DataArray.interp``.

    Returns
    -------
    np.ndarray
        Interpolated values. Output shape follows xarray interpolation semantics
        for the provided coordinate dimensions.
    """
    
    # Check if available coordinates match those of the data array; if not, raise an error
    coord_map = {"lon": lons, "lat": lats, "time": times, "depth": depths}
    for coord in da.dims:
        if coord_map.get(coord) is None:
            raise ValueError(
                f"Data array '{da.name}' has dimension '{coord}' but no corresponding coordinates were provided."
            )
    
    broadcast_depth_axis = False
    if depths is not None:
        depths_arr = np.asarray(depths, dtype=float)
        broadcast_depth_axis = depths_arr.ndim == 1 and all(
            np.asarray(c).size != len(depths_arr)
            for c in (lons, lats, times) if c is not None
        )
    
    if lons is not None:
        lons = _match_longitude_convention(da=da, lons=lons)
        da = _wrap_longitude_seam(da)

    interp_coords = {}
    if "lon" in da.dims:
        interp_coords["lon"] = xr.DataArray(np.asarray(lons, dtype=float), dims="points")
    if "lat" in da.dims:
        interp_coords["lat"] = xr.DataArray(np.asarray(lats, dtype=float), dims="points")
    if "time" in da.dims:
        interp_coords["time"] = xr.DataArray(np.asarray(times, dtype=float), dims="points")
    if "depth" in da.dims:
        depths_arr = np.asarray(depths, dtype=float)
        # Broadcast depth axis if needed to match point-wise coordinates.
        interp_coords["depth"] = (
            xr.DataArray(depths_arr, dims="depth")
            if broadcast_depth_axis
            else xr.DataArray(depths_arr, dims="points")
        )

    result = da.interp(**interp_coords, method=method).to_pandas()

    if np.isnan(result).any():
        warnings.warn(f"NaN values found in sampled mantle data ({da.name}). Check that all points are within the dataset bounds.")

    return result


def _sample_LAB_depths(
    ds: xr.Dataset,
    da: xr.DataArray,
    times: ArrayLike=None,
    lats: ArrayLike=None,
    lons: ArrayLike=None,
    offset_km: float=0.0,
) -> np.ndarray:
    """Sample mantle variable at the depth of the lithosphere-asthenosphere boundary."""
    
    coord_list = {k: v for k, v in {"times": times, "lats": lats, "lons": lons}.items() if v is not None}
    
    lab_depths = sample_mantle_var(
        da=variables.get("LAB_Depth", ds),
        **coord_list,
        method="linear",
    )
    
    return sample_mantle_var(
        da=da,
        **coord_list,
        depths=lab_depths + offset_km,
        method="nearest",
    )


def calculate_lambdas(da: xr.DataArray, n_lambdas: int=5) -> pd.DataFrame:
    pass


# ==================
# API functions
# ==================

def extract_basic_mantle_features(
    points: pd.DataFrame,
    mantle_dir: _PathLike,
    depths: ArrayLike|str = DEFAULT_DEPTHS,
    features_to_extract: ArrayLike = BASE_MANTLE_VARS,
) -> pd.DataFrame:
    """Extract values of mantle fields at (time, depth, lat, lon) points.

    Parameters
    ----------
    mantle_dir : path-like
        Directory containing G-ADOPT output files.
    points : DataFrame
        Must contain columns 'lon', 'lat', and 'age (Ma)'.
    depths : array-like, optional
        Depths in km below surface. Default is `DEFAULT_DEPTHS` (100, 200, 300, 400 km).
        If size of `depths` matches number of rows in `points`, each point is sampled at its corresponding depth.
        Otherwise, 
    features_to_extract : array-like of str, optional
        Names of mantle variables to extract.

    Returns
    -------
    DataFrame
        Copy of `points` with additional columns for each sampled mantle
        variable and depth (e.g. 'Temperature_Deviation_CG_100km').
    """
    depths = list(depths)
    out = points.copy()
    new_cols = {}

    with xr.open_mfdataset(
        sorted(mantle_dir.glob("*.nc")),
        combine='nested',
        concat_dim='time',
        chunks={"time": 1, "depth": 25},
    ) as ds:
        for var in features_to_extract:
            sampled = sample_mantle_var(
                ds=ds,
                var=var,
                lons=points["lon"],
                lats=points["lat"],
                times=points["age (Ma)"],
                depths=depths,
            )
            for i, depth_km in enumerate(depths):
                new_cols[f"{var}_{int(depth_km)}km"] = sampled[:, i]

    # Join all new columns at once to avoid DataFrame fragmentation.
    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)

    return out


def extract_derived_mantle_features(
    mantle_dir: _PathLike, 
    points: pd.DataFrame, 
    depths: ArrayLike = DEFAULT_DEPTHS,
    features_to_extract: ArrayLike = None,
) -> pd.DataFrame:
    """Calculate derived mantle features and add to dataset."""
    pass