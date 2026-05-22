"""Functions to sample mantle data from G-ADOPT
output grids and join to point data.
"""
from dataclasses import dataclass
from typing import Callable, Any
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import pint_xarray # noqa: F401
from scipy.interpolate import RegularGridInterpolator

from numpy.typing import ArrayLike

# Non-dimensionalisation offset used by G-ADOPT: surface radius in Earth radii.
_GADOPT_SURFACE_R = 2.208
_MANTLE_THICKNESS = 2891.0  # km

DEFAULT_DEPTHS: np.ndarray = np.arange(100, 500, 100)  # km (upper mantle)
BASE_MANTLE_VARS: list[str] = [
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
]

# ==================
# Feature registry
# ==================

@dataclass
class MantleVariable:
    """A registered mantle variable with a computation function."""
    name: str
    compute: Callable[..., xr.DataArray]


class MantleVariableRegistry:
    def __init__(self):
        self._variables: dict[str, MantleVariable] = {}
        self._transforms: dict[str, Callable[..., Any]] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, name: str):
        """Decorator to register a variable computation function."""
        def decorator(fn):
            self._variables[name] = MantleVariable(name=name, compute=fn)
            return fn
        return decorator

    def register_base(self, *names: str):
        """Register one or more base variables that are loaded directly from the dataset."""
        for name in names:
            self._variables[name] = MantleVariable(
                name=name,
                compute=lambda ds, n=name: ds[n],
            )

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
        self._variables[name] = MantleVariable(
            name=name,
            compute=lambda ds: self._transforms[transform](ds, **kwargs),
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get(self, name: str, ds: xr.Dataset) -> xr.DataArray:
        if name not in self._variables:
            raise KeyError(f"Unknown variable: '{name}'. Available: {list(self._variables)}")

        if name in ds.data_vars:
            return ds[name]

        result = self._variables[name].compute(ds)
        ds[name] = result  # Cache in dataset for future retrievals
        return result.rename(name) if isinstance(result, xr.DataArray) else result

    @property
    def available(self) -> list[str]:
        return list(self._variables)


variables = MantleVariableRegistry()

variables.register_base(*BASE_MANTLE_VARS)


# ==================
# Transforms
# ==================

@variables.register_transform("average_over_depth")
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


def _average_over_LAB_depth(
    ds: xr.Dataset,
    var_name: str,
    offset: float = None,
    name_suffix: str = None,
) -> xr.DataArray:
    """Average a variable over the depth range from LAB to LAB+depth."""
    da = variables.get(var_name, ds)
    if "depth" not in da.dims:
        raise ValueError(f"'{var_name}' has no 'depth' dimension.")

    lab_depth = variables.get("LAB_Depth", ds)
    result = da.where((da.depth >= lab_depth) & (da.depth <= lab_depth + offset)).mean(dim="depth")
    result.attrs = {**da.attrs, "long_name": f"{result.attrs.get('long_name', var_name)} averaged from LAB to {offset}km below"}
    if name_suffix is not None:
        name = f"{da.name}{name_suffix}"
    else:
        name = f"{var_name}_avg_LAB_{int(offset)}km"
    return result.rename(name)


@variables.register_transform("average_over_rolling_time")
def _average_over_rolling_time(
    ds: xr.Dataset,
    var_name: str,
    window_size: int,
) -> xr.DataArray:
    """Calculate a rolling average of a variable over time."""
    da = variables.get(var_name, ds)
    if "time" not in da.dims:
        raise ValueError(f"'{var_name}' has no 'time' dimension.")

    da = da.sortby("time", ascending=False)  # Ensure time (Ma) decreases so rolling average looks backward
    result = da.rolling(time=window_size, center=True, min_periods=1).mean()
    result.attrs = {**da.attrs, "long_name": f"{result.attrs.get('long_name', var_name)} rolling average over {window_size} Ma"}
    return result.rename(f"{var_name}_rolling_{window_size}Ma")


@variables.register_transform("contour_depth")
def _calculate_contour_depth(
    ds: xr.Dataset,
    var_name: str,
    target_contour: float,
    first_crossing: bool = True,
    nth_crossing: int | None = None,
) -> xr.DataArray:
    # --- Validate ---
    try:
        da = variables.get(var_name, ds)
    except KeyError:
        da = ds[var_name]
    assert "depth" in da.dims, "DataArray must have a 'depth' dimension"
    depth_vals = da.depth.values
    assert np.all(np.diff(depth_vals) < 0), "Depth must be monotonically decreasing"

    # --- Transpose to canonical order, compute if Dask-backed ---
    canonical_order = [d for d in ("time", "depth", "lat", "lon") if d in da.dims]
    extra_dims = [d for d in canonical_order if d != "depth"]
    da = da.transpose(*canonical_order).compute()

    # --- Move depth to axis 0, flip to shallow→deep, flatten remaining dims ---
    arr = np.moveaxis(da.values, canonical_order.index("depth"), 0)   # (D, ...)
    arr = arr[::-1]                                                   # shallow→deep
    depth_vals = depth_vals[::-1]
    horiz_shape = arr.shape[1:]                                       # e.g. (T, lat, lon)
    arr_2d = arr.reshape(arr.shape[0], -1)                            # (D, N)

    # --- Resolve effective crossing index ---
    if nth_crossing is None:
        nth_crossing = 0 if first_crossing else -1

    # --- Find nth sign-change along depth for each column ---
    diff = arr_2d - target_contour                                    # (D, N)
    sign_changes = np.diff(np.sign(diff), axis=0) != 0                # (D-1, N)

    n_needed = nth_crossing + 1 if nth_crossing >= 0 else -nth_crossing
    has_crossing = sign_changes.sum(axis=0) >= n_needed               # (N,)

    if nth_crossing >= 0:
        cumsum = sign_changes.cumsum(axis=0)                          # (D-1, N)
        idx = np.argmax(cumsum == nth_crossing + 1, axis=0)           # (N,)
    else:
        sign_flipped = np.flip(sign_changes, axis=0)
        cumsum_flipped = sign_flipped.cumsum(axis=0)
        idx_flipped = np.argmax(cumsum_flipped == -nth_crossing, axis=0)
        idx = (sign_changes.shape[0] - 1) - idx_flipped

    # --- Linear interpolation to exact crossing depth ---
    col = np.arange(arr_2d.shape[1])
    d0, d1 = diff[idx, col], diff[idx + 1, col]
    z0, z1 = depth_vals[idx], depth_vals[idx + 1]

    depth_crossing = z0 + (-d0) / (d1 - d0) * (z1 - z0)
    depth_crossing = np.where(has_crossing, depth_crossing, np.nan)  # (N,)

    # --- Reshape back and wrap in DataArray ---
    coords = {d: da.coords[d] for d in extra_dims if d in da.coords}
    if nth_crossing == 0:
        crossing_label = "first"
    elif nth_crossing == -1:
        crossing_label = "last"
    else:
        crossing_label = f"nth={nth_crossing}"
    return xr.DataArray(
        depth_crossing.reshape(horiz_shape),
        dims=extra_dims,
        coords=coords,
        attrs={
            "long_name": f"Depth of {crossing_label} {target_contour} {da.attrs.get('units', '').strip()} contour",
            "units": "km",
        },
    ).rename(f"{da.name}_{target_contour}_contour_depth_{crossing_label}")


# ==================
# Derived variables
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
    vr = ds["Radial_Velocity"]
    vt = _tangential_speed(ds)

    for v in [vr, vt]:
        v = v.pint.quantify()
    da = np.abs(vr) / vt

    da.attrs = {"long_name": "radial-to-tangential velocity ratio"}
    da = da.pint.dequantify()
    return da.rename("Radial_Tangential_Ratio")


@variables.register("LAB_Depth_Terraced")
def _LAB_depth(ds: xr.Dataset) -> xr.DataArray:
    if "Lithosphere_Indicator" not in ds:
        raise ValueError("Dataset must contain 'Lithosphere_Indicator' variable to calculate LAB depth.")

    da = xr.where(ds["Lithosphere_Indicator"] > 0.5, ds["depth"], 0, keep_attrs=True).max(dim="depth")

    da.attrs = {"long_name": "depth to lithosphere-asthenosphere boundary"}
    return da.rename("LAB_Depth")


@variables.register("LAB_Depth")
def _LAB_depth_contour(ds: xr.Dataset) -> xr.DataArray:
    da = _calculate_contour_depth(
        ds=ds,
        var_name="Lithosphere_Indicator",
        target_contour=0.5,
        first_crossing=False, # LAB is defined as the last crossing of the 0.5 contour from shallow to deep.
    )
    # Fallback to first non-boundary depth where no crossing is found, to avoid NaNs in difference calculations.
    # This is a heuristic choice; the actual LAB depth in these regions may be different.
    min_permissible_depth = ds.depth.isel(depth=-2).to_numpy()
    da = da.where(~da.isnull() | (da > min_permissible_depth), min_permissible_depth)
    da.attrs = {"long_name": "depth to lithosphere-asthenosphere boundary", "units": "km"}
    return da.rename("LAB_Depth_Contour")


def _compute_slab_crossings(
    ds: xr.Dataset,
    target_contour: float = 1000.0,
    contour_var: str = "Temperature_CG",
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Compute the three temperature crossings needed for slab geometry, caching results in ds.

    Returns:
        c1: first crossing in full mantle (base of lithosphere / LAB proxy)
        c2: second crossing in upper mantle ≤410 km (slab top); NaN where no slab present
        c3: third crossing in full mantle (slab bottom); NaN-filled to CMB where slab has no exit
    """
    cache_prefix = f"_slab_cross_{int(target_contour)}_{contour_var}"
    c1_key, c2_key, c3_key = f"{cache_prefix}_1", f"{cache_prefix}_2", f"{cache_prefix}_3"
    if c1_key in ds:
        return ds[c1_key], ds[c2_key], ds[c3_key]

    c1 = _calculate_contour_depth(ds.sel(depth=slice(2900, 0)), contour_var,
                                   target_contour=target_contour, nth_crossing=0)
    c2 = _calculate_contour_depth(ds.sel(depth=slice(410, 0)),  contour_var,
                                   target_contour=target_contour, nth_crossing=1)
    c3 = _calculate_contour_depth(ds.sel(depth=slice(2900, 0)), contour_var,
                                   target_contour=target_contour, nth_crossing=2)

    lab = variables.get("LAB_Depth", ds)
    has_2nd = ~c2.isnull()
    cmb_depth = float(ds.depth.max())  # depth is descending; max = deepest ≈ 2900 km

    # Slab enters upper mantle but temperature never recovers → extend to CMB
    c3 = xr.where(c3.isnull() & has_2nd, cmb_depth, c3)
    # No cold anomaly at all → anchor first crossing at LAB so arithmetic stays defined
    c1 = c1.fillna(lab)

    for key, arr in [(c1_key, c1), (c2_key, c2), (c3_key, c3)]:
        ds[key] = arr
    return c1, c2, c3


@variables.register("Slab_Top_Depth")
def _slab_top_depth(ds: xr.Dataset) -> xr.DataArray:
    c1, c2, c3 = _compute_slab_crossings(ds)
    lab = variables.get("LAB_Depth", ds)
    has_2nd = ~c2.isnull()
    da = xr.where(has_2nd, c2, lab)
    da.attrs = {
        "long_name": "slab top depth (1000 K Temperature_CG 2nd crossing / LAB)",
        "units": "km",
    }
    return da.rename("Slab_Top_Depth")


@variables.register("Slab_Bottom_Depth")
def _slab_bottom_depth(ds: xr.Dataset) -> xr.DataArray:
    c1, c2, c3 = _compute_slab_crossings(ds)
    lab = variables.get("LAB_Depth", ds)
    has_2nd = ~c2.isnull()
    da = xr.where(has_2nd, c3, np.maximum(c1, lab))
    da.attrs = {
        "long_name": "slab bottom depth (1000 K Temperature_CG 3rd crossing / 1st crossing)",
        "units": "km",
    }
    return da.rename("Slab_Bottom_Depth")


@variables.register("Slab_Thickness")
def _slab_thickness(ds: xr.Dataset) -> xr.DataArray:
    da = variables.get("Slab_Bottom_Depth", ds) - variables.get("Slab_Top_Depth", ds)
    da.attrs = {"long_name": "slab thickness (1000 K Temperature_CG)", "units": "km"}
    return da.rename("Slab_Thickness")


@variables.register("Sublithospheric_Cold_Anomaly_Thickness")
def _sublithospheric_cold_anomaly_thickness(ds: xr.Dataset) -> xr.DataArray:
    da = variables.get("Slab_Bottom_Depth", ds) - variables.get("LAB_Depth", ds)
    da.attrs = {
        "long_name": "sublithospheric cold anomaly thickness (1000 K Temperature_CG)",
        "units": "km",
    }
    return da.rename("Sublithospheric_Cold_Anomaly_Thickness")


@variables.register("Mantle_Wedge_Thickness")
def _mantle_wedge_thickness(ds: xr.Dataset) -> xr.DataArray:
    da = variables.get("Slab_Top_Depth", ds) - variables.get("LAB_Depth", ds)
    da.attrs = {"long_name": "mantle wedge thickness (1000 K Temperature_CG)", "units": "km"}
    return da.rename("Mantle_Wedge_Thickness")


@variables.register("Cold_Anomaly_Magnitude")
def _min_upper_mantle_relative_temperature(ds: xr.Dataset) -> xr.DataArray:
    da = variables.get("Temperature_Deviation_CG", ds)
    upper_mantle = da.sel(depth=(da.depth >= 0) & (da.depth <= 400))
    min_temp_dev = upper_mantle.min(dim="depth")
    min_temp_dev.attrs = {"long_name": "magnitude of sub-lithospheric cold anomaly", "units": da.attrs.get("units", "")}
    return min_temp_dev.rename("Cold_Anomaly_Magnitude")


# ====================================
# Transformed variables
# ====================================

# Register depth averages and rolling time averages of temperature deviation
for top, bot in [(0, 400), (100, 400)]:
    av_name = f"Temperature_Deviation_Avg_{top}-{bot}km"
    @variables.register(av_name)
    def _(ds: xr.Dataset, _top=top, _bot=bot) -> xr.DataArray:
        return _average_over_depth(
            ds=ds,
            var_name="Temperature_Deviation_CG",
            depth_range=(_top, _bot),
        )

    for window in [30, 50]:
        @variables.register(f"{av_name}_Rolling_{window}Ma")
        def _(ds: xr.Dataset, _av_name=av_name, _window=window) -> xr.DataArray:
             return _average_over_rolling_time(
                ds=ds,
                var_name=_av_name,
                window_size=_window,
            )

# Register depth averages and rolling time averages of temperature deviation averaged from LAB
for offset in [120, 160, 200, 300, 400]:
    av_name = f"Temperature_Deviation_Avg_LAB-{offset}km"
    @variables.register(av_name)
    def _(ds: xr.Dataset, _offset=offset) -> xr.DataArray:
        return _average_over_LAB_depth(
            ds=ds,
            var_name="Temperature_Deviation_CG",
            offset=_offset,
            name_suffix=f"_LAB-{_offset}km",
        )
    for window in [30, 50]:
        @variables.register(f"{av_name}_Rolling_{window}Ma")
        def _(ds: xr.Dataset, _av_name=av_name, _window=window) -> xr.DataArray:
             return _average_over_rolling_time(
                ds=ds,
                var_name=_av_name,
                window_size=_window,
            )


# ====================================
# Mantle variable sampling utilities
# ====================================


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


def _do_interp_cyclic_lon(
    da: xr.DataArray,
    lons: ArrayLike,
    lats: ArrayLike,
    times: ArrayLike,
    depths: ArrayLike,
    broadcast_depth: bool = False,
    method: str = "linear",
) -> xr.DataArray:
    """
    Interpolate with antimeridian longitude seam wrapping; depth is broadcast
    (treated as orthogonal) when requested. Uses RegularGridInterpolator for
    joint ND interpolation, avoiding the sequential-axis artefacts of da.interp.
    """
    if lons is not None:
        lons = _match_longitude_convention(da=da, lons=lons)
        da = _wrap_longitude_seam(da)

    # Active dims in da's own axis order — drives both RGI grid and query columns
    active_dims = [d for d in da.dims if d in {"lon", "lat", "depth", "time"}]

    rgi = RegularGridInterpolator(
        points=tuple(da[d].values.astype(float) for d in active_dims),
        values=da.values,
        method=method,
        bounds_error=False,
        fill_value=np.nan,
    )

    # Numeric query arrays, keyed by dim name
    point_coords: dict[str, np.ndarray] = {}
    if "lon"   in active_dims: point_coords["lon"]   = np.asarray(lons,   dtype=float)
    if "lat"   in active_dims: point_coords["lat"]   = np.asarray(lats,   dtype=float)
    if "time"  in active_dims: point_coords["time"]  = np.asarray(times,  dtype=float)
    if "depth" in active_dims: point_coords["depth"] = np.asarray(depths, dtype=float)

    if broadcast_depth and "depth" in active_dims:
        # Cartesian product: repeat each point coord n_depths times,
        # tile depth array n_points times → shape (n_points * n_depths, n_dims)
        n_points = len(next(v for k, v in point_coords.items() if k != "depth"))
        n_depths = len(point_coords["depth"])
        query = np.column_stack([
            np.tile(point_coords[d], n_points) if d == "depth"
            else np.repeat(point_coords[d], n_depths)
            for d in active_dims
        ])
        values = rgi(query).reshape(n_points, n_depths)
        result = xr.DataArray(
            values,
            dims=("points", "depth"),
            coords={"depth": point_coords["depth"]},
        )
    else:
        query = np.column_stack([point_coords[d] for d in active_dims])
        values = rgi(query)
        result = xr.DataArray(values, dims="points")

    if np.any(np.isnan(values)):
        warnings.warn(
            f"NaN values found in sampled mantle data ({da.name}). "
            "Check that all points are within the dataset bounds."
        )

    return result.rename(da.name)


def sample_mantle_var(
    da: xr.DataArray,
    lons: ArrayLike = None,
    lats: ArrayLike = None,
    times: ArrayLike = None,
    depths: ArrayLike = None,
    method: str = "linear",
) -> pd.DataFrame:
    """Sample/interpolate mantle data at fixed (time, depth, lat, lon) coordinates."""
    # Check if available coordinates match those of the data array; if not, raise an error
    coord_map = {"lon": lons, "lat": lats, "time": times, "depth": depths}
    for coord in da.dims:
        if coord_map.get(coord) is None:
            raise ValueError(
                f"Data array '{da.name}' has dimension '{coord}' but no corresponding coordinates were provided."
            )

    result = pd.DataFrame(_do_interp_cyclic_lon(
        da=da,
        lons=lons,
        lats=lats,
        times=times,
        depths=depths,
        broadcast_depth=False,
        method=method,
    ).to_pandas())

    result.columns = [f"{da.name.lower()} ({da.attrs.get("units", "unitless")})"]

    return result


def sample_mantle_var_depths(
    da: xr.DataArray,
    lons: ArrayLike = None,
    lats: ArrayLike = None,
    times: ArrayLike = None,
    depths: ArrayLike = DEFAULT_DEPTHS,
    method: str = "linear",
) -> pd.DataFrame:
    """Sample/interpolate mantle data at (time, lat, lon) coordinates, broadcasting depth."""
    # Check if available coordinates match those of the data array; if not, raise an error
    coord_map = {"lon": lons, "lat": lats, "time": times, "depth": depths}
    for coord in da.dims:
        if coord_map.get(coord) is None:
            raise ValueError(
                f"Data array '{da.name}' has dimension '{coord}' but no corresponding coordinates were provided."
            )

    result = pd.DataFrame(_do_interp_cyclic_lon(
        da=da,
        lons=lons,
        lats=lats,
        times=times,
        depths=depths,
        broadcast_depth=True,
        method=method,
    ).to_pandas())

    if "depth" not in da.dims:
        raise ValueError(f"Data array '{da.name}' has no 'depth' dimension, cannot sample over depths.")
    result.columns = [
        f"{da.name.lower()}_{depth:.0f}km ({da.attrs.get('units', 'unitless')})"
        for depth in depths
    ]

    return result


def _make_per_point_depths(
    ref_depths: np.ndarray,
    slice_size: float,
    n_depth_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-point depth sample arrays for profile extraction.

    Returns:
        depths_flat : (N*S,) absolute depths for RGI query
        offsets     : (S,) relative offsets from reference, km
    """
    offsets = np.linspace(0.0, slice_size, n_depth_samples)     # (S,)
    depths_2d = ref_depths[:, None] + offsets[None, :]          # (N, S) via broadcasting
    return depths_2d.ravel(), offsets


def sample_mantle_var_depth_profile(
    da: xr.DataArray,
    lons: ArrayLike,
    lats: ArrayLike,
    times: ArrayLike,
    ref_depths: np.ndarray,
    slice_size: float,
    n_depth_samples: int,
    method: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """Sample `da` along a per-point depth profile.

    For each point i, depths are sampled at ref_depths[i] + linspace(0, slice_size, S).

    Returns:
        profiles : (N, S) float array — sampled values
        offsets  : (S,) float array — depth offsets from reference, km
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    times = np.asarray(times, dtype=float)

    depths_flat, offsets = _make_per_point_depths(ref_depths, slice_size, n_depth_samples)
    lons_flat = np.repeat(lons, n_depth_samples)
    lats_flat = np.repeat(lats, n_depth_samples)
    times_flat = np.repeat(times, n_depth_samples)

    result = _do_interp_cyclic_lon(
        da, lons_flat, lats_flat, times_flat, depths_flat,
        broadcast_depth=False,
        method=method,
    )
    profiles = result.values.reshape(len(ref_depths), n_depth_samples)
    return profiles, offsets


def sample_mantle_var_depth_profile_interp(
    da: xr.DataArray,
    floor_lons: ArrayLike,
    floor_lats: ArrayLike,
    floor_times: ArrayLike,
    ceil_lons: ArrayLike,
    ceil_lats: ArrayLike,
    ceil_times: ArrayLike,
    alpha: np.ndarray,
    ref_depths_floor: np.ndarray,
    ref_depths_ceil: np.ndarray,
    slice_size: float,
    n_depth_samples: int,
    method: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """Sample depth profiles at both brackets and linearly interpolate.

    ref_depths_floor and ref_depths_ceil must be pre-computed by the caller
    (e.g. by interpolating LAB_Depth at bracket coords) so that a single LAB
    interpolation covers all variables sharing the same reference.

    Floor and ceil profiles are sampled on different absolute depth grids
    (ref_floor + offsets vs ref_ceil + offsets) before blending on the shared
    offset axis — consistent with all other temporal interpolation in this codebase.

    Returns:
        profiles : (N, S) float array — alpha-blended profiles
        offsets  : (S,) float array — depth offsets from reference, km
    """
    floor_profiles, offsets = sample_mantle_var_depth_profile(
        da, floor_lons, floor_lats, floor_times,
        ref_depths_floor, slice_size, n_depth_samples, method,
    )
    ceil_profiles, _ = sample_mantle_var_depth_profile(
        da, ceil_lons, ceil_lats, ceil_times,
        ref_depths_ceil, slice_size, n_depth_samples, method,
    )
    profiles = floor_profiles + alpha[:, None] * (ceil_profiles - floor_profiles)
    return profiles, offsets


def sample_mantle_var_interp(
    da: xr.DataArray,
    floor_lons: ArrayLike,
    floor_lats: ArrayLike,
    floor_times: ArrayLike,
    ceil_lons: ArrayLike,
    ceil_lats: ArrayLike,
    ceil_times: ArrayLike,
    alpha: np.ndarray,
    depths: ArrayLike | None = None,
) -> pd.DataFrame:
    """Sample da at bracket positions and linearly interpolate to actual age.

    alpha = (actual_time - floor_time) / (ceil_time - floor_time), pre-computed
    by validate_brackets. Pass depths to broadcast over depth levels.
    """
    if depths is None:
        v_floor = sample_mantle_var(da, floor_lons, floor_lats, floor_times)
        v_ceil  = sample_mantle_var(da, ceil_lons,  ceil_lats,  ceil_times)
    else:
        v_floor = sample_mantle_var_depths(da, floor_lons, floor_lats, floor_times, depths)
        v_ceil  = sample_mantle_var_depths(da, ceil_lons,  ceil_lats,  ceil_times,  depths)
    return v_floor + alpha[:, None] * (v_ceil.values - v_floor.values)


def sample_LAB_depths(
    ds: xr.Dataset,
    da: xr.DataArray,
    times: ArrayLike=None,
    lats: ArrayLike=None,
    lons: ArrayLike=None,
    offset_km: float=0.0,
) -> pd.DataFrame:
    """
    Sample mantle variable at the depth of the lithosphere-asthenosphere boundary.

    `offset_km` allows sampling at a fixed depth below the LAB (positive) or above it (negative).
    """

    coords = {
        "lons": lons,
        "lats": lats,
        "times": times,
    }

    lab_depths = np.asarray(sample_mantle_var(
        da=variables.get("LAB_Depth", ds),
        **coords,
        method="linear",
    )).flatten()  # Flatten to 1D array to use as depth coordinate

    result = sample_mantle_var(
        da=da,
        **coords,
        depths=lab_depths + offset_km,
        method="nearest",
    )

    result.columns = [
        f"{da.name.lower()}_LAB_{offset_km:+.0f}km ({da.attrs.get('units', 'unitless')})"
        if offset_km != 0 else
        f"{da.name.lower()}_LAB ({da.attrs.get('units', 'unitless')})"
        ]

    return result


def sample_LAB_depths_interp(
    ds: xr.Dataset,
    da: xr.DataArray,
    floor_lons: ArrayLike,
    floor_lats: ArrayLike,
    floor_times: ArrayLike,
    ceil_lons: ArrayLike,
    ceil_lats: ArrayLike,
    ceil_times: ArrayLike,
    alpha: np.ndarray,
    offset_km: float = 0.0,
) -> pd.DataFrame:
    """Sample da at LAB depth using bracket interpolation."""
    v_floor = sample_LAB_depths(ds, da, floor_times, floor_lats, floor_lons, offset_km)
    v_ceil  = sample_LAB_depths(ds, da, ceil_times,  ceil_lats,  ceil_lons,  offset_km)
    return v_floor + alpha[:, None] * (v_ceil.values - v_floor.values)


def calculate_lambdas(
    da: xr.DataArray,
    times: ArrayLike=None,
    lats: ArrayLike=None,
    lons: ArrayLike=None,
    depth_range: tuple[float, float] | None = None,
    n_lambdas: int=5,
) -> pd.DataFrame:
    """Calculate the first `n_lambdas` polynomial regression coefficients for the given data and depth range."""

    if "depth" not in da.dims:
        raise ValueError(f"Data array '{da.name}' has no 'depth' dimension, cannot calculate lambdas.")

    d_min, d_max = depth_range or (float(da.depth.min()), float(da.depth.max()))

    # Select depth range for polynomial fitting using explicit bounds (works with descending coordinates)
    depths = np.asarray(da.sel(depth=(da.depth >= d_min) & (da.depth <= d_max)).depth.values, dtype=float)

    if depths.size == 0:
        raise ValueError(
            f"No depths found in range [{d_min}, {d_max}] km. "
            f"Dataset depth range: [{float(da.depth.min()):.1f}, {float(da.depth.max()):.1f}] km"
        )

    # Sample with broadcast_depth=True to get (n_points, n_depths) matrix
    sampled = _do_interp_cyclic_lon(
        da=da,
        times=times,
        lats=lats,
        lons=lons,
        depths=depths,
        broadcast_depth=True,
        method="linear",
    )  # shape (n_points, n_depths)

    depth_profiles = sampled.to_numpy().T  # shape (n_depths, n_points)

    # Fit a polynomial of degree n_lambdas-1 to each depth profile and return the coefficients as features
    lambda_coeffs = np.polyfit(depths, depth_profiles, deg=n_lambdas-1) # shape (n_lambdas, n_points)

    lambda_cols = [f"{da.name.lower()}_lambda_{i} ({da.attrs.get('units', 'unitless')})" for i in reversed(range(n_lambdas))]
    result = pd.DataFrame(lambda_coeffs.T, columns=lambda_cols)  # shape (n_points, n_lambdas)

    return result
