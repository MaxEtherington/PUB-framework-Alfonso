"""Grid feature registry for sampling mantle and plate model data.

Provides decorator-based registration and sampling of gridded features
at point locations (lon, lat, time).

A valid feature is defined by a sampler function that takes arrays of longitudes,
latitudes, and times as input and returns a pandas Series or DataFrame of sampled 
values. Optionally, a coordinate resolver function can be provided to specify how 
to obtain the coordinates for sampling (e.g., input present-day deposit coords vs 
reconstructed to birth time).

Registered features are cached in a DataFrame to avoid redundant sampling, and can
be extracted for a given point dataset and context (e.g., mantle data directory, 
plate reconstruction) using the `extract` method.
"""
import warnings
from dataclasses import dataclass
from typing import Callable
from functools import partial

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import xarray as xr
import pint_xarray  # noqa
import gplately as gpl

import lib.mantle_variables as mv  # noqa
from .mantle_variables import variables, sample_mantle_var, sample_mantle_var_depths, sample_LAB_depths, calculate_lambdas

# ==================
# Grid feature registry
# ==================


@dataclass
class GridFeature:
    """A registered grid feature with a sampler function and metadata."""
    name: str
    sampler: Callable[[np.ndarray, np.ndarray, np.ndarray], pd.DataFrame]
    coordinate_resolver: Callable | None = None


class GridFeatureRegistry:
    def __init__(self):
        self._features: dict[str, GridFeature] = {}
        self._resolved_coordinates: dict[Callable, tuple] = {}
        self._results: pd.DataFrame | None = None
        self._mantle_dataset: xr.Dataset | None = None
        # Injected at runtime from notebook
        self._point_data: pd.DataFrame | None = None
        self._plate_reconstruction: gpl.PlateReconstruction | None = None
        self._valid_time_window: tuple[int, int] | None = None  # (min, max)
        self.mantle_data_dir: str | None = None

    @property
    def point_data(self) -> pd.DataFrame:
        if self._point_data is None:
            raise RuntimeError("point_data has not been set on the feature registry.")
        return self._point_data
    @point_data.setter
    def point_data(self, df: pd.DataFrame):
        if not all(col in df.columns for col in ["lon", "lat", "age (Ma)", "present_lon", "present_lat"]):
            raise ValueError("point_data must contain columns: 'lon', 'lat', 'age (Ma)', 'present_lon', 'present_lat'")
        if self._point_data is not None:
            if not df.equals(self._point_data):
                warnings.warn("Overwriting existing point_data with new DataFrame. Resetting cached results.", stacklevel=2)
                self._results = None
        self._point_data = df

    @property
    def plate_reconstruction(self) -> gpl.PlateReconstruction:
        if self._plate_reconstruction is None:
            raise RuntimeError("plate_reconstruction has not been set on the feature registry.")
        return self._plate_reconstruction
    @plate_reconstruction.setter
    def plate_reconstruction(self, recon: gpl.PlateReconstruction):
        self._plate_reconstruction = recon

    @property
    def valid_time_window(self) -> tuple[float, float]:
        if self._valid_time_window is None:
            if self.mantle_dataset is None:
                raise RuntimeError("No mantle dataset is loaded; registry valid_time_window must be set manually.")
            ds_times = [self.mantle_dataset.time.values, self.point_data["age (Ma)"].values]
            self._valid_time_window = (
                max(ds.min() for ds in ds_times),
                min(ds.max() for ds in ds_times),
            )
        return self._valid_time_window
    @valid_time_window.setter
    def valid_time_window(self, value: tuple[float, float]):
        self._valid_time_window = tuple(sorted(value))

    @property
    def mantle_dataset(self) -> xr.Dataset:
        if (self._mantle_dataset is None
            or hash(self.mantle_data_dir.resolve()) != getattr(self, "_mantle_data_dir_hash", None)
        ):
            if self.mantle_data_dir is None:
                raise RuntimeError("mantle_data_dir has not been set on the feature registry.")
            self._mantle_dataset = xr.open_mfdataset(
                sorted(self.mantle_data_dir.glob("*.nc")),
                combine='nested',
                concat_dim='time',
                chunks={"time": 1, "depth": 25},
            )
            self._mantle_data_dir_hash = hash(self.mantle_data_dir.resolve())
            self.reset()  # Clear cached results when loading new dataset
        return self._mantle_dataset

    # ── Registration ──────────────────────────────────────────────────────────
    # TODO: upgrade registration so that it wraps output functions such that,
    # when run, they will use self.get() unless the coordinates provided do not match
    # those previously in the function definition.
    def register(self, name: str, coords: Callable | None = None):
        """Decorator to register a grid feature sampler function."""
        def decorator(fn):
            self._features[name] = GridFeature(
                name=name,
                sampler=fn,
                coordinate_resolver=coords,
            )
            return fn
        return decorator

    def register_batch(
        self,
        declares: list[str] | str | None = None,
        coords: Callable | None = None,
        probe: bool = True,
    ):
        """Decorator to register a batch producer as one or more features."""
        def decorator(fn):
            feature_properties = {"sampler": fn, "coordinate_resolver": coords}
            try:
                if not probe:
                    raise RuntimeError("Batch producer registration skipped probing as requested.")

                probe_df = fn(
                    lons=np.array([0.0]),
                    lats=np.array([0.0]),
                    times=np.array([0.0]),
                )

                if not isinstance(probe_df, pd.DataFrame):
                    raise TypeError(
                        f"Batch producer '{fn.__name__}' returned {type(probe_df).__name__}; expected pandas.DataFrame."
                    )
                for col in probe_df.columns:
                    self._features[str(col)] = GridFeature(
                        name=str(col),
                        **feature_properties
                    )
            except Exception as exc:
                if probe and not isinstance(exc, RuntimeError):
                    warnings.warn(
                        f"Probing batch producer '{fn.__name__}' raised {type(exc).__name__}: {exc}. "
                        "Falling back to declares — check sampler for bugs.",
                        stacklevel=3,
                    )
                if isinstance(declares, list):
                    for name in declares:
                        self._features[str(name)] = GridFeature(
                            name=str(name),
                            **feature_properties
                        )
                elif isinstance(declares, str):
                    self._features[declares] = GridFeature(
                        name=declares,
                        **feature_properties
                    )
                elif declares is None:
                    raise RuntimeError(
                        f"Unable to register batch producer '{fn.__name__}': probing failed and/or no declarations were provided. "
                        "Provide declares as a list of output names or a placeholder string."
                    ) from exc
                else:
                    raise RuntimeError(
                        f"Invalid declares value for batch producer '{fn.__name__}': {declares!r}. "
                        "Expected list[str], str, or None."
                    ) from exc

            return fn

        return decorator

    # ── Retrieval ─────────────────────────────────────────────────────────────

    @property
    def available(self) -> list[str]:
        """Return list of available feature names."""
        return list(self._features)

    def _get_gridfeature(self, name: str) -> GridFeature:
        """Return the GridFeature object (without calling it)."""
        if name not in self._features:
            raise KeyError(f"Unknown feature: '{name}'. Available: {list(self._features)}")
        return self._features[name]

    def get_coordinates(self, resolver: Callable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get resolved coordinates for a given resolver, using cache if available."""
        if resolver in self._resolved_coordinates:
            resolved_coordinates = self._resolved_coordinates[resolver]
        else:
            resolved_coordinates = resolver()
            self._resolved_coordinates[resolver] = resolved_coordinates
        return resolved_coordinates

    def get(
        self,
        name: str,
        *,
        coords: tuple[ArrayLike, ArrayLike, ArrayLike] | None=None,
        coordinate_resolver=None,
    ) -> pd.Series | pd.DataFrame:
        """Get a feature and cache result columns in ``_results``.

        If ``coords`` or ``coordinate_resolver`` is provided, the feature is
        sampled at those coordinates instead of the registered resolver and
        results are not cached. ``coords`` takes precedence over
        ``coordinate_resolver`` if both are supplied.
        """
        override = coords is not None or coordinate_resolver is not None
        feature = self._get_gridfeature(name)

        # Fetch cached result if available
        if not override:
            if self._results is not None and name in self._results.columns:
                return self._results[name]

        #  Get coordinate resolver for sampling
        if override:
            if coords is not None:
                resolved_coordinates = coords
            else:  # coordinate_resolver is not None
                resolved_coordinates = coordinate_resolver()
        else:
            coordinate_resolver = feature.coordinate_resolver or reconstructed
            resolved_coordinates = self.get_coordinates(coordinate_resolver)

        result = feature.sampler(*resolved_coordinates)

        is_placeholder = False
        if isinstance(result, pd.Series):
            col_name = result.name or name
            result_df = result.rename(col_name).to_frame()
        else:
            result_df = result
            col_names = [str(c) for c in result_df.columns]
            for col_name in col_names:
                self._features[col_name] = GridFeature(
                    name=col_name,
                    sampler=feature.sampler,
                    coordinate_resolver=feature.coordinate_resolver,
                )
            is_placeholder = name not in col_names
            if is_placeholder:
                self._features.pop(name, None)

        if not override:
            if self._results is None:
                self._results = result_df.copy()
            else:
                new_cols = [c for c in result_df.columns if c not in self._results.columns]
                if new_cols:
                    self._results = pd.concat([self._results, result_df[new_cols]], axis=1)

            if isinstance(result, pd.Series):
                return self._results[col_name]
            elif is_placeholder:
                return self._results[result_df.columns]
            return self._results[name]
        else:
            if isinstance(result, pd.Series):
                return result_df[col_name]
            elif is_placeholder:
                return result_df[result_df.columns]
            return result_df[name]


    def compute_all(
        self,
        feature_names: list[str] | None = None,
    ) -> None:
        """Populate ``_results`` by sampling all registered features."""
        if feature_names is None:
            feature_names = self.available
        for name in feature_names:
            try:
                self.get(name)
            except ValueError as e:
                warnings.warn(f"Could not calculate feature '{name}': {e}", stacklevel=2)

    def extract(
        self,
        point_data,
        mantle_data_dir,
        plate_reconstruction,
        feature_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """Extract all features for the given point data and context, returning a coregistered DataFrame."""

        self.point_data = point_data
        self.mantle_data_dir = mantle_data_dir
        self.plate_reconstruction = plate_reconstruction

        self.compute_all(feature_names)

        new_cols = [c for c in self._results.columns if c not in point_data.columns]
        return point_data.join(self._results[new_cols])

    def reset(self):
        """Clear cached results."""
        self._results = None

    # ── Plotting ─────────────────────────────────────────────────────────────

    def grid_sample(self, feature_names: list[str] | str, times: list[float] | float) -> xr.Dataset:
        """Sample a feature on a regular global grid at a specified time."""
        if isinstance(feature_names, str):
            feature_names = [feature_names]
        if isinstance(times, float | int):
            times = [times]

        old_point_data = self._point_data.copy() if self._point_data is not None else None
        old_results = self._results.copy() if self._results is not None else None

        lons = np.arange(-180, 180, 1)
        lats = np.arange(-90, 91, 1)
        grid_lons, grid_lats = np.meshgrid(lons, lats)
        flat_lons = grid_lons.flatten()
        flat_lats = grid_lats.flatten()

        plate_ages = gpl.Points(self.plate_reconstruction, lons=flat_lons, lats=flat_lats, age=None).age

        time_slices = {name: [] for name in feature_names}

        try:
            for time in times:
                self.reset()
                mask = plate_ages >= time
                valid_indices = np.nonzero(mask)[0]

                self._point_data = pd.DataFrame({
                    "lon": flat_lons[valid_indices],
                    "lat": flat_lats[valid_indices],
                    "age (Ma)": np.full(len(valid_indices), float(time)),
                    "present_lon": flat_lons[valid_indices],
                    "present_lat": flat_lats[valid_indices],
                })

                for name in feature_names:
                    result = self.get(name)
                    da = np.full(len(flat_lons), np.nan, dtype=np.float32)
                    da[valid_indices] = result.values
                    time_slices[name].append(da.reshape(len(lats), len(lons)))
        finally:
            self._point_data = old_point_data
            self._results = old_results

        ds = xr.Dataset(
            {
                name: xr.DataArray(
                    np.stack(time_slices[name], axis=0),
                    coords={"time": times, "lat": lats, "lon": lons},
                    dims=["time", "lat", "lon"],
                    name=name,
                )
                for name in feature_names
            }
        )

        return ds


features = GridFeatureRegistry()


# ==================
# Coordinate resolvers
# ==================

def _snap_to_valid_times(
    times: np.ndarray,
    valid_times: np.ndarray,
) -> np.ndarray:
    """Snap times to nearest valid timestep. Ties snap to older time."""
    valid_times = np.sort(valid_times)
    midpoints = (valid_times[:-1] + valid_times[1:]) / 2
    return valid_times[np.digitize(times, midpoints)]


def _snap_and_reconstruct_points(
    present_lons: np.ndarray,
    present_lats: np.ndarray,
    times: np.ndarray,
    valid_times: np.ndarray,
    return_point_indices: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Snap birth `times` to nearest reconstructible `valid_times`, then reconstruct.

    Returns (rlons, rlats, snapped_times, point_indices) — all indexed to surviving points.
    """
    recon = features.plate_reconstruction

    times = np.asarray(times)
    valid_times = np.asarray(valid_times)

    plate_ages = gpl.Points(recon, present_lons, present_lats, 0.0, age=None).age

    snapped_times = _snap_to_valid_times(times, valid_times)
    snapped_times = np.where(
        snapped_times > plate_ages,
        valid_times[np.clip(np.digitize(times, valid_times) - 1, 0, len(valid_times) - 1)],
        snapped_times)

    points = gpl.Points(recon, present_lons, present_lats, 0.0, age=snapped_times)
    rlons, rlats, point_indices = points.reconstruct_to_birth_age(snapped_times, return_point_indices=True)

    if present_lons.size != point_indices.size or present_lats.size != point_indices.size:
        raise RuntimeError("Reconstruction output arrays have inconsistent sizes.")

    if return_point_indices:
        return rlons, rlats, snapped_times, point_indices
    return rlons, rlats, snapped_times


def snap_to_plate_model(
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Present-day lat/lon; birth times of points snapped to nearest valid reconstruction time (1 Myr timesteps).

    Only surviving points (those that reconstruct successfully) are returned, matching the filtering
    applied by snap_to_mantle so that both resolvers always produce the same number of outputs.
    """
    lons, lats, times = features.point_data[["present_lon", "present_lat", "age (Ma)"]].values.T
    min_time, max_time = features.valid_time_window
    valid_times = np.arange(int(min_time), int(max_time) + 1, 1)
    _, _, times = _snap_and_reconstruct_points(
        present_lons=lons,
        present_lats=lats,
        times=times,
        valid_times=valid_times,
    )
    return lons, lats, times


def snap_to_mantle(
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    lat/lon reconstructed to birth times of points, snapped to nearest valid mantle output time (~10 Myr timesteps).
    Returns:
    - lons, lats, times: arrays of present-day longitudes, latitudes, and snapped birth times for points that successfully reconstruct.
    """
    lons, lats, times = features.point_data[["present_lon", "present_lat", "age (Ma)"]].values.T
    valid_times = features.mantle_dataset["time"].values
    lons, lats, times = _snap_and_reconstruct_points(
        present_lons=lons,
        present_lats=lats,
        times=times,
        valid_times=valid_times,
    )
    return lons, lats, times


def snap_to_mantle_present(
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Present-day lat/lon; birth times of points snapped to nearest valid mantle output time (~10 Myr timesteps).
    Returns:
    - present_lons, present_lats, times: arrays of present-day longitudes, latitudes, and snapped birth times for points that successfully reconstruct.
    """
    present_lons, present_lats, times = features.point_data[["present_lon", "present_lat", "age (Ma)"]].values.T
    min_time, max_time = features.valid_time_window
    valid_times = features.mantle_dataset["time"].values
    _, _, times = _snap_and_reconstruct_points(
        present_lons=present_lons,
        present_lats=present_lats,
        times=times,
        valid_times=valid_times,
    )
    return present_lons, present_lats, times


def reconstructed(
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """lat/lon reconstructed to birth times of points"""
    lons, lats, times = features.point_data[["lon", "lat", "age (Ma)"]].values.T
    return lons, lats, times


def present_day(
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Present-day lat/lon, birth times of points"""
    lons, lats, times = features.point_data[["present_lon", "present_lat", "age (Ma)"]].values.T
    return lons, lats, times


def _offset_coordinates_by_time(
    coordinate_resolver: Callable,
    n_timesteps: int = 1,
    return_point_indices: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resolve coordinates by reconstructing points to a different time step.
    By default, resolves coordinates backwards.
    To resolve forwards, use a negative time_offset and set n_timesteps to 
    the number of steps forward to resolve.
    """
    present_lons, present_lats = features.point_data[["present_lon", "present_lat"]].values.T
    _, _, times = coordinate_resolver()
    valid_times = np.sort(np.unique(times))[::-1] # Sort descending (e.g. 100 Ma, 90 Ma, ..., 10 Ma)

    # Find previous times by snapping current time indices back by n_timesteps
    previous_time_indices = np.clip(
        np.digitize(times, valid_times) - n_timesteps,
        0, len(valid_times) - 1)
    previous_times = valid_times[previous_time_indices]

    # Use Points to reconstruct points to timestep before birth age
    points = gpl.Points(
        features.plate_reconstruction,
        present_lons, present_lats,
        0.0,
    )
    if return_point_indices:
        offset_lons, offset_lats, point_indices = points.reconstruct_to_birth_age(
            previous_times, return_point_indices=True
        )
        return offset_lons, offset_lats, previous_times, point_indices
    else:
        offset_lons, offset_lats = points.reconstruct_to_birth_age(
            previous_times, return_point_indices=False
        )
        return offset_lons, offset_lats, previous_times


# ===========================
# Batch-register mantle vars
# ===========================

@features.register_batch(declares="Base_Mantle_Features", coords=snap_to_mantle, probe=False)
def _base_mantle_features_depths(
    lons: np.ndarray,
    lats: np.ndarray,
    times: np.ndarray,
) -> pd.DataFrame:
    """Sample a suite of basic mantle features at requested coordinates."""
    depths_to_sample = [100, 200, 300, 400]
    results = []

    vars_to_sample = [
        # 'FullTemperature_CG',
        # 'Pressure',
        'Radial_Velocity',
        # 'Temperature_CG',
        'Temperature_Deviation_CG',
        # 'Velocity_x',
        # 'Velocity_y',
        # 'Velocity_z',
        # 'Viscosity_CG',
        # 'East_Velocity',
        # 'North_Velocity',
        # 'Cell_Volume',
        'Speed',
        'Tangential_Speed',
        'Radial_Tangential_Ratio',

        'LAB_Depth',
        '1000K_Isotherm_Depth',
        'Sublithospheric_Cold_Anomaly_Thickness',
        'Cold_Anomaly_Magnitude',
        'Temperature_Deviation_Avg_0-400km',
        'Temperature_Deviation_Avg_0-400km_Rolling_30Ma',
        'Temperature_Deviation_Avg_0-400km_Rolling_50Ma',
        'Temperature_Deviation_Avg_100-400km',
        'Temperature_Deviation_Avg_100-400km_Rolling_30Ma',
        'Temperature_Deviation_Avg_100-400km_Rolling_50Ma',
        'Temperature_Deviation_Avg_LAB-120km',
        'Temperature_Deviation_Avg_LAB-120km_Rolling_30Ma',
        'Temperature_Deviation_Avg_LAB-120km_Rolling_50Ma',
        'Temperature_Deviation_Avg_LAB-160km',
        'Temperature_Deviation_Avg_LAB-160km_Rolling_30Ma',
        'Temperature_Deviation_Avg_LAB-160km_Rolling_50Ma',
        'Temperature_Deviation_Avg_LAB-200km',
        'Temperature_Deviation_Avg_LAB-200km_Rolling_30Ma',
        'Temperature_Deviation_Avg_LAB-200km_Rolling_50Ma',
        'Temperature_Deviation_Avg_LAB-300km',
        'Temperature_Deviation_Avg_LAB-300km_Rolling_30Ma',
        'Temperature_Deviation_Avg_LAB-300km_Rolling_50Ma',
        'Temperature_Deviation_Avg_LAB-400km',
        'Temperature_Deviation_Avg_LAB-400km_Rolling_30Ma',
        'Temperature_Deviation_Avg_LAB-400km_Rolling_50Ma'
    ]

    for var in vars_to_sample:
        da = variables.get(var, features.mantle_dataset)
        kwargs = {
            "da": da,
            "lons": lons,
            "lats": lats,
            "times": times,
            "depths": depths_to_sample,
            "method": "linear",
        }

        result = sample_mantle_var_depths(**kwargs) if "depth" in da.dims else sample_mantle_var(**kwargs)

        # Rename columns to include variable name and depth if applicable
        if len(result.columns) == len(depths_to_sample):  # Sampled across depths
            # Rename actual DataFrame columns
            result.columns = [f"{var}_{int(d)}km" for d in depths_to_sample]
        elif len(result.columns) == 1:  # Single depth or depth-independent
            result.columns = [var]
        else:
            raise ValueError(f"Unexpected number of columns in result for variable '{var}': {len(result.columns)}")

        results.append(result)

    return pd.concat(results, axis=1)


@features.register_batch(declares="Base_Mantle_Features_LAB", coords=snap_to_mantle, probe=False)
def _base_mantle_features_LAB(
    lons: np.ndarray,
    lats: np.ndarray,
    times: np.ndarray,
) -> pd.DataFrame:
    """Sample a suite of basic mantle features at requested coordinates, starting from the LAB."""
    offsets_to_sample = [0, 40, 80, 120]
    results = []

    vars_to_sample = [
        # 'FullTemperature_CG',
        # 'Pressure',
        'Radial_Velocity',
        # 'Temperature_CG',
        'Temperature_Deviation_CG',
        # 'Velocity_x',
        # 'Velocity_y',
        # 'Velocity_z',
        # 'Viscosity_CG',
        # 'East_Velocity',
        # 'North_Velocity',
        # 'Cell_Volume',
        'Speed',
        'Tangential_Speed',
        'Radial_Tangential_Ratio',
        # 'LAB_Depth',
        # '1000K_Isotherm_Depth',
        # 'Sublithospheric_Cold_Anomaly_Thickness',
        # 'Cold_Anomaly_Magnitude',
        # 'Temperature_Deviation_Avg_0-400km',
        # 'Temperature_Deviation_Avg_0-400km_Rolling_30Ma',
        # 'Temperature_Deviation_Avg_0-400km_Rolling_50Ma',
        # 'Temperature_Deviation_Avg_100-400km',
        # 'Temperature_Deviation_Avg_100-400km_Rolling_30Ma',
        # 'Temperature_Deviation_Avg_100-400km_Rolling_50Ma',
        # 'Temperature_Deviation_Avg_LAB-120km',
        # 'Temperature_Deviation_Avg_LAB-120km_Rolling_30Ma',
        # 'Temperature_Deviation_Avg_LAB-120km_Rolling_50Ma',
        # 'Temperature_Deviation_Avg_LAB-160km',
        # 'Temperature_Deviation_Avg_LAB-160km_Rolling_30Ma',
        # 'Temperature_Deviation_Avg_LAB-160km_Rolling_50Ma',
        # 'Temperature_Deviation_Avg_LAB-200km',
        # 'Temperature_Deviation_Avg_LAB-200km_Rolling_30Ma',
        # 'Temperature_Deviation_Avg_LAB-200km_Rolling_50Ma',
        # 'Temperature_Deviation_Avg_LAB-300km',
        # 'Temperature_Deviation_Avg_LAB-300km_Rolling_30Ma',
        # 'Temperature_Deviation_Avg_LAB-300km_Rolling_50Ma',
        # 'Temperature_Deviation_Avg_LAB-400km',
        # 'Temperature_Deviation_Avg_LAB-400km_Rolling_30Ma',
        # 'Temperature_Deviation_Avg_LAB-400km_Rolling_50Ma'
    ]

    for var in vars_to_sample:
        da = variables.get(var, features.mantle_dataset)
        new_columns = []
        for offset in offsets_to_sample:
            result = sample_LAB_depths(
                ds=features.mantle_dataset,
                da=da,
                lons=lons,
                lats=lats,
                times=times,
                offset_km=offset,
            )
            new_columns.append(result)
        result = pd.concat(new_columns, axis=1)
        # Rename columns to include variable name and depth if applicable
        if len(result.columns) == len(offsets_to_sample):  # Sampled across depths
            # Rename actual DataFrame columns
            result.columns = [f"{var}_LAB+{int(d)}km" for d in offsets_to_sample]
            # Append overall average column
        elif len(result.columns) == 1:  # Single depth or depth-independent
            result.columns = [f"{var}_LAB"]
        else:
            raise ValueError(f"Unexpected number of columns in result for variable '{var}': {len(result.columns)}")

        results.append(result)

    return pd.concat(results, axis=1)


# ==================
# Feature definitions
# ==================


@features.register_batch(declares=["east_plate_velocity (cm/yr)", "north_plate_velocity (cm/yr)"], coords=snap_to_plate_model)
def _plate_velocity_components(
    present_lons: np.ndarray,
    present_lats: np.ndarray,
    times: np.ndarray,
) -> pd.DataFrame:
    """Sample plate velocity components at requested coordinates.
    Returns `(velocity_east, velocity_north)` in cm/yr"""
    recon = features.plate_reconstruction

    east_vels = np.zeros(len(times), dtype=np.float64)
    north_vels = np.zeros(len(times), dtype=np.float64)

    for t in np.unique(times):
        indices_at_time = np.where(times == t)[0]
        points_at_time = gpl.Points(
            recon,
            present_lons[indices_at_time],
            present_lats[indices_at_time],
        )
        (
            east_vels[indices_at_time],
            north_vels[indices_at_time],
            successful_indices
        ) = points_at_time.plate_velocity(t, return_point_indices=True)

        mask = np.ones_like(indices_at_time, dtype=bool)
        mask[successful_indices] = False  # Mask points where velocity could not be calculated (e.g., points that do not fall on a plate at this time)
        for arr in (east_vels, north_vels):
            arr[indices_at_time[mask]] = np.nan

    return pd.DataFrame({
        "east_plate_velocity (cm/yr)": east_vels,
        "north_plate_velocity (cm/yr)": north_vels,
    })


@features.register("plate_acceleration (cm/yr/Myr)", coords=snap_to_plate_model)
def _plate_acceleration(
    present_lons: np.ndarray,
    present_lats: np.ndarray,
    times: np.ndarray,

    offset: float = 1.0,  # Myr; increment in the plate reconstruction over which to calculate acceleration
) -> pd.DataFrame:
    """
    Sample magnitude of plate acceleration at requested coordinates over a small time period.
    Default period = 1 Myr
    """
    v1 = _plate_velocity_components(
        present_lons=present_lons,
        present_lats=present_lats,
        times=times + offset,
    ).to_numpy()
    v2 = _plate_velocity_components(
        present_lons=present_lons,
        present_lats=present_lats,
        times=times,
    ).to_numpy()

    delta_v = (v2 - v1) / offset
    acceleration = np.linalg.norm(delta_v, axis=1)

    return pd.DataFrame(acceleration, columns=["plate_acceleration (cm/yr/Myr)"])


def _relative_tangential_velocity_LAB(
    lons: np.ndarray,
    lats: np.ndarray,
    times: np.ndarray,

    offset_km: float = 0,  # km; depth to sample mantle velocity below the LAB
) -> pd.DataFrame:
    """Sample relative tangential velocity between plate and mantle at requested coordinates."""

    def _fetch_tangential_velocity_at_LAB(var):
        da = variables.get(var, features.mantle_dataset)
        da = da.pint.quantify().pint.to({da.name: "cm/yr"}).pint.dequantify()
        return sample_LAB_depths(
            ds = features.mantle_dataset,
            da = da,
            lons=lons,
            lats=lats,
            times=times,
            offset_km=offset_km,
        ).to_numpy()

    v_mantle = np.column_stack([
        _fetch_tangential_velocity_at_LAB(variable)
        for variable in ("East_Velocity", "North_Velocity")
    ])
    v_plate = np.column_stack([
        features.get(feature).to_numpy()
        for feature in ("east_plate_velocity (cm/yr)", "north_plate_velocity (cm/yr)")
    ])

    delta_v = v_mantle - v_plate
    v_mag = np.linalg.norm(delta_v, axis=1)

    result = pd.DataFrame(
        np.column_stack([delta_v, v_mag]),
        columns=[
            f"mantle_relative_east_velocity_LAB_{offset_km}km (cm/yr)",
            f"mantle_relative_north_velocity_LAB_{offset_km}km (cm/yr)",
            f"mantle_relative_speed_LAB_{offset_km}km (cm/yr)"
        ]
    )
    return result



def _calculate_plate_frame_velocity_components(
) -> (np.ndarray, np.ndarray):
    """
    Samples the relative velocity in the reference frame of the overlying plate. 
    Returns plate-parallel velocity and plate-transverse velocity. 
    Plate-parallel velocity is positive in the direction of plate motion, negative in the opposite direction.
    Plate transverse velocity is positive to the right of the direction of plate motion, negative to the left.
    """
    v_east = features.get("east_plate_velocity (cm/yr)").to_numpy()
    v_north = features.get("north_plate_velocity (cm/yr)").to_numpy()

    v = np.column_stack([v_east, v_north])
    v_mag = np.linalg.norm(v, axis=1)

    with np.errstate(invalid='ignore', divide='ignore'):
        v_hat_parallel = np.where(v_mag[:, None] > 0, v / v_mag[:, None], 0)
    rotation_matrix = np.array([[0, 1], [-1, 0]])  # 90 degree rotation matrix to get plate transverse direction
    v_hat_transverse = v_hat_parallel @ rotation_matrix.T

    return v_hat_parallel, v_hat_transverse


def _relative_velocity_in_plate_frame(
    lons: np.ndarray,
    lats: np.ndarray,
    times: np.ndarray,

    offset_km: float = 0,  # km; depth to sample mantle velocity below the LAB for calculating plate frame velocity components
) -> pd.DataFrame:
    """Sample relative velocity between plate and mantle in the reference frame of the overlying plate."""
    v_rel = np.column_stack([
        features.get(f"mantle_relative_east_velocity_LAB_{offset_km}km (cm/yr)").to_numpy(),
        features.get(f"mantle_relative_north_velocity_LAB_{offset_km}km (cm/yr)").to_numpy(),
    ])

    v_hat_parallel, v_hat_transverse = _calculate_plate_frame_velocity_components()

    relative_parallel = np.vecdot(v_rel, v_hat_parallel, axis=1)
    relative_transverse = np.vecdot(v_rel, v_hat_transverse, axis=1)
    relative_speed = np.linalg.norm(np.column_stack([relative_parallel, relative_transverse]), axis=1)

    result = pd.DataFrame(
        np.column_stack([relative_parallel, relative_transverse, relative_speed]),
        columns=[
            f"relative_velocity_parallel_to_plate_LAB_{offset_km}km (cm/yr)",
            f"relative_velocity_transverse_to_plate_LAB_{offset_km}km (cm/yr)",
        ]
    )
    return result


@features.register_batch("Temperature_Deviation_Lambdas", coords=snap_to_mantle)
def _temperature_lambdas(
    lons: np.ndarray,
    lats: np.ndarray,
    times: np.ndarray,

    n_lambdas: int = 5,
) -> pd.DataFrame:
    """Calculate the first `n_lambdas` polynomial regression coefficients of the mantle geotherm."""

    result = calculate_lambdas(
        da=variables.get("Temperature_Deviation_CG", features.mantle_dataset),
        times=times,
        lats=lats,
        lons=lons,
        depth_range=(0, 400),
        n_lambdas=n_lambdas
    )

    return result

# ==================
# Feature transforms
# ==================

# Feature transforms accept a feature and registers a new feature based on it.


def _feature_LAB_depth_averages(
    lons,
    lats,
    times,
    sampler: Callable[[np.ndarray, np.ndarray, np.ndarray, float], pd.DataFrame],
    offset_depths: list[float] = [0, 40, 80, 120],
) -> pd.DataFrame:
    """
    Sample a depth-dependent feature at a series of depth offsets from the LAB, then return the average.
    """
    samples = []
    col_names = None
    for offset in offset_depths:
        sample_df = sampler(lons=lons, lats=lats, times=times, offset_km=offset)
        if col_names is None:
            col_names = sample_df.columns
        samples.append(sample_df.to_numpy())

    samples = np.stack(samples, axis=0)
    average = np.nanmean(samples, axis=0)

    result_df = pd.DataFrame(average, columns=col_names)
    return result_df


# ===========================
# Register derivative features
# ===========================


# LAB offsets
LAB_offsets = [0, 40, 80, 120, 160]
for LAB_offset in LAB_offsets:
    # Register relative tangential velocity features at multiple depths below the LAB
    features.register_batch(
        declares=[
            f"mantle_relative_east_velocity_LAB_{LAB_offset}km (cm/yr)",
            f"mantle_relative_north_velocity_LAB_{LAB_offset}km (cm/yr)",
            f"mantle_relative_speed_LAB_{LAB_offset}km (cm/yr)"
        ],
        coords=snap_to_mantle
    )(partial(_relative_tangential_velocity_LAB, offset_km=LAB_offset))

    # Register plate-relative tangential velocity features at multiple depths below the LAB
    features.register_batch(
        declares=[
            f"relative_velocity_parallel_to_plate_LAB_{LAB_offset}km (cm/yr)",
            f"relative_velocity_transverse_to_plate_LAB_{LAB_offset}km (cm/yr)",
        ],
        coords=snap_to_mantle
    )(partial(_relative_velocity_in_plate_frame, offset_km=LAB_offset))

# Register depth averages of mantle-relative velocity components
features.register_batch(
    declares=[
        f"mantle_relative_east_velocity_LAB_{LAB_offset}km_avg (cm/yr)",
        f"mantle_relative_north_velocity_LAB_{LAB_offset}km_avg (cm/yr)",
        f"mantle_relative_speed_LAB_{LAB_offset}km_avg (cm/yr)"
    ], coords=snap_to_mantle
    )(partial(_feature_LAB_depth_averages, sampler=_relative_tangential_velocity_LAB, offset_depths=LAB_offsets)
)

features.register_batch(
    declares=[
        f"relative_velocity_parallel_to_plate_LAB_{LAB_offset}km (cm/yr)_avg",
        f"relative_velocity_transverse_to_plate_LAB_{LAB_offset}km (cm/yr)_avg",
    ], coords=snap_to_mantle
    )(partial(
        _feature_LAB_depth_averages, sampler=_relative_velocity_in_plate_frame,
        offset_depths=LAB_offsets
    )
)



# Hardcoded because these live in the Base_Mantle_Features placeholder batch and aren't
# individually resolvable in features.available until the batch sampler is first called.
base_mantle_features = [
    'LAB_Depth',
    '1000K_Isotherm_Depth',
    'Sublithospheric_Cold_Anomaly_Thickness',
    'Cold_Anomaly_Magnitude',
    'Temperature_Deviation_Avg_0-400km',
    'Temperature_Deviation_Avg_0-400km_Rolling_30Ma',
    'Temperature_Deviation_Avg_0-400km_Rolling_50Ma',
    'Temperature_Deviation_Avg_100-400km',
    'Temperature_Deviation_Avg_100-400km_Rolling_30Ma',
    'Temperature_Deviation_Avg_100-400km_Rolling_50Ma',
    'Temperature_Deviation_Avg_LAB-120km',
    'Temperature_Deviation_Avg_LAB-120km_Rolling_30Ma',
    'Temperature_Deviation_Avg_LAB-120km_Rolling_50Ma',
    'Temperature_Deviation_Avg_LAB-160km',
    'Temperature_Deviation_Avg_LAB-160km_Rolling_30Ma',
    'Temperature_Deviation_Avg_LAB-160km_Rolling_50Ma',
    'Temperature_Deviation_Avg_LAB-200km',
    'Temperature_Deviation_Avg_LAB-200km_Rolling_30Ma',
    'Temperature_Deviation_Avg_LAB-200km_Rolling_50Ma',
    'Temperature_Deviation_Avg_LAB-300km',
    'Temperature_Deviation_Avg_LAB-300km_Rolling_30Ma',
    'Temperature_Deviation_Avg_LAB-300km_Rolling_50Ma',
    'Temperature_Deviation_Avg_LAB-400km',
    'Temperature_Deviation_Avg_LAB-400km_Rolling_30Ma',
    'Temperature_Deviation_Avg_LAB-400km_Rolling_50Ma',
    'Temperature_Deviation_Lambdas',
]

@features.register_batch(
    declares=[f"{fn.replace(' ', '_')}_delta" for fn in base_mantle_features if fn != 'Temperature_Deviation_Lambdas'],
    coords=snap_to_mantle,
    probe=False,
)
def _mantle_variable_deltas(lons: np.ndarray, lats: np.ndarray, times: np.ndarray) -> pd.DataFrame:
    offset_lons, offset_lats, offset_times = _offset_coordinates_by_time(snap_to_mantle, n_timesteps=1)
    current = _base_mantle_features_depths(lons, lats, times)
    previous = _base_mantle_features_depths(offset_lons, offset_lats, offset_times)
    return pd.DataFrame({
        f"{fn.replace(' ', '_')}_delta": current[fn].values - previous[fn].values
        for fn in base_mantle_features if fn != 'Temperature_Deviation_Lambdas'
    })


@features.register_batch(
    declares="Temperature_Deviation_Lambdas_delta",
    coords=snap_to_mantle,
    probe=False,
)
def _temperature_lambdas_delta(lons: np.ndarray, lats: np.ndarray, times: np.ndarray) -> pd.DataFrame:
    offset_lons, offset_lats, offset_times = _offset_coordinates_by_time(snap_to_mantle, n_timesteps=1)
    current = _temperature_lambdas(lons, lats, times)
    previous = _temperature_lambdas(offset_lons, offset_lats, offset_times)
    delta_df = current.subtract(previous.values)
    delta_df.columns = [f"{col.replace(' ', '_')}_delta" for col in current.columns]
    return delta_df



