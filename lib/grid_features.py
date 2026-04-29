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

import numpy as np
import pandas as pd
import xarray as xr
import pint_xarray  # noqa
import gplately as gpl

from .mantle_variables import variables, sample_mantle_var, sample_mantle_var_depths, sample_LAB_depths, calculate_lambdas

# ==================
# Grid feature registry
# ==================


@dataclass
class GridFeature:
    """A registered grid feature with a sampler function and metadata."""
    name: str
    sampler: Callable[[np.ndarray, np.ndarray, np.ndarray], pd.Series | pd.DataFrame]
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
        if self._mantle_dataset is None:
            if self.mantle_data_dir is None:
                raise RuntimeError("mantle_data_dir has not been set on the feature registry.")
            self._mantle_dataset = xr.open_mfdataset(
                sorted(self.mantle_data_dir.glob("*.nc")),
                combine='nested',
                concat_dim='time',
                chunks={"time": 1, "depth": 25},
            )
        return self._mantle_dataset
    
    # ── Registration ──────────────────────────────────────────────────────────

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

    def get(
        self,
        name: str,
    ) -> pd.Series | pd.DataFrame:
        """Get a feature and cache result columns in ``_results``."""
        feature = self._get_gridfeature(name)

        # Check if this feature has already been sampled and cached in results
        if self._results is not None:
            if name in self._results.columns:
                return self._results[name]
        
        coordinate_resolver = feature.coordinate_resolver or reconstructed  # Default to reconstructed coordinates if no resolver specified
        if coordinate_resolver in self._resolved_coordinates:  # Cache resolved coordinates to avoid redundant computation across features that share the same resolver
            resolved_coordinates = self._resolved_coordinates[coordinate_resolver]
        else:
            resolved_coordinates = coordinate_resolver()
            self._resolved_coordinates[coordinate_resolver] = resolved_coordinates
        
        result = feature.sampler(*(resolved_coordinates))

        if isinstance(result, pd.Series):
            col_name = result.name or name
            result_df = result.rename(col_name).to_frame()
        else:
            result_df = result

            # If the declared name is not an actual output column, assume 
            ## it's a placeholder for all columns produced by this sampler.
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
# Sampling utilities
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
    """lat/lon reconstructed to birth times of points, snapped to nearest valid mantle output time (~10 Myr timesteps)"""
    lons, lats, times = features.point_data[["present_lon", "present_lat", "age (Ma)"]].values.T
    valid_times = features.mantle_dataset["time"].values
    lons, lats, times = _snap_and_reconstruct_points(
        present_lons=lons,
        present_lats=lats,
        times=times,
        valid_times=valid_times,
    )
    return lons, lats, times


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


# ==================
# Feature definitions
# ==================


@features.register_batch(declares="Base_Mantle_Features", coords=snap_to_mantle, probe=False)
def _base_mantle_features(
    lons: np.ndarray,
    lats: np.ndarray,
    times: np.ndarray,
) -> pd.DataFrame:
    """Sample a suite of basic mantle features at requested coordinates."""
    vars_to_exclude = {
        'FullTemperature_CG',
        # 'Pressure',
        # 'Radial_Velocity',
        'Temperature_CG',
        # 'Temperature_Deviation_CG',
        'Velocity_x',
        'Velocity_y',
        'Velocity_z',
        # 'Viscosity_CG',
        'East_Velocity',
        'North_Velocity',
        'Cell_Volume',
        # 'Speed',
        # 'Tangential_Speed',
        # 'Radial_Tangential_Ratio',
        # 'LAB_Depth',
        'Slab_Depth',
        # 'Temperature_Deviation_avg_0-400km',
        # 'Temperature_Deviation_avg_0-400km_rolling_30Ma',
        # 'Temperature_Deviation_avg_0-400km_rolling_50Ma',
        # 'Temperature_Deviation_avg_100-400km',
        # 'Temperature_Deviation_avg_100-400km_rolling_30Ma',
        # 'Temperature_Deviation_avg_100-400km_rolling_50Ma'
    }
    vars_to_sample = [v for v in variables.available if v not in vars_to_exclude]
    depths_to_sample = [100, 200, 300, 400]
    results = []

    for var in vars_to_sample:
        da = variables.get(var, features.mantle_dataset)
        result = sample_mantle_var_depths(
            da=da,
            lons=lons,
            lats=lats,
            times=times,
            depths=depths_to_sample,
            method="linear",
        )

        if isinstance(result, pd.DataFrame):
            # Rename actual DataFrame columns
            result = result.copy()
            result.columns = [f"{var}_{int(d)}km" for d in depths_to_sample]
        else:
            result = result.rename(var)

        results.append(result)

    return pd.concat(results, axis=1)


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
    
    
def _plate_velocity_magnitude(
    east_vels: pd.Series,
    north_vels: pd.Series,
) -> pd.Series:
    """Sample plate velocity magnitude at requested coordinates."""
    return np.sqrt(east_vels**2 + north_vels**2)


@features.register(registered_name:="plate_acceleration (cm/yr/Myr)", coords=snap_to_plate_model)
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
    
    return pd.DataFrame(acceleration, columns=[registered_name])


@features.register_batch(registered_names:=[
    "mantle_relative_east_velocity (cm/yr)", 
    "mantle_relative_north_velocity (cm/yr)",
    "mantle_relative_speed (cm/yr)"
    ], coords=snap_to_mantle)
def _relative_tangential_velocity(
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

    result = pd.DataFrame(np.column_stack([delta_v, v_mag]), columns=registered_names)    
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
        da=features.mantle_dataset.get("Temperature_Deviation"),  # TODO: Remove _CG once mantle data has been reprocessed
        times=times,
        lats=lats,
        lons=lons,
        depth_range=(0, 400),
        n_lambdas=n_lambdas
    )
    
    return result