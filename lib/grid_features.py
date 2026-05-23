"""Grid feature registry for sampling mantle and plate model data.

Provides decorator-based registration and sampling of gridded features
at point locations (lon, lat, time).

A valid feature is defined by a sampler function that takes arrays of longitudes,
latitudes, and times as input and returns a pandas DataFrame of sampled values.
Optionally, a coordinate resolver function can be provided to specify how to
obtain the coordinates for sampling (e.g., input present-day deposit coords vs
reconstructed to birth time).

Registered features are cached in a DataFrame to avoid redundant sampling, and can
be extracted for a given point dataset and context (e.g., set of mantle outputs,
plate reconstruction) using the `extract` method.
"""
import re as _re
import warnings
from dataclasses import dataclass
from typing import Callable
# from functools import partial

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import xarray as xr
import pint_xarray  # noqa
import gplately as gpl

import lib.mantle_variables as mv  # noqa
from .mantle_variables import (
    variables,
    sample_mantle_var, sample_mantle_var_interp,
    sample_mantle_var_depth_profile,
    sample_LAB_depths_interp,
)
from .depth_profile_stats import (
    DEPTH_PROFILE_STATS,
    DEPTH_PROFILE_VAR_UNITS,
    compute_depth_profile_stats,
    profile_stat_column_names,
)

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
        self._profile_cache: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        self._bracket_results: dict = {}
        self._raw_profile_cache: dict[tuple[float, float], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
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
                self._bracket_results = {}
                self._resolved_coordinates = {}
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
        verbose: bool = False,
    ) -> None:
        """Populate ``_results`` by sampling all registered features."""
        if feature_names is None:
            feature_names = self.available
        for name in feature_names:
            if verbose:
                print(f"Calculating feature '{name}'...")
            try:
                self.get(name)
            except ValueError as e:
                warnings.warn(f"Could not calculate feature '{name}': {e}", stacklevel=3)

    def validate_brackets(self, valid_times: np.ndarray, require_both: bool = True) -> None:
        """Filter point_data to points that reconstruct successfully at floor and/or ceil timesteps.

        For each point, compute the bracketing valid timesteps (floor ≤ age ≤ ceil) and attempt
        reconstruction at both. Points where the bracket time exceeds local plate age are treated
        as failures for that bracket without attempting reconstruction. Surviving points are kept
        in self._point_data with a new boolean column 'bracket_infilled'; failed points are dropped.
        Bracket reconstruction results are stored in self._bracket_results for future interpolation.
        """
        recon = self.plate_reconstruction
        lons = self._point_data["present_lon"].values
        lats = self._point_data["present_lat"].values
        times = self._point_data["age (Ma)"].values

        floor_times, ceil_times = _compute_floor_ceil_times(times, valid_times)

        # Degenerate case: times[i] == min(valid_times) gives floor == ceil == min.
        # Shift ceil to the next older step so the bracket spans a non-zero interval.
        exact_hit = floor_times == ceil_times
        if exact_hit.any():
            sorted_vt = np.sort(valid_times)
            hit_idx = np.searchsorted(sorted_vt, ceil_times[exact_hit], side="left")
            next_idx = np.clip(hit_idx + 1, 0, len(sorted_vt) - 1)
            ceil_times = ceil_times.copy()
            ceil_times[exact_hit] = sorted_vt[next_idx]

        plate_ages = gpl.Points(recon, lons, lats, 0.0, age=None).age

        plate_floor_valid = floor_times <= plate_ages
        plate_ceil_valid = ceil_times <= plate_ages

        rlons_floor, rlats_floor, recon_floor_mask = _try_reconstruct(lons, lats, floor_times)
        rlons_ceil, rlats_ceil, recon_ceil_mask = _try_reconstruct(lons, lats, ceil_times)

        floor_mask = plate_floor_valid & recon_floor_mask
        ceil_mask = plate_ceil_valid & recon_ceil_mask

        floor_only = floor_mask & ~ceil_mask
        ceil_only = ~floor_mask & ceil_mask
        neither = ~floor_mask & ~ceil_mask

        if require_both:
            drop_mask = neither | floor_only | ceil_only
            infill_mask = np.zeros(len(times), dtype=bool)
        else:
            drop_mask = neither
            infill_mask = floor_only | ceil_only
            rlons_floor = np.where(floor_only, rlons_ceil, rlons_floor)
            rlats_floor = np.where(floor_only, rlats_ceil, rlats_floor)
            floor_times = np.where(floor_only, ceil_times, floor_times)
            rlons_ceil = np.where(ceil_only, rlons_floor, rlons_ceil)
            rlats_ceil = np.where(ceil_only, rlats_floor, rlats_ceil)
            ceil_times = np.where(ceil_only, floor_times, ceil_times)

        n_infilled = int(infill_mask.sum())
        n_dropped = int(drop_mask.sum())
        if n_infilled > 0:
            infill_ages = times[infill_mask]
            warnings.warn(
                f"bracket_infilled: {n_infilled} point(s) had one bracket fail and were infilled "
                f"(mean age {infill_ages.mean():.1f} Ma, range {infill_ages.min():.1f}–{infill_ages.max():.1f} Ma).",
                stacklevel=2,
            )
        if n_dropped > 0:
            drop_ages = times[drop_mask]
            warnings.warn(
                f"bracket_dropped: {n_dropped} point(s) failed reconstruction at both brackets and were removed "
                f"(mean age {drop_ages.mean():.1f} Ma, range {drop_ages.min():.1f}–{drop_ages.max():.1f} Ma).",
                stacklevel=2,
            )

        keep_mask = ~drop_mask
        self._point_data = self._point_data[keep_mask].copy().reset_index(drop=True)
        self._point_data["bracket_infilled"] = infill_mask[keep_mask]

        t_span = ceil_times[keep_mask] - floor_times[keep_mask]
        alpha = np.where(t_span > 0, (times[keep_mask] - floor_times[keep_mask]) / t_span, 0.0)
        self._bracket_results[_BRACKET_KEY] = {
            "floor": (rlons_floor[keep_mask], rlats_floor[keep_mask], floor_times[keep_mask]),
            "ceil": (rlons_ceil[keep_mask], rlats_ceil[keep_mask], ceil_times[keep_mask]),
            "alpha": alpha,
        }

    def extract(
        self,
        point_data,
        mantle_data_dir,
        plate_reconstruction,
        feature_names: list[str] | None = None,
        verbose: bool = False,
        require_both_brackets: bool | None = True,
    ) -> pd.DataFrame:
        """Extract all features for the given point data and context, returning a coregistered DataFrame.

        If require_both_brackets is True or False, bracket validation runs before sampling and filters
        point_data to only reconstructable points. Pass None to skip validation (e.g. synthetic grids).
        """
        self.point_data = point_data
        self.mantle_data_dir = mantle_data_dir
        self.plate_reconstruction = plate_reconstruction

        if require_both_brackets is not None:
            valid_times = self.mantle_dataset["time"].values
            self.validate_brackets(valid_times, require_both=require_both_brackets)

        self.compute_all(feature_names, verbose=verbose)

        new_cols = [c for c in self._results.columns if c not in self._point_data.columns]
        return self._point_data.join(self._results[new_cols])

    def reset(self):
        """Clear cached results, profile cache, bracket results, and resolved coordinates."""
        self._results = None
        self._profile_cache = {}
        self._bracket_results = {}
        self._resolved_coordinates = {}

    # ── Plotting ─────────────────────────────────────────────────────────────

    def grid_sample(
        self,
        feature_names: list[str] | str | None = None,
        times: list[float] | float = 0.0,
        resolution: float = 1.0,
        save_path: str | None = None,
    ) -> xr.Dataset:
        """Sample registered features on a regular global grid at one or more times.

        Points are reconstructed from 0 Ma to the bracket floor/ceil endpoints using
        plate motion — done once per unique bracket pair regardless of how many times
        share that bracket. Feature values are then blended with the appropriate alpha
        per time step, so the result correctly reflects where each plate region was in
        the mantle reference frame at each requested time.

        Args:
            feature_names: Feature name(s) to sample. Defaults to all registered features.
            times: Reconstruction time(s) in Ma.
            resolution: Grid spacing in degrees (default 1°).
            save_path: If given, write the result to this path as NetCDF.

        Returns:
            xr.Dataset with dimensions (time, lat, lon). Variable names have units
            stripped; units are stored as the ``units`` DataArray attribute and the
            full column name as ``long_name``.
        """
        if isinstance(feature_names, str):
            feature_names = [feature_names]
        if feature_names is None:
            feature_names = list(self._features.keys())
        if isinstance(times, float | int):
            times = [float(times)]

        old_point_data = self._point_data.copy() if self._point_data is not None else None
        old_results    = self._results.copy()    if self._results    is not None else None

        lons = np.arange(-180, 180, resolution)
        lats = np.arange(-90, 90 + resolution, resolution)
        flat_lons, flat_lats = np.meshgrid(lons, lats)
        flat_lons = flat_lons.flatten()
        flat_lats = flat_lats.flatten()
        N = len(flat_lons)

        valid_mantle_times = self.mantle_dataset["time"].values
        times_arr = np.array(times, dtype=float)
        floor_arr, ceil_arr = _compute_floor_ceil_times(times_arr, valid_mantle_times)

        bracket_groups: dict[tuple[float, float], list[float]] = {}
        for t, ft, ct in zip(times, floor_arr, ceil_arr):
            bracket_groups.setdefault((float(ft), float(ct)), []).append(float(t))

        self._raw_profile_cache = {}
        time_slices: dict[str, dict[float, np.ndarray]] = {name: {} for name in feature_names}

        try:
            for (floor_t, ceil_t), group_times in bracket_groups.items():

                # ── Step 1: plate-reconstruction filter (once per bracket group) ──────
                # Reconstruct 0 Ma grid to ceil_t; ceil_idx = points with valid plates.
                if ceil_t == 0.0:
                    ceil_idx = np.arange(N)
                    ceil_rlons_s = flat_lons.copy()
                    ceil_rlats_s = flat_lats.copy()
                else:
                    gpts_ceil = gpl.Points(self.plate_reconstruction, lons=flat_lons, lats=flat_lats)
                    ceil_rlons_s, ceil_rlats_s, ceil_idx = gpts_ceil.reconstruct(
                        ceil_t, return_array=True, return_point_indices=True)

                valid_flat_lons = flat_lons[ceil_idx]
                valid_flat_lats = flat_lats[ceil_idx]
                M = len(ceil_idx)

                if floor_t == 0.0:
                    floor_rlons_valid = valid_flat_lons.copy()
                    floor_rlats_valid = valid_flat_lats.copy()
                else:
                    gpts_floor = gpl.Points(self.plate_reconstruction, lons=flat_lons, lats=flat_lats)
                    floor_rlons_s, floor_rlats_s, floor_idx = gpts_floor.reconstruct(
                        floor_t, return_array=True, return_point_indices=True)
                    floor_rlons_full = np.full(N, np.nan)
                    floor_rlats_full = np.full(N, np.nan)
                    floor_rlons_full[floor_idx] = floor_rlons_s
                    floor_rlats_full[floor_idx] = floor_rlats_s
                    floor_rlons_valid = floor_rlons_full[ceil_idx]
                    floor_rlats_valid = floor_rlats_full[ceil_idx]

                # ── Step 2: snap-to-plate-model filter (once per bracket group) ───────
                # Find which of the M ceil-valid points also survive snap_to_plate_model.
                # Pre-cache the result so the feature sampler never re-calls the resolver,
                # guaranteeing that plate velocity uses the exact same point set as the
                # profile cache (preventing shape mismatch in _cache_plate_relative_profiles).
                if self._valid_time_window is not None:
                    min_t, max_t = self._valid_time_window
                    plate_valid_times = np.arange(int(min_t), int(max_t) + 1, 1)
                    _, _, snap_times_full, snap_pidx = _snap_and_reconstruct_points(
                        valid_flat_lons, valid_flat_lats,
                        np.full(M, group_times[0]), plate_valid_times,
                        return_point_indices=True,
                    )
                    snap_times_m2     = snap_times_full[snap_pidx]
                    final_lons        = valid_flat_lons[snap_pidx]
                    final_lats        = valid_flat_lats[snap_pidx]
                    M2                = len(snap_pidx)
                    combined_idx      = ceil_idx[snap_pidx]
                    floor_rlons_final = floor_rlons_valid[snap_pidx]
                    floor_rlats_final = floor_rlats_valid[snap_pidx]
                    ceil_rlons_final  = ceil_rlons_s[snap_pidx]
                    ceil_rlats_final  = ceil_rlats_s[snap_pidx]
                else:
                    snap_times_m2     = None
                    plate_valid_times = None
                    final_lons        = valid_flat_lons
                    final_lats        = valid_flat_lats
                    M2                = M
                    combined_idx      = ceil_idx
                    floor_rlons_final = floor_rlons_valid
                    floor_rlats_final = floor_rlats_valid
                    ceil_rlons_final  = ceil_rlons_s
                    ceil_rlats_final  = ceil_rlats_s

                # ── Step 3: per-time sampling ─────────────────────────────────────────
                if M2 == 0:
                    for time in group_times:
                        for name in feature_names:
                            time_slices[name][time] = np.full((len(lats), len(lons)), np.nan)
                    continue

                for time in group_times:
                    self.reset()  # clears _results, _profile_cache, _bracket_results,
                                  # _resolved_coordinates; does NOT clear _raw_profile_cache

                    alpha_val = ((time - floor_t) / (ceil_t - floor_t)
                                 if ceil_t != floor_t else 0.0)

                    self._point_data = pd.DataFrame({
                        "lon":         final_lons,
                        "lat":         final_lats,
                        "age (Ma)":    np.full(M2, time),
                        "present_lon": final_lons,
                        "present_lat": final_lats,
                    })

                    self._bracket_results[_BRACKET_KEY] = {
                        "floor": (floor_rlons_final, floor_rlats_final, np.full(M2, floor_t)),
                        "ceil":  (ceil_rlons_final,  ceil_rlats_final,  np.full(M2, ceil_t)),
                        "alpha": np.full(M2, alpha_val),
                    }

                    # Pre-populate snap_to_plate_model resolver cache so all features
                    # that use it see the same M2 points as the profile cache.
                    if snap_times_m2 is not None:
                        snapped_t = float(_snap_to_valid_times(np.array([time]), plate_valid_times)[0])
                        self._resolved_coordinates[snap_to_plate_model] = (
                            final_lons, final_lats, np.full(M2, snapped_t),
                        )

                    for name in feature_names:
                        result = self.get(name)
                        full = np.full(N, np.nan)
                        full[combined_idx] = result.values
                        time_slices[name][time] = full.reshape(len(lats), len(lons))

        finally:
            self._point_data = old_point_data
            self._results    = old_results
            self._raw_profile_cache = {}

        time_coord = [float(t) for t in times]
        ds = xr.Dataset({
            _strip_units(name): xr.DataArray(
                np.stack([time_slices[name][t] for t in time_coord], axis=0),
                coords={"time": time_coord, "lat": lats, "lon": lons},
                dims=["time", "lat", "lon"],
                attrs={"units": _extract_units(name), "long_name": name},
            )
            for name in feature_names
        })

        mantle_time = self.mantle_dataset["time"]
        for attr in ("units", "calendar"):
            if attr in mantle_time.attrs:
                ds["time"].attrs[attr] = mantle_time.attrs[attr]

        if save_path is not None:
            ds.to_netcdf(save_path)
        return ds


features = GridFeatureRegistry()
_BRACKET_KEY = object()  # sentinel for _bracket_results — independent of any resolver


def _get_bracket() -> dict:
    """Return the current bracket results, raising RuntimeError if not yet computed.

    Bracket results are populated by validate_brackets(). Call features.validate_brackets(valid_times)
    before sampling any mantle feature, or use features.extract(...) which does this automatically.
    """
    if _BRACKET_KEY not in features._bracket_results:
        raise RuntimeError(
            "Mantle bracket results are not populated. "
            "Call features.validate_brackets(valid_times) before sampling mantle features, "
            "or use features.extract(...) which handles this automatically."
        )
    return features._bracket_results[_BRACKET_KEY]


# ==========================================================================================
# Coordinate resolvers
# ==========================================================================================

def _snap_to_valid_times(
    times: np.ndarray,
    valid_times: np.ndarray,
) -> np.ndarray:
    """Snap times to nearest valid timestep. Ties snap to older time."""
    valid_times = np.sort(valid_times)
    midpoints = (valid_times[:-1] + valid_times[1:]) / 2
    return valid_times[np.digitize(times, midpoints)]


def _compute_floor_ceil_times(
    times: np.ndarray,
    valid_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (floor_times, ceil_times) bracketing each value in times.

    For non-minimum values, an exact hit on a valid timestep naturally gives
    floor < times[i] == ceil (floor is bumped one step down). The only degenerate
    case is times[i] == min(valid_times), where both clip to the minimum — resolved
    in validate_brackets by shifting ceil to the next older step.
    Values outside the valid range are clamped; caller handles further validity checks.
    """
    vt = np.sort(valid_times)
    idx = np.searchsorted(vt, times, side="left")
    floor_idx = np.clip(idx - 1, 0, len(vt) - 1)
    ceil_idx = np.clip(idx, 0, len(vt) - 1)
    return vt[floor_idx], vt[ceil_idx]


def _try_reconstruct(
    present_lons: np.ndarray,
    present_lats: np.ndarray,
    snap_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Attempt reconstruction; return (rlons, rlats, success_mask).

    success_mask is bool of original length; rlons/rlats are NaN where False.
    """
    recon = features.plate_reconstruction
    points = gpl.Points(recon, present_lons, present_lats, 0.0, age=snap_times)
    rlons_s, rlats_s, point_indices = points.reconstruct_to_birth_age(snap_times, return_point_indices=True)
    success_mask = np.zeros(len(present_lons), dtype=bool)
    success_mask[point_indices] = True
    rlons = np.full(len(present_lons), np.nan)
    rlats = np.full(len(present_lons), np.nan)
    rlons[point_indices] = rlons_s
    rlats[point_indices] = rlats_s
    return rlons, rlats, success_mask


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

    if present_lons.size != point_indices.size:
        warnings.warn(
            f"Reconstruction unexpectedly dropped {present_lons.size - point_indices.size} point(s) "
            "after bracket pre-filtering. Results may be misaligned.",
            stacklevel=3,
        )

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
    _, _, times, point_indices = _snap_and_reconstruct_points(
        present_lons=lons,
        present_lats=lats,
        times=times,
        valid_times=valid_times,
        return_point_indices=True,
    )
    return lons[point_indices], lats[point_indices], times[point_indices]


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
    valid_times = features.mantle_dataset["time"].values
    _, _, times, point_indices = _snap_and_reconstruct_points(
        present_lons=present_lons,
        present_lats=present_lats,
        times=times,
        valid_times=valid_times,
        return_point_indices=True,
    )
    return present_lons[point_indices], present_lats[point_indices], times


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


# ==========================================================================================
# Naming helpers
# ==========================================================================================

_UNITS_RE = _re.compile(r'\s*\(([^)]*)\)\s*$')


def _strip_units(col: str) -> str:
    """Return the column name without the trailing ' (units)' suffix."""
    return _UNITS_RE.sub("", col).strip()


def _extract_units(col: str) -> str:
    """Return the units string from a ' (units)' suffix, or '' if absent."""
    m = _UNITS_RE.search(col)
    return m.group(1) if m else ""


def _lab_suffix(offset_km: float) -> str:
    """Return LAB-offset column suffix: '_lab' for offset 0, '_lab+Nkm' otherwise."""
    return "_lab" if offset_km == 0 else f"_lab+{int(offset_km)}km"


def _to_delta_name(col: str) -> str:
    """'foo_bar (km)' → 'foo_bar_delta (km/Myr)'."""
    if " (" in col:
        name, _, units = col.rpartition(" (")
        return f"{name}_delta ({units[:-1]}/Myr)"
    return f"{col}_delta"


# ==========================================================================================
# Batch-register mantle vars
# ==========================================================================================

unit_conversion_mapping = {
    "kelvin": "K",
    "meter / second": "cm/yr",
}

@features.register_batch(declares="Base_Mantle_Features", coords=reconstructed, probe=False)
def _base_mantle_features_depths(
    lons: np.ndarray,
    lats: np.ndarray,
    times: np.ndarray,
) -> pd.DataFrame:
    """Sample a suite of basic mantle features using bracket interpolation."""
    bracket = _get_bracket()
    floor_lons, floor_lats, floor_times = bracket["floor"]
    ceil_lons,  ceil_lats,  ceil_times  = bracket["ceil"]
    alpha = bracket["alpha"]

    depths_to_sample = [100, 200, 300, 400]
    results = []

    vars_to_sample = [
        # 'FullTemperature_CG',
        # 'Pressure',
        # 'Radial_Velocity',           # covered by lab_relative profile stats
        # 'Temperature_CG',
        # 'Temperature_Deviation_CG',  # covered by lab_relative profile stats
        # 'Velocity_x',
        # 'Velocity_y',
        # 'Velocity_z',
        # 'Viscosity_CG',
        # 'East_Velocity',
        # 'North_Velocity',
        # 'Cell_Volume',
        # 'Speed',                     # dropped (not in profile stats)
        # 'Tangential_Speed',          # covered by lab_relative profile stats
        # 'Radial_Tangential_Ratio',   # dropped (not in profile stats)
        'LAB_Depth',
        'Slab_Top_Depth',
        'Slab_Bottom_Depth',
        'Slab_Thickness',
        'Sublithospheric_Cold_Anomaly_Thickness',
        'Mantle_Wedge_Thickness',
        # 'Cold_Anomaly_Magnitude',
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
        units = da.attrs.get("units", "unitless")
        to_units = unit_conversion_mapping.get(units, units)

        result = sample_mantle_var_interp(
            da, floor_lons, floor_lats, floor_times,
            ceil_lons, ceil_lats, ceil_times, alpha,
            depths=depths_to_sample if "depth" in da.dims else None,
        )

        clean_var = var.lower().replace("_cg", "")
        # Rename columns to include variable name and depth if applicable
        if len(result.columns) == len(depths_to_sample):  # Sampled across depths
            result.columns = [f"{clean_var}_{int(d)}km ({to_units})" for d in depths_to_sample]
        elif len(result.columns) == 1:  # Single depth or depth-independent
            result.columns = [f"{clean_var} ({to_units})"]
        else:
            raise ValueError(f"Unexpected number of columns in result for variable '{var}': {len(result.columns)}")

        results.append(result)

    return pd.concat(results, axis=1)


@features.register_batch(declares="Base_Mantle_Features_LAB", coords=reconstructed, probe=False)
def _base_mantle_features_LAB(
    lons: np.ndarray,
    lats: np.ndarray,
    times: np.ndarray,
) -> pd.DataFrame:
    """Sample a suite of basic mantle features at LAB depth using bracket interpolation."""
    bracket = _get_bracket()
    floor_lons, floor_lats, floor_times = bracket["floor"]
    ceil_lons,  ceil_lats,  ceil_times  = bracket["ceil"]
    alpha = bracket["alpha"]

    offsets_to_sample = [0]
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
        units = da.attrs.get("units", "unitless")
        to_units = unit_conversion_mapping.get(units, units)
        if to_units != units:
            da = da.pint.quantify().pint.to(to_units).pint.dequantify()

        new_columns = []
        for offset in offsets_to_sample:
            result = sample_LAB_depths_interp(
                ds=features.mantle_dataset,
                da=da,
                floor_lons=floor_lons, floor_lats=floor_lats, floor_times=floor_times,
                ceil_lons=ceil_lons,   ceil_lats=ceil_lats,   ceil_times=ceil_times,
                alpha=alpha,
                offset_km=offset,
            )
            new_columns.append(result)
        result = pd.concat(new_columns, axis=1)
        clean_var = var.lower().replace("_cg", "")
        # Rename columns to include variable name and LAB offset
        if len(result.columns) == len(offsets_to_sample):  # Sampled across offsets
            result.columns = [f"{clean_var}{_lab_suffix(d)} ({to_units})" for d in offsets_to_sample]
        elif len(result.columns) == 1:  # Single offset or depth-independent
            result.columns = [f"{clean_var}_lab ({to_units})"]
        else:
            raise ValueError(f"Unexpected number of columns in result for variable '{var}': {len(result.columns)}")

        results.append(result)

    return pd.concat(results, axis=1)


# ==========================================================================================
# Feature definitions
# ==========================================================================================


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


# def _relative_tangential_velocity_LAB(
#     lons: np.ndarray,
#     lats: np.ndarray,
#     times: np.ndarray,
#
#     offset_km: float = 0,  # km; depth to sample mantle velocity below the LAB
# ) -> pd.DataFrame:
#     """Sample relative tangential velocity between plate and mantle using bracket interpolation."""
#     bracket = _get_bracket()
#     floor_lons, floor_lats, floor_times = bracket["floor"]
#     ceil_lons,  ceil_lats,  ceil_times  = bracket["ceil"]
#     alpha = bracket["alpha"]
#
#     def _fetch_tangential_velocity_at_LAB(var):
#         da = variables.get(var, features.mantle_dataset)
#         da = da.pint.quantify().pint.to("cm/yr").pint.dequantify()
#         return sample_LAB_depths_interp(
#             ds=features.mantle_dataset, da=da,
#             floor_lons=floor_lons, floor_lats=floor_lats, floor_times=floor_times,
#             ceil_lons=ceil_lons,   ceil_lats=ceil_lats,   ceil_times=ceil_times,
#             alpha=alpha,
#             offset_km=offset_km,
#         ).to_numpy()
#
#     v_mantle = np.column_stack([
#         _fetch_tangential_velocity_at_LAB(variable)
#         for variable in ("East_Velocity", "North_Velocity")
#     ])
#     v_plate = np.column_stack([
#         features.get(feature).to_numpy()
#         for feature in ("east_plate_velocity (cm/yr)", "north_plate_velocity (cm/yr)")
#     ])
#
#     delta_v = v_mantle - v_plate
#     v_mag = np.linalg.norm(delta_v, axis=1)
#
#     result = pd.DataFrame(
#         np.column_stack([delta_v, v_mag]),
#         columns=[
#             f"mantle_relative_east_velocity{_lab_suffix(offset_km)} (cm/yr)",
#             f"mantle_relative_north_velocity{_lab_suffix(offset_km)} (cm/yr)",
#             f"mantle_relative_speed{_lab_suffix(offset_km)} (cm/yr)",
#         ]
#     )
#     return result



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


# def _relative_velocity_in_plate_frame(
#     lons: np.ndarray,
#     lats: np.ndarray,
#     times: np.ndarray,
#
#     offset_km: float = 0,  # km; depth to sample mantle velocity below the LAB for calculating plate frame velocity components
# ) -> pd.DataFrame:
#     """Sample relative velocity between plate and mantle in the reference frame of the overlying plate."""
#     v_rel = np.column_stack([
#         features.get(f"mantle_relative_east_velocity{_lab_suffix(offset_km)} (cm/yr)").to_numpy(),
#         features.get(f"mantle_relative_north_velocity{_lab_suffix(offset_km)} (cm/yr)").to_numpy(),
#     ])
#
#     v_hat_parallel, v_hat_transverse = _calculate_plate_frame_velocity_components()
#
#     relative_parallel = np.vecdot(v_rel, v_hat_parallel, axis=1)
#     relative_transverse = np.vecdot(v_rel, v_hat_transverse, axis=1)
#
#     result = pd.DataFrame(
#         np.column_stack([relative_parallel, relative_transverse]),
#         columns=[
#             f"relative_velocity_parallel_to_plate{_lab_suffix(offset_km)} (cm/yr)",
#             f"relative_velocity_transverse_to_plate{_lab_suffix(offset_km)} (cm/yr)",
#         ]
#     )
#     return result


# @features.register_batch("Temperature_Deviation_Lambdas", coords=snap_to_mantle)
# def _temperature_lambdas(
#     lons: np.ndarray,
#     lats: np.ndarray,
#     times: np.ndarray,

#     n_lambdas: int = 5,
# ) -> pd.DataFrame:
#     """Calculate the first `n_lambdas` polynomial regression coefficients of the mantle geotherm."""

#     result = calculate_lambdas(
#         da=variables.get("Temperature_Deviation_CG", features.mantle_dataset),
#         times=times,
#         lats=lats,
#         lons=lons,
#         depth_range=(0, 400),
#         n_lambdas=n_lambdas
#     )

#     return result

# ==========================================================================================
# Feature transforms
# ==========================================================================================

# Feature transforms accept a feature and registers a new feature based on it.


# def _feature_LAB_depth_averages(
#     lons,
#     lats,
#     times,
#     sampler: Callable[[np.ndarray, np.ndarray, np.ndarray, float], pd.DataFrame],
#     offset_depths: list[float] = [0, 40, 80, 120],
# ) -> pd.DataFrame:
#     """
#     Sample a depth-dependent feature at a series of depth offsets from the LAB, then return the average.
#     """
#     samples = []
#     col_names = None
#     for offset in offset_depths:
#         sample_df = sampler(lons=lons, lats=lats, times=times, offset_km=offset)
#         if col_names is None:
#             col_names = sample_df.columns
#         samples.append(sample_df.to_numpy())
#
#     samples = np.stack(samples, axis=0)
#     average = np.nanmean(samples, axis=0)
#
#     result_df = pd.DataFrame(average, columns=col_names)
#     return result_df


# ==========================================================================================
# Register derivative features
# ==========================================================================================


# LAB offsets — registration loop removed; depth-offset sampling superseded by lab_relative profile stats
# LAB_offsets = [0, 40, 80, 120, 160]
# for LAB_offset in LAB_offsets:
#     features.register_batch(
#         declares=[
#             f"mantle_relative_east_velocity{_lab_suffix(LAB_offset)} (cm/yr)",
#             f"mantle_relative_north_velocity{_lab_suffix(LAB_offset)} (cm/yr)",
#             f"mantle_relative_speed{_lab_suffix(LAB_offset)} (cm/yr)",
#         ],
#         coords=reconstructed,
#         probe=False,
#     )(partial(_relative_tangential_velocity_LAB, offset_km=LAB_offset))
#
#     features.register_batch(
#         declares=[
#             f"relative_velocity_parallel_to_plate{_lab_suffix(LAB_offset)} (cm/yr)",
#             f"relative_velocity_transverse_to_plate{_lab_suffix(LAB_offset)} (cm/yr)",
#         ],
#         coords=reconstructed,
#         probe=False,
#     )(partial(_relative_velocity_in_plate_frame, offset_km=LAB_offset))
#
# features.register_batch(
#     declares=[
#         f"mantle_relative_east_velocity_LAB_{LAB_offset}km_avg (cm/yr)",
#         f"mantle_relative_north_velocity_LAB_{LAB_offset}km_avg (cm/yr)",
#         f"mantle_relative_speed_LAB_{LAB_offset}km_avg (cm/yr)"
#     ], coords=reconstructed, probe=False,
# )(partial(_feature_LAB_depth_averages, sampler=_relative_tangential_velocity_LAB, offset_depths=LAB_offsets))
#
# features.register_batch(
#     declares=[
#         f"relative_velocity_parallel_to_plate_LAB_{LAB_offset}km (cm/yr)_avg",
#         f"relative_velocity_transverse_to_plate_LAB_{LAB_offset}km (cm/yr)_avg",
#     ], coords=reconstructed, probe=False,
# )(partial(_feature_LAB_depth_averages, sampler=_relative_velocity_in_plate_frame, offset_depths=LAB_offsets))



# Columns produced by _base_mantle_features_depths that have delta features.
# Names must match the actual output column names (new convention: lowercase, units in parens).
base_mantle_features = [
    'lab_depth (km)',
    'slab_top_depth (km)',
    'slab_bottom_depth (km)',
    'slab_thickness (km)',
    'sublithospheric_cold_anomaly_thickness (km)',
    'mantle_wedge_thickness (km)',
    # 'cold_anomaly_magnitude (K)',
    # 'temperature_deviation_avg_0-400km (K)',
    # 'temperature_deviation_avg_0-400km_rolling_30ma (K)',
    # 'temperature_deviation_avg_0-400km_rolling_50ma (K)',
    # 'temperature_deviation_avg_100-400km (K)',
    # 'temperature_deviation_avg_100-400km_rolling_30ma (K)',
    # 'temperature_deviation_avg_100-400km_rolling_50ma (K)',
    # 'temperature_deviation_avg_lab-120km (K)',
    # 'temperature_deviation_avg_lab-120km_rolling_30ma (K)',
    # 'temperature_deviation_avg_lab-120km_rolling_50ma (K)',
    # 'temperature_deviation_avg_lab-160km (K)',
    # 'temperature_deviation_avg_lab-160km_rolling_30ma (K)',
    # 'temperature_deviation_avg_lab-160km_rolling_50ma (K)',
    # 'temperature_deviation_avg_lab-200km (K)',
    # 'temperature_deviation_avg_lab-200km_rolling_30ma (K)',
    # 'temperature_deviation_avg_lab-200km_rolling_50ma (K)',
    # 'temperature_deviation_avg_lab-300km (K)',
    # 'temperature_deviation_avg_lab-300km_rolling_30ma (K)',
    # 'temperature_deviation_avg_lab-300km_rolling_50ma (K)',
    # 'temperature_deviation_avg_lab-400km (K)',
    # 'temperature_deviation_avg_lab-400km_rolling_30ma (K)',
    # 'temperature_deviation_avg_lab-400km_rolling_50ma (K)',
#     'temperature_deviation_cg_lambda_0 (K)',
#     'temperature_deviation_cg_lambda_1 (K)',
#     'temperature_deviation_cg_lambda_2 (K)',
#     'temperature_deviation_cg_lambda_3 (K)',
#     'temperature_deviation_cg_lambda_4 (K)',
]

# Maps base_mantle_features column names → mantle variable registry keys for delta sampling.
_base_mantle_col_to_var: dict[str, str] = {
    'lab_depth (km)':                              'LAB_Depth',
    'slab_top_depth (km)':                         'Slab_Top_Depth',
    'slab_bottom_depth (km)':                      'Slab_Bottom_Depth',
    'slab_thickness (km)':                         'Slab_Thickness',
    'sublithospheric_cold_anomaly_thickness (km)': 'Sublithospheric_Cold_Anomaly_Thickness',
    'mantle_wedge_thickness (km)':                 'Mantle_Wedge_Thickness',
    # 'cold_anomaly_magnitude (K)':                  'Cold_Anomaly_Magnitude',
}

@features.register_batch(
    # declares=[_to_delta_name(fn) for fn in base_mantle_features],
    declares=["Base_Mantle_Deltas"],
    coords=reconstructed,
    probe=False,
)
def _mantle_variable_deltas(lons: np.ndarray, lats: np.ndarray, times: np.ndarray) -> pd.DataFrame:
    """Compute per-Ma gradient of base mantle variables from the pre-computed bracket."""
    bracket = _get_bracket()
    floor_lons, floor_lats, floor_times = bracket["floor"]
    ceil_lons,  ceil_lats,  ceil_times  = bracket["ceil"]
    t_span = ceil_times - floor_times

    ds = features.mantle_dataset
    frames = []
    for fn in base_mantle_features:
        var_name = _base_mantle_col_to_var[fn]
        da = variables.get(var_name, ds)
        v_floor = sample_mantle_var(da, floor_lons, floor_lats, floor_times)
        v_ceil  = sample_mantle_var(da, ceil_lons,  ceil_lats,  ceil_times)
        safe_span = np.where(t_span > 0, t_span, np.nan)
        gradient = (v_ceil.values - v_floor.values) / safe_span[:, None]
        frames.append(pd.DataFrame(gradient, columns=[_to_delta_name(fn)]))
    return pd.concat(frames, axis=1)


# @features.register_batch(
#     declares="Temperature_Deviation_Lambdas_delta",
#     coords=reconstructed,
#     probe=False,
# )
# def _temperature_lambdas_delta(lons: np.ndarray, lats: np.ndarray, times: np.ndarray) -> pd.DataFrame:
#     offset_lons, offset_lats, offset_times = _offset_coordinates_by_time(reconstructed, n_timesteps=1)
#     current = _temperature_lambdas(lons, lats, times)
#     previous = _temperature_lambdas(offset_lons, offset_lats, offset_times)
#     delta_df = current.subtract(previous.values)
#     delta_df.columns = [f"{col.replace(' ', '_')}_delta" for col in current.columns]
#     return delta_df


# ==========================================================================================
# Depth-profile statistics
# ==========================================================================================

# Variables sampled as raw profiles in addition to those in DEPTH_PROFILE_STATS —
# East/North are needed to compute plate-relative velocity profiles.
_PROFILE_VARS_BASIC = [
    k for k in DEPTH_PROFILE_STATS
    if k not in {"plate_parallel_velocity", "plate_transverse_velocity"}
] + ["East_Velocity", "North_Velocity"]


def _cache_basic_profiles(
    bracket: dict,
    ref_depths_floor: np.ndarray,
    ref_depths_ceil: np.ndarray,
    slice_size: float,
    n_depth_samples: int,
    profile_set_key: str = "lab_relative",
) -> None:
    """Sample depth profiles for basic mantle variables and store in features._profile_cache.

    ref_depths_floor and ref_depths_ceil are pre-resolved reference depths at each
    bracket timestep (passed in so they are computed only once per batch call).
    Profiles are stored under features._profile_cache[profile_set_key][var].

    Raw (un-blended) floor/ceil profiles are cached in features._raw_profile_cache keyed by
    (floor_t, ceil_t) so that subsequent calls within the same bracket group (same bracket
    endpoints but different alpha) skip the expensive RGI sampling and only re-blend.
    """
    floor_lons, floor_lats, floor_times = bracket["floor"]
    ceil_lons,  ceil_lats,  ceil_times  = bracket["ceil"]
    alpha = bracket["alpha"]
    ds = features.mantle_dataset

    floor_t = float(floor_times[0])
    ceil_t  = float(ceil_times[0])
    bracket_key = (floor_t, ceil_t)
    raw_cache   = features._raw_profile_cache
    profile_set = features._profile_cache.setdefault(profile_set_key, {})

    for var in _PROFILE_VARS_BASIC:
        if bracket_key in raw_cache and var in raw_cache[bracket_key]:
            floor_profiles, ceil_profiles, offsets = raw_cache[bracket_key][var]
        else:
            da = variables.get(var, ds)
            units = da.attrs.get("units", "unitless")
            to_units = unit_conversion_mapping.get(units, units)
            if to_units != units:
                da = da.pint.quantify().pint.to(to_units).pint.dequantify()
            floor_profiles, offsets = sample_mantle_var_depth_profile(
                da, floor_lons, floor_lats, floor_times,
                ref_depths_floor, slice_size, n_depth_samples,
            )
            ceil_profiles, _ = sample_mantle_var_depth_profile(
                da, ceil_lons, ceil_lats, ceil_times,
                ref_depths_ceil, slice_size, n_depth_samples,
            )
            raw_cache.setdefault(bracket_key, {})[var] = (floor_profiles, ceil_profiles, offsets)

        profiles = floor_profiles + alpha[:, None] * (ceil_profiles - floor_profiles)
        profile_set[var] = (profiles, offsets)


def _cache_plate_relative_profiles(profile_set_key: str = "lab_relative") -> None:
    """Compute plate-relative velocity profiles from cached East/North profiles.

    Requires East_Velocity and North_Velocity to already be in
    features._profile_cache[profile_set_key] (populated by _cache_basic_profiles)
    and plate velocity features to be available via features.get().

    Floor and ceil mantle velocity profiles are each sampled on their own reference-depth
    grids before temporal interpolation, so the absolute depths differ slightly between
    brackets. The resulting plate-relative profiles inherit this approximation,
    consistent with all other bracket-interpolated features.
    """
    profile_set = features._profile_cache[profile_set_key]
    ve_profiles, offsets = profile_set["East_Velocity"]
    vn_profiles, _       = profile_set["North_Velocity"]

    v_plate_east  = features.get("east_plate_velocity (cm/yr)").to_numpy()   # (N,)
    v_plate_north = features.get("north_plate_velocity (cm/yr)").to_numpy()  # (N,)

    delta_ve = ve_profiles - v_plate_east[:, None]    # (N, S)
    delta_vn = vn_profiles - v_plate_north[:, None]   # (N, S)

    v_hat_parallel, v_hat_transverse = _calculate_plate_frame_velocity_components()

    parallel_profiles   = delta_ve * v_hat_parallel[:, 0:1]   + delta_vn * v_hat_parallel[:, 1:2]
    transverse_profiles = delta_ve * v_hat_transverse[:, 0:1] + delta_vn * v_hat_transverse[:, 1:2]

    profile_set["plate_parallel_velocity"]   = (parallel_profiles,   offsets)
    profile_set["plate_transverse_velocity"] = (transverse_profiles, offsets)


@features.register_batch(
    declares=profile_stat_column_names(descriptor="lab_relative"),
    coords=reconstructed,
    probe=False,
)
def _mantle_depth_profile_stats(
    lons: np.ndarray,
    lats: np.ndarray,
    times: np.ndarray,
    slice_size: float = 300.0,
    n_depth_samples: int = 30,
) -> pd.DataFrame:
    """Extract depth-profile summary statistics for mantle variables.

    For each point, profiles are sampled from the LAB down to LAB + slice_size km
    at n_depth_samples evenly-spaced offsets. Statistics are then computed on each
    profile and returned as a flat DataFrame of named columns.

    LAB depths are resolved separately at floor and ceil bracket timesteps and
    passed to the sampling functions — a single resolution covers all variables.
    """
    bracket = _get_bracket()
    floor_lons, floor_lats, floor_times = bracket["floor"]
    ceil_lons,  ceil_lats,  ceil_times  = bracket["ceil"]

    lab_da = variables.get("LAB_Depth", features.mantle_dataset)
    ref_depths_floor = sample_mantle_var(lab_da, floor_lons, floor_lats, floor_times).values.flatten()
    ref_depths_ceil  = sample_mantle_var(lab_da, ceil_lons,  ceil_lats,  ceil_times).values.flatten()

    _cache_basic_profiles(bracket, ref_depths_floor, ref_depths_ceil, slice_size, n_depth_samples,
                          profile_set_key="lab_relative")
    _cache_plate_relative_profiles(profile_set_key="lab_relative")

    profile_set = features._profile_cache["lab_relative"]
    frames = []
    for var_key, stats in DEPTH_PROFILE_STATS.items():
        profiles, offsets = profile_set[var_key]
        var_units = DEPTH_PROFILE_VAR_UNITS[var_key]
        frames.append(compute_depth_profile_stats(
            profiles, offsets, var_key, var_units, stats, descriptor="lab_relative",
        ))

    return pd.concat(frames, axis=1)


_TEMP_DEV_PROFILE_STATS = DEPTH_PROFILE_STATS["Temperature_Deviation_CG"]
_TEMP_DEV_DELTA_DECLARES = [
    _to_delta_name(col)
    for col in profile_stat_column_names(
        {"Temperature_Deviation_CG": _TEMP_DEV_PROFILE_STATS},
        {"Temperature_Deviation_CG": DEPTH_PROFILE_VAR_UNITS["Temperature_Deviation_CG"]},
        descriptor="lab_relative",
    )
]


@features.register_batch(
    declares=_TEMP_DEV_DELTA_DECLARES,
    coords=reconstructed,
    probe=False,
)
def _temperature_deviation_profile_stat_deltas(
    lons: np.ndarray,
    lats: np.ndarray,
    times: np.ndarray,
    slice_size: float = 300.0,
    n_depth_samples: int = 30,
) -> pd.DataFrame:
    """Time gradient of LAB-relative temperature deviation profile stats.

    Samples Temperature_Deviation_CG profiles at the floor and ceil bracket
    timesteps separately (no alpha blending), computes profile stats on each,
    and returns (ceil_stats - floor_stats) / t_span as rates per timestep.
    """
    bracket = _get_bracket()
    floor_lons, floor_lats, floor_times = bracket["floor"]
    ceil_lons,  ceil_lats,  ceil_times  = bracket["ceil"]

    lab_da = variables.get("LAB_Depth", features.mantle_dataset)
    ref_depths_floor = sample_mantle_var(lab_da, floor_lons, floor_lats, floor_times).values.flatten()
    ref_depths_ceil  = sample_mantle_var(lab_da, ceil_lons,  ceil_lats,  ceil_times).values.flatten()

    da = variables.get("Temperature_Deviation_CG", features.mantle_dataset)
    units = da.attrs.get("units", "unitless")
    to_units = unit_conversion_mapping.get(units, units)
    if to_units != units:
        da = da.pint.quantify().pint.to(to_units).pint.dequantify()

    floor_profiles, offsets = sample_mantle_var_depth_profile(
        da, floor_lons, floor_lats, floor_times, ref_depths_floor, slice_size, n_depth_samples,
    )
    ceil_profiles, _ = sample_mantle_var_depth_profile(
        da, ceil_lons, ceil_lats, ceil_times, ref_depths_ceil, slice_size, n_depth_samples,
    )

    var_units = DEPTH_PROFILE_VAR_UNITS["Temperature_Deviation_CG"]
    floor_stats = compute_depth_profile_stats(
        floor_profiles, offsets, "Temperature_Deviation_CG", var_units,
        _TEMP_DEV_PROFILE_STATS, descriptor="lab_relative",
    )
    ceil_stats = compute_depth_profile_stats(
        ceil_profiles, offsets, "Temperature_Deviation_CG", var_units,
        _TEMP_DEV_PROFILE_STATS, descriptor="lab_relative",
    )

    t_span = ceil_times - floor_times
    safe_span = np.where(t_span > 0, t_span, np.nan)
    delta = (ceil_stats.values - floor_stats.values) / safe_span[:, None]
    return pd.DataFrame(delta, columns=_TEMP_DEV_DELTA_DECLARES)
