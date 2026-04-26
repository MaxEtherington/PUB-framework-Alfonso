"""Grid feature registry for sampling mantle and plate model data.

Provides decorator-based registration and sampling of gridded features
at point locations (lon, lat, time).
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr
import gplately as gpl

from .mantle_variables import variables, sample_mantle_var

# ==================
# Grid feature registry
# ==================


@dataclass
class GridFeature:
    """A registered grid feature with a sampler function and metadata."""
    name: str
    sampler: Callable[[np.ndarray, np.ndarray, np.ndarray], pd.Series | pd.DataFrame]


class GridFeatureRegistry:
    def __init__(self):
        self._features: dict[str, GridFeature] = {}
        self._results: pd.DataFrame | None = None
        self._mantle_dataset: xr.Dataset | None = None
        self.mantle_data_dir: str | None = None  # Injected at runtime from notebook

    @property
    def mantle_dataset(self) -> xr.Dataset:
        if self._mantle_dataset is None:
            if self.mantle_data_dir is None:
                raise RuntimeError("data_dir has not been set on the registry.")
            self._mantle_dataset = xr.open_mfdataset(
                sorted(self.mantle_data_dir.glob("*.nc")),
                combine='nested',
                concat_dim='time',
                chunks={"time": 1, "depth": 25},
            )
        return self._mantle_dataset
    
    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, name: str):
        """Decorator to register a grid feature sampler function."""
        def decorator(fn):
            self._features[name] = GridFeature(
                name=name,
                sampler=fn,
            )
            return fn
        return decorator

    def register_batch(self, declares=None):
        """Decorator to register a batch producer as one or more features."""
        def decorator(fn):
            try:
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
                        sampler=fn,
                    )
            except Exception as exc:
                if isinstance(declares, list):
                    for name in declares:
                        self._features[str(name)] = GridFeature(
                            name=str(name),
                            sampler=fn,
                        )
                elif isinstance(declares, str):
                    self._features[declares] = GridFeature(
                        name=declares,
                        sampler=fn,
                    )
                elif declares is None:
                    raise RuntimeError(
                        f"Unable to register batch producer '{fn.__name__}': probing failed and no declarations were provided. "
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

    def get(self, name: str) -> GridFeature:
        """Return the GridFeature object (without calling it)."""
        if name not in self._features:
            raise KeyError(f"Unknown feature: '{name}'. Available: {list(self._features)}")
        return self._features[name]

    def sample(
        self,
        name: str,
        lons: np.ndarray,
        lats: np.ndarray,
        times: np.ndarray,
    ) -> pd.Series | pd.DataFrame:
        """Sample a feature and cache result columns in ``_results``."""
        feature = self.get(name)

        if self._results is not None:
            if name in self._results.columns:
                return self._results[name]

        result = feature.sampler(lons, lats, times)

        if isinstance(result, pd.Series):
            col_name = result.name or name
            result_df = result.rename(col_name).to_frame()
        else:
            result_df = result

            # If the declared name is not an actual output column, treat 
            ## it as a placeholder for all columns produced by this sampler.
            col_names = [str(c) for c in result_df.columns]
            is_placeholder = name not in col_names
            if is_placeholder:
                for col_name in col_names:
                    self._features[col_name] = GridFeature(
                        name=col_name,
                        sampler=feature.sampler,
                    )
                self._features.pop(name, None)

        if self._results is None:
            self._results = result_df.copy()
        else:
            new_cols = [c for c in result_df.columns if c not in self._results.columns]
            if new_cols:
                self._results = pd.concat([self._results, result_df[new_cols]], axis=1)

        if isinstance(result, pd.Series):
            return self._results[col_name]
        return self._results[result_df.columns]

    def available(self) -> list[str]:
        """Return list of available feature names."""
        return list(self._features)

    def compute(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        times: np.ndarray,
    ) -> None:
        """Populate ``_results`` by sampling all registered features."""
        for name in self.available():
            self.sample(name, lons, lats, times)

    def reset(self):
        """Clear cached results."""
        self._results = None


features = GridFeatureRegistry()


# ==================
# Feature definitions
# ==================


@features.register("Temperature_Deviation")
def _temperature_deviation(
    lons: np.ndarray,
    lats: np.ndarray,
    times: np.ndarray,
) -> pd.Series | pd.DataFrame:
    """Sample temperature deviation from mantle grid at requested coordinates.
    
    Parameters
    ----------
    lons, lats, times : array-like
        Coordinate arrays for sample points.
    
    Returns
    -------
    pd.Series or pd.DataFrame
        Sampled temperature deviation values.
    """
    if features.mantle_dataset is None:
        raise RuntimeError(
            "Dataset has not been injected into the registry. "
            "Set features.mantle_dataset in your notebook before sampling."
        )
    
    da = variables.get("Temperature_Deviation_CG", features.mantle_dataset)
    return sample_mantle_var(
        da=da,
        lons=lons,
        lats=lats,
        times=times,
        method="linear",
    )


@features.register("Plate_Velocity_Delta")
def _plate_velocity_delta(
    lons: np.ndarray,
    lats: np.ndarray,
    times: np.ndarray,
) -> pd.Series | pd.DataFrame:
    """Sample plate velocity perturbation at requested coordinates.
    
    TODO: Wire to plate model once available.
    """
    pass


# Features requiring plate velocities (need to be extracted from plate model)


def _plate_velocity_delta(ds: xr.Dataset) -> xr.DataArray: #TODO
    pass


def _relative_tangential_velocity(ds: xr.Dataset) -> xr.DataArray: #TODO
    pass