"""Summary statistics for mantle variable depth profiles.

Stat functions all have the signature:
    (values: np.ndarray, offsets: np.ndarray) -> float

where `values` is the sampled profile for one point and `offsets` is the
corresponding depth offsets from the reference depth (e.g. LAB), in km.
offsets[0] == 0 by construction (the reference depth itself).

Each stat function carries a `units_kind` attribute from the set:
    "identity"  → output has the same units as the variable
    "per_km"    → output units are {var_units}/km
    "times_km"  → output units are {var_units}·km
    "km"        → output is always in km (a depth)
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# Stat functions
# ══════════════════════════════════════════════════════════════════════════════


def ref_value(values: np.ndarray, offsets: np.ndarray) -> float:  # noqa: ARG001
    """Value at the reference depth (offset == 0)."""
    if len(values) == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(values[0])


ref_value.units_kind = "identity"


def ref_gradient(values: np.ndarray, offsets: np.ndarray) -> float:
    """Forward finite-difference gradient at the reference depth."""
    if len(values) < 2 or np.any(np.isnan(values[:2])):
        return np.nan
    return float((values[1] - values[0]) / (offsets[1] - offsets[0]))


ref_gradient.units_kind = "per_km"


def depth_integral(values: np.ndarray, offsets: np.ndarray) -> float:
    """Trapezoidal integral of the profile over the full depth window."""
    if len(values) < 2 or np.all(np.isnan(values)):
        return np.nan
    return float(np.trapz(values, offsets))


depth_integral.units_kind = "times_km"


def min_value(values: np.ndarray, offsets: np.ndarray) -> float:  # noqa: ARG001
    """Minimum value in the profile."""
    if len(values) == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmin(values))


min_value.units_kind = "identity"


def depth_of_min(values: np.ndarray, offsets: np.ndarray) -> float:
    """Depth offset (from reference) at which the minimum value occurs."""
    if len(values) == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(offsets[np.nanargmin(values)])


depth_of_min.units_kind = "km"


def max_value(values: np.ndarray, offsets: np.ndarray) -> float:  # noqa: ARG001
    """Maximum value in the profile."""
    if len(values) == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmax(values))


max_value.units_kind = "identity"


def depth_of_max(values: np.ndarray, offsets: np.ndarray) -> float:
    """Depth offset (from reference) at which the maximum value occurs."""
    if len(values) == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(offsets[np.nanargmax(values)])


depth_of_max.units_kind = "km"


def abs_max_value(values: np.ndarray, offsets: np.ndarray) -> float:  # noqa: ARG001
    """Maximum absolute value in the profile (preserves no sign information)."""
    if len(values) == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmax(np.abs(values)))


abs_max_value.units_kind = "identity"


def depth_of_abs_max(values: np.ndarray, offsets: np.ndarray) -> float:
    """Depth offset at which the largest-magnitude value occurs."""
    if len(values) == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(offsets[np.nanargmax(np.abs(values))])


depth_of_abs_max.units_kind = "km"


def zero_crossing_depth(values: np.ndarray, offsets: np.ndarray) -> float:
    """Depth offset of the first sign change, via linear interpolation.

    Returns np.nan if no sign change exists within the profile window.
    For velocity profiles this marks where the flow direction reverses.
    """
    if len(values) < 2:
        return np.nan
    for i in range(len(values) - 1):
        v0, v1 = values[i], values[i + 1]
        if np.isnan(v0) or np.isnan(v1):
            continue
        if v0 == 0.0:
            return float(offsets[i])
        if (v0 > 0.0 and v1 < 0.0) or (v0 < 0.0 and v1 > 0.0):
            d0, d1 = offsets[i], offsets[i + 1]
            return float(d0 + (-v0) / (v1 - v0) * (d1 - d0))
    return np.nan


zero_crossing_depth.units_kind = "km"


# ══════════════════════════════════════════════════════════════════════════════
# Registries
# ══════════════════════════════════════════════════════════════════════════════

DEPTH_PROFILE_STATS: dict[str, list[Callable]] = {
    "Temperature_Deviation_CG": [
        ref_value, ref_gradient, depth_integral,
        min_value, depth_of_min,
        max_value, depth_of_max,
    ],
    "Tangential_Speed": [
        ref_value, ref_gradient,
        max_value, depth_of_max,
    ],
    "Radial_Velocity": [
        ref_value, ref_gradient,
        min_value, depth_of_min,
        max_value, depth_of_max,
    ],
    "plate_parallel_velocity": [
        ref_value, ref_gradient, depth_integral,
        abs_max_value, depth_of_abs_max,
    ],
    "plate_transverse_velocity": [
        ref_value, ref_gradient, depth_integral,
        abs_max_value, depth_of_abs_max,
    ],
}

DEPTH_PROFILE_VAR_UNITS: dict[str, str] = {
    "Temperature_Deviation_CG": "K",
    "Tangential_Speed":          "cm/yr",
    "Radial_Velocity":           "cm/yr",
    "plate_parallel_velocity":   "cm/yr",
    "plate_transverse_velocity": "cm/yr",
}

_UNITS_KIND_FMT: dict[str, Callable[[str], str]] = {
    "identity": lambda u: u,
    "per_km":   lambda u: f"{u}/km",
    "times_km": lambda u: f"{u}·km",
    "km":       lambda _: "km",
}


# ══════════════════════════════════════════════════════════════════════════════
# Naming helpers
# ══════════════════════════════════════════════════════════════════════════════


def profile_stat_column_name(
    var_key: str,
    stat: Callable,
    var_units: str,
    descriptor: str = "profile",
) -> str:
    """Canonical column name for one (variable, stat) pair.

    descriptor labels the depth-profile context and appears between the
    variable name and stat name. Examples: "profile" (default),
    "lab_relative", "upper_mantle".
    """
    clean_var = var_key.lower().replace("_cg", "")
    units = _UNITS_KIND_FMT[stat.units_kind](var_units)
    return f"{clean_var}_{descriptor}_{stat.__name__} ({units})"


def profile_stat_column_names(
    stats_registry: dict[str, list[Callable]] | None = None,
    var_units: dict[str, str] | None = None,
    descriptor: str = "profile",
) -> list[str]:
    """All column names for the given registry.

    Defaults to DEPTH_PROFILE_STATS / DEPTH_PROFILE_VAR_UNITS.
    Has no runtime data dependencies — safe to call at module load time
    (e.g. as the `declares` argument of @features.register_batch).

    Examples
    --------
    profile_stat_column_names()
        # ["temperature_deviation_profile_ref_value (K)", ...]
    profile_stat_column_names(descriptor="lab_relative")
        # ["temperature_deviation_lab_relative_ref_value (K)", ...]
    """
    if stats_registry is None:
        stats_registry = DEPTH_PROFILE_STATS
    if var_units is None:
        var_units = DEPTH_PROFILE_VAR_UNITS
    return [
        profile_stat_column_name(var_key, stat, var_units[var_key], descriptor)
        for var_key, stats in stats_registry.items()
        for stat in stats
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Compute wrapper
# ══════════════════════════════════════════════════════════════════════════════


def compute_depth_profile_stats(
    profiles: np.ndarray,
    offsets: np.ndarray,
    var_key: str,
    var_units: str,
    stats: list[Callable],
    descriptor: str = "profile",
) -> pd.DataFrame:
    """Apply a list of stat functions to an (N, S) profile array.

    Parameters
    ----------
    profiles : (N, S) float array — one depth profile per point
    offsets  : (S,) float array — depth offsets from reference, km
    var_key  : variable key (used for column naming and _cg stripping)
    var_units: variable's physical units string, e.g. "K" or "cm/yr"
    stats    : list of stat functions from DEPTH_PROFILE_STATS
    descriptor: passed to profile_stat_column_name; default "profile"

    Returns
    -------
    pd.DataFrame of shape (N, len(stats)) with named columns.
    All-NaN rows produce NaN in all output columns without raising.
    """
    result = {}
    for stat in stats:
        col = profile_stat_column_name(var_key, stat, var_units, descriptor)
        result[col] = np.apply_along_axis(
            lambda row: stat(row, offsets), axis=1, arr=profiles
        )
    return pd.DataFrame(result)
