"""Thesis colour palette.

Built on matplotlib's Set2 (light) and Dark2 (dark) qualitative colormaps,
which share the same hue family but differ in luminance — making them a natural
complementary pair for split visualisations (e.g. deposits vs unlabelled).

Index assignment
───────────────
  0 : BASE      — general-purpose base colour for non-region figures
  1 : HIGHLIGHT — annotation / highlight colour (warm, loosely copper-toned)
  2 : East Asia
  3 : Southeast Asia
  4 : Tethys
  5 : North America
  6 : South America
  7 : Overall / global — grey pair, used when no region filter is applied

Feature-set colours also start at index 0 (subduction=0, mantle=1, …) — the
same Dark2 indices, used in a different plot context so there is no conflict.

Typical usage
─────────────
    from lib.colours import *

    # Region-specific colour for violin plots, maps, scatter, etc.
    deposits_colour   = region_colour("East Asia", "light")   # Set2[2]
    unlabelled_colour = region_colour("East Asia", "dark")    # Dark2[2]

    # Feature-set colours for importance bar charts
    colour = feature_set_colour("subduction")   # Dark2[0]

    # General-purpose thesis colours
    ax.plot(..., color=BASE["light"])            # base (teal family)
    ax.axhline(..., color=HIGHLIGHT["dark"])     # highlight (copper family)
"""

import matplotlib as mpl

_SET2  = mpl.colormaps["Set2"].colors   # 8 RGBA tuples, light variants
_DARK2 = mpl.colormaps["Dark2"].colors  # 8 RGBA tuples, dark variants

# ── Reserved colours (indices 0–1) ────────────────────────────────────────────

#: General-purpose base colour pair (teal family).
BASE = {"light": _SET2[0], "dark": _DARK2[0]}

#: General-purpose highlight / annotation pair (copper-toned warm orange).
HIGHLIGHT = {"light": _SET2[1], "dark": _DARK2[1]}

# ── Province colours (indices 2–6, east-to-west) ─────────────────────────────

#: Ordered list of province names; index within this list + 2 gives the
#: Set2/Dark2 palette index.  Order is geographic (east to west).
REGION_NAMES: list[str] = [
    "East Asia",       # Set2[2] / Dark2[2]
    "Southeast Asia",  # Set2[3] / Dark2[3]
    "Tethys",          # Set2[4] / Dark2[4]
    "North America",   # Set2[5] / Dark2[5]
    "South America",   # Set2[6] / Dark2[6]
]

_REGION_INDEX: dict[str, int] = {name: i + 2 for i, name in enumerate(REGION_NAMES)}

# Index 7: grey pair for the "Overall" / "Global" (no region filter) category.
_OVERALL_INDEX = 7

# ── Feature-set colours (indices 0–N, sequential from Dark2) ─────────────────

#: Ordered list of feature-set names; index gives the Dark2 palette index.
FEATURE_SET_NAMES: list[str] = ["subduction", "mantle", "crustal", "erodep"]

_FEATURE_SET_INDEX: dict[str, int] = {n: i for i, n in enumerate(FEATURE_SET_NAMES)}


def region_colour(region: str, variant: str = "light") -> tuple:
    """Return the thesis colour for *region*.

    Parameters
    ----------
    region : str
        Province name (one of :data:`REGION_NAMES`), ``"Overall"``, or
        ``"Global"``.  Unknown names fall back to the grey overall colour.
    variant : {"light", "dark"}
        ``"light"``  → Set2 colour  (used for deposits / positive class).
        ``"dark"``   → Dark2 colour (used for unlabelled class).

    Returns
    -------
    tuple
        RGBA colour tuple from Set2 or Dark2.

    Examples
    --------
    >>> region_colour("East Asia")            # Set2[2], light
    >>> region_colour("Tethys", "dark")       # Dark2[4]
    >>> region_colour("Overall", "light")     # Set2[7], grey
    >>> region_colour("Global", "light")      # Set2[7], grey (alias)
    """
    idx = _REGION_INDEX.get(region, _OVERALL_INDEX)
    return (_SET2 if variant == "light" else _DARK2)[idx]


def feature_set_colour(feature_set: str) -> tuple:
    """Return the Dark2 colour for a *feature_set*.

    Parameters
    ----------
    feature_set : str
        Feature-set name (one of :data:`FEATURE_SET_NAMES`).
        Unknown names fall back to the grey colour (index 7).

    Returns
    -------
    tuple
        RGBA colour tuple from Dark2.

    Examples
    --------
    >>> feature_set_colour("subduction")   # Dark2[0]
    >>> feature_set_colour("mantle")       # Dark2[1]
    """
    return _DARK2[_FEATURE_SET_INDEX.get(feature_set, 7)]


class FeatureSetColour:
    """Callable that maps feature names → thesis colours by feature set.

    Reads the active feature manifest from *pcm* at construction time so the
    caller never needs to manage the manifest dict or call
    :func:`feature_set_colour` directly.

    Parameters
    ----------
    pcm : PathConfigManager
        Active path config.  The manifest JSON is read from
        ``pcm.FEATURES_MANIFEST_PATH``; only feature sets listed in
        ``pcm.active_feature_sets`` are included.

    Usage::

        colour = FeatureSetColour(pcm)

        # Colour a single bar or box:
        ax.barh(..., color=colour(feature_name))

        # Add a feature-set legend:
        ax.legend(handles=colour.legend_handles())
    """

    def __init__(self, pcm) -> None:
        import json

        with open(pcm.FEATURES_MANIFEST_PATH) as _f:
            manifest = json.load(_f)
        self._filtered: dict[str, list[str]] = {
            k: v for k, v in manifest.items() if k in pcm.active_feature_sets
        }
        self._feat_to_set: dict[str, str] = {
            f: s for s, feats in self._filtered.items() for f in feats
        }

    def __call__(self, feature_name: str) -> tuple:
        """Return the Dark2 colour for *feature_name* (grey if unknown)."""
        return feature_set_colour(self._feat_to_set.get(feature_name, ""))

    def legend_handles(self) -> list:
        """Patch handles, one per active feature set, ready for ``ax.legend()``."""
        from matplotlib.patches import Patch

        return [Patch(facecolor=feature_set_colour(n), label=n) for n in self._filtered]
