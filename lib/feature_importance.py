"""Several functions to facilitate extracting feature importance values from
models and plotting the results.
"""
import colorsys
from itertools import (
    combinations,
    product,
)
from sys import stderr

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from scipy.stats import kendalltau
from sklearn.base import BaseEstimator

from .colours import region_colour
from .misc import format_feature_name


def calculate_importances(clf, names=None):
    """Extract the feature importance values from an estimator.

    Parameters
    ----------
    clf : Estimator
        The estimator from which to extract the importances. Either
        `clf.feature_importances_` must exist, or each element of
        `clf.estimators_` must have a `feature_importances_` attribute.
    names : pandas.Index
        The names of the features.

    Returns
    -------
    pandas.Series
        The importance values, with their associated names.
    """
    if not isinstance(clf, BaseEstimator):
        clf = load(clf)
    if hasattr(clf, "feature_importances_"):
        importances = np.array(clf.feature_importances_)
    elif hasattr(clf, "estimators_"):
        importances = np.array(
            [
                i.feature_importances_
                for i in clf.estimators_
            ]
        ).mean(axis=0)
    else:
        raise TypeError(
            "Could not extract feature importances from "
            f"type {type(clf)}"
        )

    if names is None:
        names = np.arange(np.size(importances))
    else:
        names = np.array(names)
    return pd.Series(importances, index=names)


def get_ranks(importances, zero_index=False):
    """Convert a Series of importance values into a series of ranks.

    Parameters
    ----------
    importances : pandas.Series
        The series of importance values, as returned by
        `calculate_importances`.
    zero_index : bool, default: False
        Whether the ranks should begin at zero or one (default).

    Returns
    -------
    pandas.Series
        The series of feature ranks.
    """
    ordered = np.array(importances.sort_values(ascending=False).index)
    ranks = pd.Series(np.arange(np.size(ordered)), index=ordered)
    if not zero_index:
        ranks += 1
    return ranks


def plot_importances(
    clf,
    names,
    normalise=True,
    num_to_keep=6,
    ax=None,
    title=None,
    **kwargs
):
    """Plot a model's most important features on a bar chart.

    Parameters
    ----------
    clf : Estimator
        The estimator from which to extract the importances. Either
        `clf.feature_importances_` must exist, or each element of
        `clf.estimators_` must have a `feature_importances_` attribute.
    names : pandas.Index
        The names of the features.
    normalise : bool, default: True
        If True, importance values will be normalised to the value of the
        most important feature.
    num_to_keep : int, default: 6
        The number of features to plot.
    ax : matplotlib.axes.Axes, optional
        If provided, the plot will be drawn in `ax`; otherwise, a new figure
        and set of axes will be created.
    title : str, optional
        Custom axes title for plot.
    **kwargs : dict
        Further keyword arguments to be passed to `Axes.barh`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
        The figure and axes for the plot.
    """
    figsize = kwargs.pop("figsize", (10, num_to_keep * 0.5))
    font_size = kwargs.pop("fontsize", num_to_keep * 2)
    xlabel_size = kwargs.pop("xlabel_size", font_size * 1.25)
    title_size = kwargs.pop("titlesize", font_size * 1.35)
    facecolor = kwargs.pop("facecolor", "lightgrey")
    edgecolor = kwargs.pop("edgecolor", "black")
    height = kwargs.pop("height", 1.0)

    importances = calculate_importances(clf, names)
    importances = importances.sort_values(ascending=False)
    if normalise:
        importances = importances / importances.max()
    if num_to_keep is None:
        num_to_keep = np.size(importances)

    ylocs = -1 * np.arange(num_to_keep)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    ax.barh(
        y=ylocs,
        width=importances.iloc[:num_to_keep],
        height=height,
        facecolor=facecolor,
        edgecolor=edgecolor,
        zorder=1,
        **kwargs,
    )
    ax.yaxis.set_ticks(ylocs)
    ax.yaxis.set_ticklabels(
        [
            format_feature_name(i)
            for i in importances.iloc[:num_to_keep].index
        ]
    )
    ax.tick_params(labelsize=font_size)
    ax.grid(linestyle="dashed", color="grey")
    ax.set_axisbelow(True)

    xlabel = "Gini importance"
    if normalise:
        xlabel = "Relative " + xlabel
    ax.set_xlabel(xlabel, fontsize=xlabel_size)
    if title is None:
        title = "Feature importances"
        if num_to_keep is not None:
            title += f" (top {num_to_keep} features)"
    ax.set_title(title, fontsize=title_size)

    return fig, ax


def plot_correlations(
    clfs,
    names,
    verbose=False,
    title=None,
    alternative="greater",
    kendalltau_kw=None,
    text_kw=None,
    **kwargs
):
    """Plot Kendall's Tau rank correlations between feature importance rankings
    for different models.

    Parameters
    ----------
    clfs : dict[str, Estimator]
        Dictionary of estimators, with keys corresponding to the names
        of the different models.
    names : pandas.Index
        The names of the features.
    verbose : bool, default: False
        Print correlation values to stderr.
    title : str, optional
        Custom axes title for the plot.
    alternative : {'two-sided', 'less', 'greater'}, default: 'greater'
        Alternative hypothesis for `scipy.stats.kendalltau`.
    kendalltau_kw : dict, optional
        Further keyword arguments for `scipy.stats.kendalltau`.
    text_kw : dict, optional
        Further keyword arguments for `matplotlib.axes.Axes.text`.
    **kwargs : dict
        Further keyword arguments for `matplotlib.axes.Axes.matshow`.
    """
    if kendalltau_kw is None:
        kendalltau_kw = {}

    if text_kw is None:
        text_kw = {}
    color = text_kw.pop("color", "black")
    ha = text_kw.pop("ha", "center")
    va = text_kw.pop("va", "center")

    cmap = kwargs.pop("cmap", "plasma")
    interpolation = kwargs.pop("interpolation", "none")
    vmin = kwargs.pop("vmin", -1)
    vmax = kwargs.pop("vmax", 1)

    importances = {
        model_name: calculate_importances(model, names)
        for model_name, model in clfs.items()
    }
    ranks = {
        model_name: get_ranks(values)
        for model_name, values in importances.items()
    }
    ranks = pd.concat(
        [
            pd.DataFrame(rank_values, columns=[model_name])
            for model_name, rank_values in ranks.items()
        ],
        axis="columns",
    )

    columns = np.array(ranks.columns)
    n = len(columns)
    vals = np.empty((n, n))
    pvals = np.empty((n, n))
    for i, j in product(range(n), repeat=2):
        cola = columns[i]
        colb = columns[j]
        result = kendalltau(
            ranks[cola],
            ranks[colb],
            alternative=alternative,
            **kendalltau_kw,
        )
        vals[i, j] = result.statistic
        pvals[i, j] = result.pvalue

    if verbose:
        for i, j in combinations(range(n), 2):
            cola = columns[i]
            colb = columns[j]
            val = vals[i, j]
            pval = pvals[i, j]
            print(
                f"{cola}/{colb}: "
                + f"tau = {val:0.2f}, p = {pval:0.2f}",
                file=stderr,
            )

    fontsize = text_kw.pop("fontsize", n * 4)
    figsize = (n * 1.5, n * 1.5)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    cax = fig.add_axes([0.1, 0.05, 0.8, 0.03])
    im = ax.matshow(
        vals,
        interpolation=interpolation,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        **kwargs,
    )

    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.ax.set_xlabel(r"Kendall's $\tau$", fontsize=fontsize)
    cbar.ax.set_xticks(np.arange(-1, 1.5, 0.5))
    cbar.ax.tick_params(labelsize=fontsize)

    for i, j in product(range(n), repeat=2):
        text = (
            r"$\tau = "
            + f"{vals[i, j]:0.2f}"
            + "$"
            + "\n"
            + r"($p = "
            + f"{pvals[i, j]:0.2f}"
            + r"$)"
        )
        ax.text(
            j, i,
            text,
            color=color,
            ha=ha,
            va=va,
            fontsize=fontsize,
        )
    ax.set_xticks(range(n))
    ax.set_xticklabels(columns)
    ax.set_yticks(range(n))
    ax.set_yticklabels(columns)
    ax.tick_params(labelsize=fontsize, bottom=False)

    if title is None:
        title = (
            r"Correlation (Kendall's $\tau$)"
            + "\nof feature importance rankings"
        )
    fig.suptitle(
        title,
        fontsize=fontsize * 1.25,
        y=1.1,
    )

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Violin plot functions for feature distribution analysis
# ──────────────────────────────────────────────────────────────────────────────

_VIOLIN_ALPHA = 0.72  # shared translucency for all violin bodies


def _adjust_lightness(color, factor: float) -> tuple:
    """Adjust the lightness of *color* by *factor* in HLS space.

    Parameters
    ----------
    color : color-like
        Any matplotlib-parseable colour specification.
    factor : float
        Multiplier applied to the HLS L channel.  Values > 1 lighten;
        values < 1 darken.  The result is clamped to [0, 1].

    Returns
    -------
    tuple
        ``(R, G, B)`` floats in [0, 1].
    """
    import matplotlib.colors as mcolors

    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, max(0.0, min(1.0, l * factor)), s)


def create_violin_plot(
    feature_name: str,
    categories: list,
    training_data: pd.DataFrame,
    ax=None,
    data_source: str = "training",
    deployment_data: pd.DataFrame | None = None,
    figsize=None,
    tick_rotation: int = 30,
) -> "plt.Axes":
    """Draw a split violin plot for one feature across multiple categories.

    Each x-axis position represents a category (``"Overall"`` or a named
    metallogenic province).  The *left* half of each violin shows the
    **Positive** (deposit) class; the *right* half shows the **Unlabelled**
    class.  Every half-violin is scaled to the same maximum width regardless
    of sample size, so shapes are directly comparable across categories.
    Quartile lines (median + IQR) are drawn as dashed lines inside each half.

    Colours are sourced from :mod:`lib.colours` (Set2 for deposits, Dark2 for
    unlabelled), so they are automatically consistent across all thesis figures.

    Parameters
    ----------
    feature_name : str
        Column name in the data frames.  Must be a feature column in
        physical units (not standardised).
    categories : list[str]
        Ordered list of x-axis categories.  ``"Overall"`` applies no region
        filter; any other string is matched against the ``"region"`` column.
    training_data : pd.DataFrame
        Full training data frame (loaded in notebook 01c).  Positive rows are
        always sourced from here.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  If ``None`` a standalone figure and axes are
        created.
    data_source : {"training", "deployment"}
        Source for the Unlabelled class:

        * ``"training"`` — rows with ``label == "unlabelled"`` in
          *training_data*.
        * ``"deployment"`` — all rows of *deployment_data* (treated as
          unlabelled; no ``label`` column required).

    deployment_data : pd.DataFrame, optional
        Required when *data_source* is ``"deployment"``.
    figsize : tuple, optional
        ``(width, height)`` in inches for standalone mode.  Defaults to the
        thesis text-width at the golden-ratio aspect ratio.
    tick_rotation : int, default 30
        Rotation in degrees for the x-axis tick labels.  Use 0 for horizontal
        labels (may overlap with many categories).

    Returns
    -------
    matplotlib.axes.Axes
    """
    import seaborn as sns

    if data_source == "deployment" and deployment_data is None:
        raise ValueError("deployment_data must be provided when data_source='deployment'")

    # ── Build long-format DataFrame ───────────────────────────────────────────
    parts = []
    for cat in categories:
        # Positive class: always from training data
        if cat == "Overall":
            pos = training_data[training_data["label"] == "positive"][[feature_name]].copy()
        else:
            pos = training_data[
                (training_data["region"] == cat) & (training_data["label"] == "positive")
            ][[feature_name]].copy()
        pos = pos.rename(columns={feature_name: "value"})
        pos["category"] = cat
        pos["label"] = "positive"
        parts.append(pos)

        # Unlabelled class: from training or deployment data
        if data_source == "training":
            if cat == "Overall":
                unl = training_data[training_data["label"] == "unlabelled"][[feature_name]].copy()
            else:
                unl = training_data[
                    (training_data["region"] == cat) & (training_data["label"] == "unlabelled")
                ][[feature_name]].copy()
        else:
            if cat == "Overall":
                unl = deployment_data[[feature_name]].copy()
            else:
                unl = deployment_data[deployment_data["region"] == cat][[feature_name]].copy()
        unl = unl.rename(columns={feature_name: "value"})
        unl["category"] = cat
        unl["label"] = "unlabelled"
        parts.append(unl)

    long_df = pd.concat(parts, ignore_index=True).dropna(subset=["value"])

    # ── Create axes ───────────────────────────────────────────────────────────
    if ax is None:
        TW = 150 / 25.4
        if figsize is None:
            figsize = (TW, TW * 0.62)
        _fig, ax = plt.subplots(figsize=figsize)

    # ── Draw split violin ─────────────────────────────────────────────────────
    # Placeholder palette — violin bodies are recoloured below.
    sns.violinplot(
        data=long_df,
        x="category",
        y="value",
        hue="label",
        hue_order=["positive", "unlabelled"],
        order=categories,
        split=True,
        inner="quartile",
        density_norm="width",
        palette={"positive": "0.75", "unlabelled": "0.55"},
        linewidth=0.0,
        inner_kws={"color": "white", "linewidth": 0.7, "linestyle": "--"},
        ax=ax,
    )

    # ── Recolour violin bodies ────────────────────────────────────────────────
    # seaborn 0.13 with split=True adds PolyCollection objects in the order:
    #   [cat0/positive, cat0/unlabelled, cat1/positive, cat1/unlabelled, ...]
    # (one pair per x-axis position, left = hue_order[0], right = hue_order[1])
    # Deposits → Set2 (light), Unlabelled → Dark2 (dark) from lib.colours.
    violin_polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
    for i, cat in enumerate(categories):
        idx_pos = 2 * i
        idx_unl = 2 * i + 1
        if idx_pos < len(violin_polys):
            violin_polys[idx_pos].set_facecolor(region_colour(cat, "light"))
            violin_polys[idx_pos].set_alpha(_VIOLIN_ALPHA)
        if idx_unl < len(violin_polys):
            violin_polys[idx_unl].set_facecolor(region_colour(cat, "dark"))
            violin_polys[idx_unl].set_alpha(_VIOLIN_ALPHA)

    # ── Axes decoration ───────────────────────────────────────────────────────
    ax.set_ylabel(format_feature_name(feature_name))
    ax.set_xlabel("")
    # Solid, light gridlines — override thesis.mplstyle's dashed default.
    ax.grid(axis="y", linestyle="-", linewidth=0.3, color="#E8E8E8")
    ax.set_axisbelow(True)
    ax.margins(x=0.08, y=0.05)

    # thesis.mplstyle sets tick label colour to 0.5 grey; x-labels must be black.
    ax.tick_params(axis="x", rotation=tick_rotation, labelcolor="black")
    if tick_rotation != 0:
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")

    # Remove seaborn's per-axes legend; grid function adds a shared one.
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    return ax


def plot_feature_violin_grid(
    feature_importances: pd.DataFrame,
    layout: tuple,
    categories: list,
    training_data: pd.DataFrame,
    data_source: str = "training",
    deployment_data: pd.DataFrame | None = None,
    figsize=None,
    tick_rotation: int = 30,
) -> "plt.Figure":
    """Create a multi-panel violin grid ranked by feature importance.

    Panels are ordered left-to-right, top-to-bottom by descending feature
    importance rank (column order of *feature_importances*).  Each panel is
    produced by :func:`create_violin_plot`.  Subplot labels ``(a)``, ``(b)``,
    … are placed just outside the top-left corner of each panel.  A single
    shared legend is placed at the bottom of the figure.

    Colours are sourced automatically from :mod:`lib.colours` (Set2 for
    deposits, Dark2 for unlabelled).

    Parameters
    ----------
    feature_importances : pd.DataFrame
        Wide DataFrame as saved by notebook 01c — columns are feature names
        sorted by ``median()`` importance descending, rows are fold × estimator
        samples.  The first column is the most important feature.  Load with::

            pd.read_csv(gini_importance_basename.with_suffix(".csv"))

    layout : tuple[int, int]
        ``(n_rows, n_cols)`` for the subplot grid.
    categories : list[str]
        Category names for the x-axis of each violin.  Passed directly to
        :func:`create_violin_plot`.
    training_data : pd.DataFrame
        Full training data frame passed to :func:`create_violin_plot`.
    data_source : {"training", "deployment"}
        Passed to :func:`create_violin_plot`.
    deployment_data : pd.DataFrame, optional
        Passed to :func:`create_violin_plot`.
    figsize : tuple, optional
        Figure size in inches.  Defaults to thesis text-width with height
        scaled by the layout aspect ratio.
    tick_rotation : int, default 30
        Passed to :func:`create_violin_plot`.  Rotation in degrees for
        x-axis tick labels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ── Rank features by importance ───────────────────────────────────────────
    ranked_features = list(feature_importances.median().sort_values(ascending=False).index)

    # ── Create figure ─────────────────────────────────────────────────────────
    n_rows, n_cols = layout
    TW = 150 / 25.4
    if figsize is None:
        panel_h = TW / n_cols * 1.3
        figsize = (TW, panel_h * n_rows)

    # sharex so all panels share the same category x-axis; tick labels are
    # shown only on the bottom row (handled explicitly below).
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    axes_flat = list(np.asarray(axes).flat)

    # ── Draw panels ───────────────────────────────────────────────────────────
    n_features = len(ranked_features)
    for i, ax in enumerate(axes_flat):
        if i < n_features:
            create_violin_plot(
                feature_name=ranked_features[i],
                categories=categories,
                training_data=training_data,
                ax=ax,
                data_source=data_source,
                deployment_data=deployment_data,
                tick_rotation=tick_rotation,
            )
            # Subplot label: bold, outside the axes frame, flush with top edge.
            ax.text(
                -0.02, 1.02,
                f"({chr(ord('a') + i)})",
                transform=ax.transAxes,
                va="bottom", ha="right",
                fontsize=plt.rcParams.get("font.size", 8),
                fontweight="bold",
                clip_on=False,
            )
        else:
            ax.set_visible(False)

    # ── Hide x-tick labels on all but the bottom row ──────────────────────────
    for i, ax in enumerate(axes_flat):
        if not ax.get_visible():
            continue
        row = i // n_cols
        if row < n_rows - 1:
            # Hide both labels AND tick marks for rows that share but don't display.
            ax.tick_params(axis="x", labelbottom=False, bottom=False)

    # ── Shared legend ─────────────────────────────────────────────────────────
    # Two class-indicator patches (light/dark grey) + one coloured patch per
    # category to identify the province colours.
    # Class-indicator swatches use the "Overall" grey pair from the palette.
    legend_handles = [
        Patch(facecolor=region_colour("Overall", "light"),
              alpha=_VIOLIN_ALPHA, label="Deposits"),
        Patch(facecolor=region_colour("Overall", "dark"),
              alpha=_VIOLIN_ALPHA, label="Unlabelled"),
    ]
    for cat in categories:
        legend_handles.append(
            Patch(facecolor=region_colour(cat, "light"), alpha=_VIOLIN_ALPHA, label=cat)
        )

    n_cols_legend = min(len(legend_handles), 4)
    n_legend_rows = (len(legend_handles) + n_cols_legend - 1) // n_cols_legend
    # Reserve just enough room for the legend rows; tight_layout handles the rest.
    legend_frac = 0.01 + n_legend_rows * 0.035

    fig.tight_layout(rect=[0, legend_frac, 1, 1])
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        bbox_transform=fig.transFigure,
        ncols=n_cols_legend,
        fontsize=plt.rcParams.get("xtick.labelsize", 7),
        frameon=False,
        handlelength=0.9,
        handleheight=0.9,
        handletextpad=0.4,
        columnspacing=0.6,
        labelspacing=0.3,
        borderpad=0,
    )

    return fig
