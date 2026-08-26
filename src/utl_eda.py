from typing import Any, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from utl_config_loader import load_config

FONT_SIZE = 20


def _resolve_config(config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return load_config() if config is None else config


def _get_plotting_cfg(config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return _resolve_config(config).get("plotting", {})


def _target_is_classification(
    series: pd.Series, config: Optional[dict[str, Any]] = None
) -> bool:
    task = "auto"
    if config is not None:
        task = config.get("modeling", {}).get("task", "auto")
    if task == "classification":
        return True
    if task == "regression":
        return False
    return not pd.api.types.is_numeric_dtype(series)


def _build_custom_cmap(plotting_cfg: dict[str, Any]) -> LinearSegmentedColormap:
    cmap_cfg = plotting_cfg.get("custom_colormap", {})
    return LinearSegmentedColormap.from_list(
        cmap_cfg.get("name", "my_palette"),
        cmap_cfg.get("colors", ["#7FAFD9", "#FFF8EE", "#E38AAA"]),
    )


def plot_target_distributions(
    df: pd.DataFrame,
    target_columns: Sequence[str],
    fig_size: Optional[Tuple[float, float]] = None,
    colors: Optional[Sequence[str]] = None,
    violin_colors: Optional[Sequence[str]] = None,
    scatter_color: str = "#52687e",
    median_color: str = "#a73b3e",
    outlier_color: str = "#872d30",
    edge_color: str = "#81858a",
    jitter_std: float = 0.03,
    xlim: Tuple[float, float] = (-0.6, 0.6),
    config: Optional[dict[str, Any]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot target columns: violin + box + scatter for numeric values, class-count
    bars for classification labels.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the target columns.
    target_columns : sequence of str
        Names of the columns in df to plot.
    fig_size : tuple, optional
        Overall figure size.
    colors : sequence of str, optional
        Colors for the boxplots. Cycled if fewer than n_targets.
    violin_colors : sequence of str, optional
        Colors for the violin bodies. Cycled if fewer than n_targets.
    scatter_color : str
        Color for individual data points.
    median_color : str
        Color for the median line in the boxplot.
    outlier_color : str
        Color for outlier markers.
    edge_color : str
        Color for box/violin edges and spines.
    jitter_std : float
        Standard deviation of x-jitter for scatter points.
    xlim : tuple
        x-axis limits for each subplot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
    """
    plotting_cfg = _get_plotting_cfg(config)
    font_size = plotting_cfg.get("font_size", FONT_SIZE)
    plot_colors = plotting_cfg.get("colors", {})

    # Extract target data
    y_regression = df[target_columns]

    n_targets = len(y_regression.columns)
    if fig_size is None:
        fig_size = (6.5 * n_targets, 12)

    fig, axes = plt.subplots(1, n_targets, figsize=fig_size)
    fig.subplots_adjust(wspace=0.35)
    axes = np.atleast_1d(axes).flatten()

    # Default color schemes if none provided
    if colors is None:
        colors = plot_colors.get("boxplot", ["#25517e", "#4c7da1", "#cd7a69"])
    if violin_colors is None:
        violin_colors = plot_colors.get("violin", ["#cedae6", "#abc5d3", "#e9c2b3"])
    scatter_color = plot_colors.get("scatter", scatter_color)
    median_color = plot_colors.get("median", median_color)
    outlier_color = plot_colors.get("outlier", outlier_color)
    edge_color = plot_colors.get("edge", edge_color)

    for i, column in enumerate(y_regression.columns):
        ax = axes[i]
        y_data = y_regression[column].dropna()

        if _target_is_classification(y_data, config):
            counts = y_data.astype(str).value_counts()
            xpos = np.arange(len(counts))
            ax.bar(
                xpos,
                counts.to_numpy(),
                color=colors[i % len(colors)],
                edgecolor=edge_color,
                linewidth=2,
                alpha=0.8,
            )
            ax.set_xticks(xpos)
            ax.set_xticklabels(list(counts.index))
            ax.set_ylabel(
                "count", fontsize=font_size, fontweight="bold", color="#252525"
            )
            ax.set_xlabel(
                column, fontsize=font_size, fontweight="bold", color="#252525"
            )
            ax.tick_params(
                axis="both", which="major", labelsize=font_size, colors="#252525"
            )
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        else:
            parts = ax.violinplot(
                [y_data],
                positions=[0],
                widths=0.5,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )

            for pc in parts["bodies"]:
                pc.set_facecolor(violin_colors[i % len(violin_colors)])
                pc.set_alpha(0.4)
                pc.set_edgecolor(edge_color)
                pc.set_linewidth(1.5)

            ax.boxplot(
                [y_data],
                positions=[0],
                widths=0.25,
                patch_artist=True,
                medianprops=dict(color=median_color, linewidth=3),
                boxprops=dict(
                    facecolor=colors[i % len(colors)],
                    edgecolor=edge_color,
                    linewidth=2,
                    alpha=0.8,
                ),
                whiskerprops=dict(color=edge_color, linewidth=2),
                capprops=dict(color=edge_color, linewidth=2),
                flierprops=dict(
                    marker="D",
                    markerfacecolor=outlier_color,
                    markersize=8,
                    linestyle="none",
                    markeredgecolor=edge_color,
                    markeredgewidth=1,
                    alpha=0.7,
                ),
            )

            x_data = np.random.normal(0, jitter_std, size=len(y_data))
            ax.scatter(
                x_data,
                y_data,
                alpha=0.25,
                s=30,
                color=scatter_color,
                edgecolors="none",
            )

            ax.set_xlim(*xlim)
            ax.set_ylabel(
                column, fontsize=font_size, fontweight="bold", color="#252525"
            )
            ax.set_xlabel("", fontsize=font_size)
            ax.tick_params(
                axis="both", which="major", labelsize=font_size, colors="#252525"
            )
            ax.set_xticks([])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))

        ax.grid(False)

        # Spines
        for spine_name in ['top', 'right', 'left', 'bottom']:
            spine = ax.spines[spine_name]
            spine.set_linewidth(2)
            spine.set_edgecolor(edge_color)
    
    return fig, axes


def compute_and_plot_pearson_heatmap(
    df,
    threshold=None,
    fig_size=(12, 10),
    config: Optional[dict[str, Any]] = None,
    show_annotations: Optional[bool] = None,
):
    """
    Compute pairwise Pearson correlations and plot heatmap with correlation threshold filtering.
    
    This function handles missing values by computing correlations only on valid pairs
    and creates a symmetric correlation matrix for visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with numerical features
    threshold : float, default=0.5
        Correlation threshold (0-1). Only correlations with absolute value > threshold are displayed
    fig_size : tuple, default=(12, 10)
        Figure size for the heatmap.
    show_annotations : bool, optional
        Whether to draw numeric correlation annotations inside cells.
        If None, this is chosen automatically based on matrix width.

    Returns:
    --------
    pandas.DataFrame
        Complete correlation matrix with all pairwise correlations
    """
    plotting_cfg = _get_plotting_cfg(config)
    data_analysis_cfg = _resolve_config(config).get("data_analysis", {})
    font_size = plotting_cfg.get("font_size", FONT_SIZE)
    if threshold is None:
        thresholds = data_analysis_cfg.get("correlation_thresholds", [0.5])
        threshold = thresholds[-1] if thresholds else 0.5
    if fig_size is None:
        fig_size = tuple(plotting_cfg.get("global_figure_size", (12, 10)))

    cols = df.columns
    n = len(cols)
    # Initialize correlation matrix with NaN values
    corr_matrix = pd.DataFrame(np.nan, index=cols, columns=cols)
    
    # Compute pairwise correlations, handling missing values
    for i in range(n):
        for j in range(i, n):
            col_i, col_j = cols[i], cols[j]
            # Only use rows where both columns have valid (non-NaN) values
            valid_rows = df[[col_i, col_j]].dropna()
            if len(valid_rows) > 1:
                corr = valid_rows.corr().iloc[0, 1]
            else:
                corr = np.nan
            # Set both upper and lower triangular elements for symmetry
            corr_matrix.at[col_i, col_j] = corr
            corr_matrix.at[col_j, col_i] = corr
    
    # Plot heatmap with threshold filtering
    try:
        import seaborn as sns
    except Exception as exc:
        raise ImportError(
            "seaborn is required for compute_and_plot_pearson_heatmap()."
        ) from exc

    # Use an explicit mask and adaptive annotation behavior for dense matrices.
    corr_for_plot = corr_matrix.copy()
    mask = corr_for_plot.abs() <= float(threshold)
    n_cols = max(1, len(corr_for_plot.columns))
    annot_font_size = min(font_size, max(6, int(240 / n_cols)))
    if show_annotations is None:
        # Default to showing annotations unless explicitly disabled in config.
        show_annotations = bool(data_analysis_cfg.get("pearson_heatmap_show_annotations", True))

    fig, ax = plt.subplots(figsize=fig_size)
    hm = sns.heatmap(
        corr_for_plot,
        mask=mask,
        annot=bool(show_annotations),
        fmt=".2f",
        annot_kws={"size": annot_font_size, "color": "black"},
        cmap=_build_custom_cmap(plotting_cfg),
        center=0,
        cbar_kws={"shrink": .8},
        square=True,
        linewidths=.5,
        linecolor='black',
        ax=ax,
    )
    
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=font_size, rotation=45, ha='right', fontfamily='Arial')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=font_size, rotation=0, fontfamily='Arial')

    # Remove grid and tick marks
    ax.grid(False)
    ax.tick_params(which='both', length=0)

    # Thicken the outer frame
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(3)
        spine.set_edgecolor('black')

    # Match colorbar style used in ellipse heatmap
    cbar = hm.collections[0].colorbar
    cbar.set_label('Correlation', fontsize=font_size, fontfamily='Arial')
    cbar.ax.tick_params(labelsize=font_size)
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily('Arial')
    cbar.outline.set_linewidth(3)
    cbar.outline.set_edgecolor('black')

    plt.title(f'Pairwise Pearson Correlation Heatmap > {threshold}', fontsize=font_size, fontfamily='Arial', fontweight='bold')
    plt.tight_layout()
    fig = plt.gcf()
    # Return (corr_matrix, fig). Callers that only need the matrix can unpack: corr, _ = ...
    return corr_matrix, fig
