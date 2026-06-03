import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

FONT_SIZE = 20
custom_cmap = LinearSegmentedColormap.from_list(
    "my_palette",
    ["#7FAFD9", "#FFF8EE", "#E38AAA"],
)

from typing import Sequence, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd


def plot_target_distributions(
    df: pd.DataFrame,
    target_columns: Sequence[str],
    figsize: Optional[Tuple[float, float]] = None,
    colors: Optional[Sequence[str]] = None,
    violin_colors: Optional[Sequence[str]] = None,
    scatter_color: str = "#52687e",
    median_color: str = "#a73b3e",
    outlier_color: str = "#872d30",
    edge_color: str = "#81858a",
    jitter_std: float = 0.03,
    xlim: Tuple[float, float] = (-0.6, 0.6),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot violin + boxplots with overlaid scatter for a set of regression targets.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the target columns.
    target_columns : sequence of str
        Names of the columns in df to plot.
    figsize : tuple, optional
        Overall figure size; defaults to (6.5 * n_targets, 12).
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
    # Extract target data
    y_regression = df[target_columns]

    n_targets = len(y_regression.columns)
    if figsize is None:
        figsize = (6.5 * n_targets, 12)

    fig, axes = plt.subplots(1, n_targets, figsize=figsize)
    fig.subplots_adjust(wspace=0.35)
    axes = np.atleast_1d(axes).flatten()

    # Default color schemes if none provided
    if colors is None:
        colors = ['#25517e', '#4c7da1', '#cd7a69']
    if violin_colors is None:
        violin_colors = ['#cedae6', '#abc5d3', '#e9c2b3']

    for i, column in enumerate(y_regression.columns):
        ax = axes[i]
        y_data = y_regression[column].dropna()

        # Violin plot as background
        parts = ax.violinplot(
            [y_data],
            positions=[0],
            widths=0.5,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        for pc in parts['bodies']:
            pc.set_facecolor(violin_colors[i % len(violin_colors)])
            pc.set_alpha(0.4)
            pc.set_edgecolor(edge_color)
            pc.set_linewidth(1.5)

        # Boxplot on top
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
                marker='D',
                markerfacecolor=outlier_color,
                markersize=8,
                linestyle='none',
                markeredgecolor=edge_color,
                markeredgewidth=1,
                alpha=0.7,
            ),
        )

        # Scatter of individual points
        x_data = np.random.normal(0, jitter_std, size=len(y_data))
        ax.scatter(
            x_data,
            y_data,
            alpha=0.25,
            s=30,
            color=scatter_color,
            edgecolors='none',
        )

        # Axes formatting
        ax.set_xlim(*xlim)
        ax.set_ylabel(column, fontsize=FONT_SIZE, fontweight='bold', color='#252525')
        ax.set_xlabel('', fontsize=FONT_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=20, colors='#252525')
        ax.set_xticks([])

        # Sparse y-ticks
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))

        ax.grid(False)

        # Spines
        for spine_name in ['top', 'right', 'left', 'bottom']:
            spine = ax.spines[spine_name]
            spine.set_linewidth(2)
            spine.set_edgecolor(edge_color)
    
    return fig, axes


def compute_and_plot_pearson_heatmap(df, threshold=0.5):
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

    Returns:
    --------
    pandas.DataFrame
        Complete correlation matrix with all pairwise correlations
    """
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
    fig, ax = plt.subplots()
    hm = sns.heatmap(corr_matrix.where(corr_matrix.abs() > threshold), annot=True, fmt=".2f",
                     annot_kws={"size": 20}, cmap=custom_cmap, center=0, cbar_kws={"shrink": .8}, square=True,
                     linewidths=.5, linecolor='black', ax=ax)
    
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=FONT_SIZE, rotation=45, ha='right', fontfamily='Arial')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=FONT_SIZE, rotation=0, fontfamily='Arial')

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
    cbar.set_label('Correlation', fontsize=FONT_SIZE, fontfamily='Arial')
    cbar.ax.tick_params(labelsize=FONT_SIZE)
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily('Arial')
    cbar.outline.set_linewidth(3)
    cbar.outline.set_edgecolor('black')

    plt.title(f'Pairwise Pearson Correlation Heatmap > {threshold}', fontsize=FONT_SIZE, fontfamily='Arial', fontweight='bold')
    plt.tight_layout()
    plt.show()
    return corr_matrix
