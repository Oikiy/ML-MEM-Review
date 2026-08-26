import glob
import joblib
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split

import sys

SRC_PATH = Path(__file__).resolve().parent
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from utl_config_loader import get_model_dir, load_config
from utl_modelling import _infer_task, _resolve_model_path
from utl_model_arch import is_tree_model, normalize_model_key

FONT_SIZE = 20
FIG_SIZE = (20, 16)


def _resolve_config(config=None):
    return load_config() if config is None else config


def _get_plotting_cfg(config=None):
    return _resolve_config(config).get("plotting", {})


def _build_custom_cmap(plotting_cfg):
    cmap_cfg = plotting_cfg.get("custom_colormap", {})
    return LinearSegmentedColormap.from_list(
        cmap_cfg.get("name", "my_palette"),
        cmap_cfg.get("colors", ["#7FAFD9", "#FFF8EE", "#E38AAA"]),
    )


def _infer_model_task_from_target(y, config=None):
    """Return task name used by model filenames (classification / regression)."""
    requested = "auto"
    if config is not None:
        requested = config.get("modeling", {}).get("task", "auto")
    return _infer_task(requested, None, y)

def shap_explain_target(
    target,
    df_enriched,
    features_to_model,
    config=None,
    random_state=None,
    max_samples=1000,
    fig_size=None,
 ):
    """
    Load a saved ExtraTrees pipeline for a given target, rebuild training-like
    inputs, and generate SHAP summary plots.

    Parameters
    ----------
    target : str
        Name of the target variable (used in both build_feature_table and model filename).
    df_enriched : pandas.DataFrame
        Enriched membrane dataset (after feature_engineering).
    features_to_model : list of str
        Base feature columns to pass into build_feature_table.
    config : dict
        Loaded project configuration dictionary.
        Filename is resolved from modeling.model_filename_pattern and
        modeling.model_selection in config.yaml, with task inferred as
        "regressor" or "classifier".
    random_state : int, optional
        Random seed for train/test split used to construct SHAP background data.
        If None, uses config["modeling"]["random_state"].
    max_samples : int, optional
        Maximum number of samples to use for SHAP (subsampled if larger).
    fig_size : tuple, optional
        Figure size for SHAP summary and bar plots.
        If None, uses config["plotting"]["global_figure_size"].

    Returns
    -------
    shap_values : numpy.ndarray
        SHAP values for the sampled background set.
    feature_names : list of str
        Names of the transformed features used in the SHAP plots.
    summary_fig : matplotlib.figure.Figure
        SHAP beeswarm summary figure.
    bar_fig : matplotlib.figure.Figure
        SHAP mean absolute contribution bar figure.
    """
    try:
        import shap
    except Exception as exc:
        raise ImportError(
            "SHAP could not be imported. Install with: pip install -e '.[shap]'. "
            "A broken PyTorch install can also block shap; reinstall shap/torch if needed."
        ) from exc

    print(f"\n=== SHAP interpretation for target: {target} ===")

    config = _resolve_config(config)
    modeling_cfg = config.get("modeling", {})

    # Same task inference as train_model (honours modeling.task + auto rules).
    y_full = df_enriched[target].dropna()
    inferred_task = _infer_model_task_from_target(y_full, config=config)
    plotting_cfg = _get_plotting_cfg(config)
    if random_state is None:
        random_state = modeling_cfg.get("random_state", 42)
    if fig_size is None:
        fig_size = tuple(plotting_cfg.get("global_figure_size", FIG_SIZE))
    font_size = plotting_cfg.get("font_size", FONT_SIZE)
    colors = plotting_cfg.get("colors", {})
    summary_cmap = _build_custom_cmap(plotting_cfg)
    bar_color = colors.get("scatter", "#7FAFD9")

    selected_model_name = modeling_cfg.get("model_selection", "extratrees")
    model_filename_pattern = modeling_cfg.get("model_filename_pattern", "{target}_{model}_{task}.pkl")
    model_dir = get_model_dir(config)

    # Resolve model path from config naming pattern and inferred task label.
    model_path = _resolve_model_path(
        model_dir=model_dir,
        model_filename_pattern=model_filename_pattern,
        target=target,
        task=inferred_task,
        model=selected_model_name,
    )
    if not os.path.exists(model_path):
        available_pickles = sorted(glob.glob(os.path.join(model_dir, "*.pkl")))
        raise FileNotFoundError(
            "No saved model file found for SHAP target "
            f"'{target}' with inferred task '{inferred_task}' and model '{selected_model_name}'. "
            f"Expected: {model_path}. "
            f"Available .pkl files in model_dir: {available_pickles}"
        )

    pipe = joblib.load(model_path)
    print(f"Loaded model from: {model_path}")

    # Rebuild model inputs from the SAVED pipeline's expected columns so the
    # SHAP background sample matches the fitted feature representation exactly.
    preprocess_only = pipe.named_steps["preprocess"]
    expected_columns = list(preprocess_only.feature_names_in_)
    missing_columns = [c for c in expected_columns if c not in df_enriched.columns]
    if missing_columns:
        raise KeyError(
            "Missing expected model input columns for SHAP background data: "
            f"{missing_columns}. Available: {sorted(df_enriched.columns.tolist())}"
        )

    X = df_enriched[expected_columns].copy()
    y = df_enriched[target].copy()
    valid_mask = y.notna()
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Use the fitted preprocessing step from the saved pipeline directly.
    model_step = pipe.named_steps["model"]
    model_key = normalize_model_key(selected_model_name)

    # Transform training features into the representation used by the model.
    # Do not refit here: refitting would desynchronize SHAP feature names from the saved model.
    X_train_trans = preprocess_only.transform(X_train)

    # Recover transformed feature names from the fitted preprocessing step.
    feature_names = [
        name.split("__", 1)[-1] if "__" in name else name
        for name in preprocess_only.get_feature_names_out()
    ]

    # Optional subsampling keeps SHAP responsive on larger datasets
    if X_train_trans.shape[0] > max_samples:
        idx = np.random.choice(X_train_trans.shape[0], max_samples, replace=False)
        X_shap = X_train_trans[idx]
    else:
        X_shap = X_train_trans

    # Tree models -> TreeExplainer; linear-ish models -> LinearExplainer; else skip.
    if is_tree_model(model_key):
        explainer = shap.TreeExplainer(model_step)
        shap_values = explainer.shap_values(X_shap)
    else:
        try:
            # Background sample for linear / coefficient-based explainers.
            bg_size = min(100, X_shap.shape[0])
            background = shap.sample(X_shap, bg_size) if hasattr(shap, "sample") else X_shap[:bg_size]
            explainer = shap.LinearExplainer(model_step, background)
            shap_values = explainer.shap_values(X_shap)
        except Exception as exc:  # noqa: BLE001
            print(
                f"SHAP is not supported for model '{model_key}' "
                f"({type(model_step).__name__}): {exc}. Skipping SHAP plots."
            )
            return None, feature_names, None, None

    # SHAP beeswarm: global importance + direction of effect
    plt.rcParams.update({'font.size': font_size})
    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=feature_names,
        show=False,
        cmap=summary_cmap,
        plot_size=fig_size,
    )
    summary_fig = plt.gcf()
    summary_fig.set_size_inches(*fig_size)
    summary_fig.subplots_adjust(left=0.32, right=0.98, top=0.92, bottom=0.12)
    plt.title(f"SHAP summary:{target}", fontsize=font_size, fontweight='bold')
    ax = plt.gca()
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')
    summary_fig = plt.gcf()

    # SHAP bar chart: mean absolute SHAP magnitude per feature
    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        plot_size=fig_size,
        color=bar_color,
    )
    bar_fig = plt.gcf()
    bar_fig.set_size_inches(*fig_size)
    bar_fig.subplots_adjust(left=0.32, right=0.98, top=0.92, bottom=0.12)
    ax = plt.gca()
    ax.grid(False)
    for patch in ax.patches:
        patch.set_facecolor(bar_color)
        patch.set_edgecolor(bar_color)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')
    plt.title(f"SHAP feature importance : {target}", fontsize=font_size, fontweight='bold')
    bar_fig = plt.gcf()

    return shap_values, feature_names, summary_fig, bar_fig