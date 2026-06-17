import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
import sys
import os
PROJECT_ROOT         = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
CODE_PATH            = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, CODE_PATH)
from utl_modelling import _build_feature_table

FONT_SIZE = 20
FIG_SIZE = (20, 16)
custom_cmap = LinearSegmentedColormap.from_list(
    "my_palette",
    ["#7FAFD9", "#FFF8EE", "#E38AAA"],
)

def shap_explain_target(
    target,
    df_enriched,
    base_features,
    MODEL_PATH,
    random_state=42,
    max_samples=1000,
    fig_size=(20, 16),
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
    base_features : list of str
        Base feature columns to pass into build_feature_table.
    model_path : str, optional
        Directory where the fitted pipelines are stored.
        Models are assumed to be named f"{target}_extratrees.pkl".
    random_state : int, optional
        Random seed for train/test split used to construct SHAP background data.
    max_samples : int, optional
        Maximum number of samples to use for SHAP (subsampled if larger).
    fig_size : tuple, optional
        Figure size for SHAP summary and bar plots.

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
    print(f"\n=== SHAP interpretation for target: {target} ===")

    # Load the saved pipeline for this target
    model_path = os.path.join(MODEL_PATH, f"{target}_extratrees.pkl")
    pipe = joblib.load(model_path)
    print(f"Loaded model from: {model_path}")

    # Rebuild feature table to obtain training-like inputs for SHAP background data
    X, y = _build_feature_table(df_enriched, base_features=base_features, target=target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Use the fitted preprocessing step from the saved pipeline directly.
    preprocess_only = pipe.named_steps["preprocess"]
    etr = pipe.named_steps["model"]

    # Transform training features into the representation used by the tree model.
    # Do not refit here: refitting would desynchronize SHAP feature names from the saved model.
    X_train_trans = preprocess_only.transform(X_train)

    # Recover transformed feature names from the fitted preprocessing step.
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()

    onehot_feature_names = []
    if categorical_cols:
        ohe = preprocess_only.named_transformers_["cat"]
        onehot_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()

    feature_names = onehot_feature_names + numerical_cols

    # Build SHAP explainer for tree ensemble
    explainer = shap.TreeExplainer(etr)

    # Optional subsampling keeps SHAP responsive on larger datasets
    if X_train_trans.shape[0] > max_samples:
        idx = np.random.choice(X_train_trans.shape[0], max_samples, replace=False)
        X_shap = X_train_trans[idx]
    else:
        X_shap = X_train_trans

    shap_values = explainer.shap_values(X_shap)

    # SHAP beeswarm: global importance + direction of effect
    plt.rcParams.update({'font.size': FONT_SIZE})
    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=feature_names,
        show=False,
        cmap=custom_cmap,  # use the same custom colormap for consistency
        plot_size=fig_size,
    )
    summary_fig = plt.gcf()
    summary_fig.set_size_inches(*fig_size)
    summary_fig.subplots_adjust(left=0.32, right=0.98, top=0.92, bottom=0.12)
    plt.title(f"SHAP summary:{target}", fontsize=FONT_SIZE, fontweight='bold')
    ax = plt.gca()
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')

    plt.show()

    # SHAP bar chart: mean absolute SHAP magnitude per feature
    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        plot_size=fig_size,
        color="#7FAFD9",
    )
    bar_fig = plt.gcf()
    bar_fig.set_size_inches(*fig_size)
    bar_fig.subplots_adjust(left=0.32, right=0.98, top=0.92, bottom=0.12)
    ax = plt.gca()
    ax.grid(False)
    for patch in ax.patches:
        patch.set_facecolor("#7FAFD9")
        patch.set_edgecolor("#7FAFD9")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')
    plt.title(f"SHAP feature importance : {target}", fontsize=FONT_SIZE, fontweight='bold')
    plt.show()

    return shap_values, feature_names, summary_fig, bar_fig