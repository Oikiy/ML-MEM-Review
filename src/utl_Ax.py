
import os
import sys
import joblib
import pandas as pd
import warnings
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig, ChoiceParameterConfig

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models")
CODE_PATH = os.path.join(PROJECT_ROOT, "src")
CATION_PROP_PATH = os.path.join(DATA_PATH, "cation_prop.xlsx")
ANION_PROP_PATH = os.path.join(DATA_PATH, "anion_prop.xlsx")

if CODE_PATH not in sys.path:
    sys.path.insert(0, CODE_PATH)

from utl_feateng import feature_engineering

def normalize(value: float, norm_dict: dict, key: str) -> float:
    """
    Normalize a metric into [0, 1] based on fixed min/max.

    Parameters
    ----------
    value : float
        Raw metric value.
    key : str
        Metric name ('PureWaterPermeability' or 'solute1Rejection').

    Returns
    -------
    float
        Normalized value between 0 and 1.
    """
    vmin, vmax = norm_dict[key]
    return (float(value) - vmin) / (vmax - vmin)


def scalar_objective(pwp_norm: float, rej_norm: float, weight_dict: dict) -> float:
    """
    Combine normalized permeability and rejection into a single scalar objective.

    Parameters
    ----------
    pwp_norm : float
        Normalized pure water permeability.
    rej_norm : float
        Normalized solute1 rejection.

    Returns
    -------
    float
        Scalarized objective in [0, 1], larger is better.
    """
    return (
        weight_dict["PureWaterPermeability"] * pwp_norm
        + weight_dict["solute1Rejection"] * rej_norm
    )


def _normalize_optimization_mode(optimization_mode: str) -> str:
    mode = optimization_mode.strip().lower()
    if mode not in {"scalarized", "multiobjective"}:
        raise ValueError(
            "optimization_mode must be 'scalarized' or 'multiobjective', "
            f"got {optimization_mode}"
        )
    return mode


def _build_optimization_config(optimization_mode: str, objective_name: str) -> dict:
    mode = _normalize_optimization_mode(optimization_mode)
    if mode == "scalarized":
        return {"objective": objective_name}
    return {"objective": "normPureWaterPermeability,solute1Rejection"}


def _build_multiobjective_thresholds(objective_thresholds=None):
    # Provide explicit thresholds to avoid Ax inferring unstable defaults across runs.
    if objective_thresholds is None:
        return [
            "normPureWaterPermeability>=0.0",
            "solute1Rejection>=0.0",
        ]
    return list(objective_thresholds)


def prepare_objective_history(
    trial_df: pd.DataFrame,
    objective_names,
    minimize_metrics=None,
    trial_index_column: str = "trial_index",
    aggregate_name: str = "objective_sum",
) -> pd.DataFrame:
    history = trial_df.loc[:, list(objective_names)].copy()
    minimize_metrics = set(minimize_metrics or [])

    for objective_name in objective_names:
        series = pd.to_numeric(history[objective_name], errors="coerce")
        if objective_name in minimize_metrics:
            series = -series
        history[objective_name] = series

    if trial_index_column in trial_df.columns:
        history[trial_index_column] = pd.to_numeric(
            trial_df[trial_index_column], errors="coerce"
        )
    else:
        history[trial_index_column] = range(len(history))

    history[aggregate_name] = history.loc[:, list(objective_names)].sum(axis=1)
    return history


def plot_objective_evolution(
    history_df: pd.DataFrame,
    objective_names,
    trial_index_column: str = "trial_index",
    aggregate_name: str = "objective_sum",
    xtick_step: int = 10,
    output_dir: str | None = None,
    aggregate_title: str = "Evolution of Sum of Objective Values",
):
    import matplotlib.pyplot as plt
    import numpy as np

    trial_indices = history_df[trial_index_column].tolist()
    figures = {}

    for objective_name in objective_names:
        objective_values = pd.to_numeric(history_df[objective_name], errors="coerce")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(trial_indices, objective_values, marker="o", linestyle="-")
        ax.set_title(f"Evolution of Objective Value Over Trials on {objective_name}")
        ax.set_xlabel("Trial Index")
        ax.set_ylabel("Objective Value")
        if trial_indices:
            ax.set_xticks(np.arange(min(trial_indices), max(trial_indices) + 5, xtick_step))
            ax.set_xlim(0, max(trial_indices) + 5)
        if output_dir is not None:
            fig.savefig(
                os.path.join(output_dir, f"{objective_name}_evolution.png"),
                dpi=300,
                bbox_inches="tight",
            )
        figures[objective_name] = fig

    aggregate_values = pd.to_numeric(history_df[aggregate_name], errors="coerce")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(trial_indices, aggregate_values, marker="o", linestyle="-")
    for index in range(1, len(trial_indices)):
        color = "green" if aggregate_values.iloc[index] < aggregate_values.iloc[index - 1] else "red"
        ax.arrow(
            trial_indices[index - 1],
            aggregate_values.iloc[index - 1],
            trial_indices[index] - trial_indices[index - 1],
            aggregate_values.iloc[index] - aggregate_values.iloc[index - 1],
            head_width=0.5,
            head_length=0.01,
            fc=color,
            ec=color,
            length_includes_head=True,
        )
    ax.set_title(aggregate_title)
    ax.set_xlabel("Trial Index")
    ax.set_ylabel("Sum of Objective Values")
    if trial_indices:
        ax.set_xlim(0, max(trial_indices) + 5)
    if output_dir is not None:
        fig.savefig(
            os.path.join(output_dir, f"{aggregate_name}_evolution.png"),
            dpi=300,
            bbox_inches="tight",
        )
    figures[aggregate_name] = fig
    return figures


def build_frontier_dataframe(client, use_model_predictions: bool = False) -> pd.DataFrame:
    try:
        frontier = client.get_pareto_frontier(use_model_predictions=use_model_predictions)
    except TypeError:
        frontier = client.get_pareto_frontier()

    rows = []
    for item in frontier:
        if isinstance(item, tuple):
            params = item[0] if len(item) > 0 else {}
            metrics = item[1] if len(item) > 1 else {}
        elif isinstance(item, dict):
            params = item.get("parameterization") or item.get("parameters") or item.get("params") or {}
            metrics = item.get("metrics") or item.get("means") or {}
        else:
            continue

        flattened_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (tuple, list)) and len(value) >= 1:
                flattened_metrics[key] = value[0]
            else:
                flattened_metrics[key] = value
        rows.append({**params, **flattened_metrics})

    return pd.DataFrame(rows)


def build_rbf_surface(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    grid_size: int = 100,
    smoothing: float = 0.0001,
    kernel: str = "thin_plate_spline",
    y_padding: float = 0.0,
) -> dict:
    import numpy as np
    from scipy.interpolate import RBFInterpolator

    clean = data.dropna(subset=[x_col, y_col, z_col]).copy()
    x = clean[x_col].to_numpy(dtype=float)
    y = clean[y_col].to_numpy(dtype=float)
    z = clean[z_col].to_numpy(dtype=float)

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min - y_padding, y_max, grid_size)
    X, Y = np.meshgrid(xi, yi)

    points = np.column_stack((x, y))
    interpolator = RBFInterpolator(points, z, kernel=kernel, smoothing=smoothing)
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    Z = interpolator(grid_points).reshape(X.shape)

    return {
        "data": clean,
        "x": x,
        "y": y,
        "z": z,
        "points": points,
        "rbf": interpolator,
        "X": X,
        "Y": Y,
        "Z": Z,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }


def _darker_color(color, factor: float = 0.6):
    import numpy as np
    from matplotlib.colors import to_rgb

    return tuple(np.clip(np.array(to_rgb(color)) * factor, 0, 1))


def plot_inverse_design_surface(
    surface_data: dict,
    x_col: str,
    y_col: str,
    z_col: str,
    frontier_df: pd.DataFrame | None = None,
    cmap="viridis",
    trial_color: str = "r",
    frontier_color: str = "b",
    marker_size: int = 100,
    elev: float = 30,
    azim: float = -45,
    y_padding: float = 0.0,
):
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        surface_data["x"],
        surface_data["y"],
        surface_data["z"],
        c=trial_color,
        marker="o",
        label="Trials",
        edgecolors=[_darker_color(trial_color)],
        linewidths=0.9,
    )

    if frontier_df is not None and not frontier_df.empty:
        frontier_clean = frontier_df.dropna(subset=[x_col, y_col]).copy()
        if z_col not in frontier_clean.columns:
            frontier_points = frontier_clean[[x_col, y_col]].to_numpy(dtype=float)
            frontier_clean[z_col] = np.asarray(surface_data["rbf"](frontier_points)).reshape(-1)
        else:
            frontier_clean = frontier_clean.dropna(subset=[z_col])
        ax.scatter(
            frontier_clean[x_col],
            frontier_clean[y_col],
            frontier_clean[z_col],
            c=frontier_color,
            marker="^",
            s=marker_size,
            label="Pareto Frontier",
            edgecolors=[_darker_color(frontier_color)],
            linewidths=0.9,
        )

    surf = ax.plot_surface(
        surface_data["X"],
        surface_data["Y"],
        surface_data["Z"],
        cmap=cmap,
        edgecolor="none",
        alpha=0.8,
        rstride=1,
        cstride=1,
        shade=True,
        antialiased=True,
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_xlim([surface_data["x_min"], surface_data["x_max"]])
    ax.set_ylim([surface_data["y_min"] - y_padding, surface_data["y_max"]])
    ax.set_zlim([float(surface_data["z"].min()), float(surface_data["z"].max())])
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper left", fontsize=12)
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.12, aspect=10, label=z_col)
    return fig, ax


def plot_inverse_design_contour(
    surface_data: dict,
    x_col: str,
    y_col: str,
    z_col: str,
    frontier_df: pd.DataFrame | None = None,
    cmap="viridis",
    trial_color: str = "r",
    frontier_color: str = "b",
    marker_size: int = 20,
    y_padding: float = 0.0,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(surface_data["X"], surface_data["Y"], surface_data["Z"], levels=50, cmap=cmap)
    fig.colorbar(contour, label=z_col)
    ax.scatter(
        surface_data["x"],
        surface_data["y"],
        c=trial_color,
        marker="o",
        s=marker_size,
        label="Trials",
        edgecolors=[_darker_color(trial_color)],
        linewidths=0.9,
    )

    if frontier_df is not None and not frontier_df.empty:
        frontier_clean = frontier_df.dropna(subset=[x_col, y_col])
        ax.scatter(
            frontier_clean[x_col],
            frontier_clean[y_col],
            c=frontier_color,
            marker="^",
            s=marker_size,
            label="Pareto Frontier",
            edgecolors=[_darker_color(frontier_color)],
            linewidths=0.9,
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_xlim([surface_data["x_min"], surface_data["x_max"]])
    ax.set_ylim([surface_data["y_min"] - y_padding, surface_data["y_max"]])
    ax.legend(loc="upper left", fontsize=12)
    return fig, ax

def oracle_from_params(
    params: dict,
    constant_params:dict,
    norm_dict: dict,
    # model_path_dict:dict,
    weight_dict: dict,
    cation_path:str,
    anion_path:str,
    optimization_mode: str = "scalarized",
    
) -> dict:
    mode = _normalize_optimization_mode(optimization_mode)

    # 1) Defaults for missing recipe/test variables (Constant or not included in search space).
    # Merge Ax parameters with defaults (Ax parameters override defaults).
    combined = {**constant_params, **params}

    # 2) Compute implied solvent mass fractions so each dope is approximately mass-balanced.
    # Updated column names to match the rest of the codebase:
    combined["Dope1_Solvent1_Mass"] = max(
        0.0,
        1.0
        - combined["Dope1_Monomer1_Mass"]
        - combined["Dope1_Add1_Mass"]
        - combined.get("Dope1_Monomer2_Mass", 0.0)
        - combined.get("Dope1_Solvent2_Mass", 0.0)
        - combined.get("Dope1_Add2_Mass", 0.0),
    )
    combined["Dope2_Solvent1_Mass"] = max(
        0.0,
        1.0
        - combined["Dope2_Monomer1_Mass"]
        - combined.get("Dope2_Monomer2_Mass", 0.0)
        - combined.get("Dope2_Solvent2_Mass", 0.0)
        - combined.get("Dope2_Add_Mass", 0.0),
    )

    # 3) Build one-row DataFrame for model inference and apply feature engineering to enrich with descriptors.
    X_raw = pd.DataFrame([combined])
    x_raw_enriched = feature_engineering(
        X_raw,
        cation_path,
        anion_path,
        verbose=False,
    )


    # 4) Predict with pre-trained models.
    # Resolve from project-level MODEL_PATH so loading is independent of current working directory.
    pwp_model_path = os.path.join(
        MODEL_PATH,
        "Pure_Water_Permeability_{Lm^{-2}^h{-1}bar^{-1}}_extratrees.pkl",
    )
    rej_model_path = os.path.join(
        MODEL_PATH,
        "solute1_Rejection_extratrees.pkl",
    )

    pipe_pwp = joblib.load(pwp_model_path)
    pipe_rej = joblib.load(rej_model_path)

    # Infer expected input columns directly from fitted preprocessors.
    # This removes the need for manually maintained feature_cols_* lists.
    def _extract_expected_input_columns(pipe, model_name: str):
        preprocess = pipe.named_steps.get("preprocess", None)
        if preprocess is not None and hasattr(preprocess, "feature_names_in_"):
            return list(preprocess.feature_names_in_)
        if hasattr(pipe, "feature_names_in_"):
            return list(pipe.feature_names_in_)
        raise ValueError(
            f"Could not infer input feature columns from pipeline for {model_name}. "
            "Expected a fitted pipeline with feature_names_in_."
        )

    feature_cols_pwp = _extract_expected_input_columns(pipe_pwp, "PureWaterPermeability")
    feature_cols_rej = _extract_expected_input_columns(pipe_rej, "solute1Rejection")

    # Validate required columns exist in current candidate row before prediction.
    missing_pwp = [c for c in feature_cols_pwp if c not in x_raw_enriched.columns]
    missing_rej = [c for c in feature_cols_rej if c not in x_raw_enriched.columns]
    if missing_pwp:
        raise KeyError(f"Missing columns for PWP model input: {missing_pwp}")
    if missing_rej:
        raise KeyError(f"Missing columns for rejection model input: {missing_rej}")

    # Keep exact training column order for stable inference.
    X_pwp = x_raw_enriched.reindex(columns=feature_cols_pwp)
    X_rej = x_raw_enriched.reindex(columns=feature_cols_rej)

    pwp_pred = pipe_pwp.predict(X_pwp)[0]
    rej_pred = pipe_rej.predict(X_rej)[0]

    # 5) Normalize and scalarize.
    pwp_norm = normalize(pwp_pred, norm_dict, "PureWaterPermeability")
    rej_norm = normalize(rej_pred, norm_dict, "solute1Rejection")
    scalar = scalar_objective(pwp_norm, rej_norm, weight_dict)

    metrics = {
        "normPureWaterPermeability": pwp_norm,
        "solute1Rejection": rej_norm,
    }

    if mode == "scalarized":
        metrics["scalarizedobjective"] = scalar

    return metrics

def run_ax_optimization(
    search_space_parameters,
    oracle_from_params,
    constant_params,
    norm_dict,
    weight_dict,
    cation_path,
    anion_path,
    optimization_mode="scalarized",
    experiment_name="NF_Inverse_Design_Tutorial",
    experiment_description="Toy inverse design of NF IP membrane recipe",
    objective_name="scalarizedobjective",
    objective_thresholds=None,
    sobol_init=5,
    n_trials=20,
    trial_error=0.01,
    verbose=True,
):
    client = Client()

    optimization_config = _build_optimization_config(optimization_mode, objective_name)
    if _normalize_optimization_mode(optimization_mode) == "multiobjective":
        # Ax client versions in this project accept outcome_constraints but not
        # objective_thresholds in configure_optimization.
        optimization_config["outcome_constraints"] = _build_multiobjective_thresholds(
            objective_thresholds
        )

    client.configure_experiment(
        parameters=search_space_parameters,
        name=experiment_name,
        description=experiment_description,
    )

    client.configure_optimization(**optimization_config)

    client.configure_generation_strategy(
        method="fast",
        initialization_budget=sobol_init,
    )

    results = []

    with pd.option_context("future.no_silent_downcasting", False), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Downcasting behavior in `replace` is deprecated.*",
            category=FutureWarning,
            module=r"ax\.adapter\.transforms\.choice_encode",
        )

        for i in range(n_trials):
            trials = client.get_next_trials(max_trials=1)

            for trial_index, trial_params in trials.items():
                if verbose:
                    print(f"\nTrial {trial_index}: parameters = {trial_params}")

                metrics_raw = oracle_from_params(
                    trial_params,
                    constant_params=constant_params,
                    norm_dict=norm_dict,
                    weight_dict=weight_dict,
                    cation_path=cation_path,
                    anion_path=anion_path,
                    optimization_mode=optimization_mode,
                )

                metrics = {k: (float(v), trial_error) for k, v in metrics_raw.items()}

                if verbose:
                    print("  metrics:", metrics)

                client.complete_trial(
                    trial_index=trial_index,
                    raw_data=metrics,
                )

                results.append(
                    {
                        "trial_index": trial_index,
                        "params": trial_params,
                        "metrics_raw": metrics_raw,
                        "metrics": metrics,
                    }
                )

    return client, results