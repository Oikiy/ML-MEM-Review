
import os
import sys
import joblib
import pandas as pd
import warnings
from numbers import Number
from ax.api.client import Client
from matplotlib.colors import LinearSegmentedColormap
custom_cmap = LinearSegmentedColormap.from_list(
    "my_palette",
    ["#7FAFD9", "#FFF8EE", "#E38AAA"]
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models")
CODE_PATH = os.path.join(PROJECT_ROOT, "src")
CATION_PROP_PATH = os.path.join(DATA_PATH, "cation_prop.xlsx")
ANION_PROP_PATH = os.path.join(DATA_PATH, "anion_prop.xlsx")

if CODE_PATH not in sys.path:
    sys.path.insert(0, CODE_PATH)

from utl_feateng import feature_engineering
from utl_fileIO import file_io

def _normalize(value: float, norm_dict: dict, key: str) -> float:
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


def _scalar_objective(pwp_norm: float, rej_norm: float, weight_dict: dict) -> float:
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
    pwp_norm = _normalize(pwp_pred, norm_dict, "PureWaterPermeability")
    rej_norm = _normalize(rej_pred, norm_dict, "solute1Rejection")
    scalar = _scalar_objective(pwp_norm, rej_norm, weight_dict)

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
    client_path=None,
    load_client=False,
    force_rebuild=False,
    save_every_trials=None,
    return_trial_df: bool = False,
):
    if save_every_trials is not None:
        if not isinstance(save_every_trials, int) or save_every_trials <= 0:
            raise ValueError("save_every_trials must be a positive integer or None")

    if client_path is not None and load_client and os.path.exists(client_path) and not force_rebuild:
        if verbose:
            print(f"Loading saved Ax client from: {client_path}")
        saved_payload = file_io("load", "joblib", os.path.dirname(client_path), os.path.basename(client_path))
        if isinstance(saved_payload, dict) and "client" in saved_payload:
            loaded_client = saved_payload["client"]
            loaded_results = saved_payload.get("results", [])
            if return_trial_df:
                loaded_trial_df = results_to_dataframe(loaded_results)
                return loaded_client, loaded_results, loaded_trial_df
            return loaded_client, loaded_results

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

                if (
                    client_path is not None
                    and save_every_trials is not None
                    and len(results) % save_every_trials == 0
                ):
                    client_dir = os.path.dirname(client_path)
                    client_name = os.path.basename(client_path)
                    payload = {"client": client, "results": results}
                    file_io("save", "joblib", client_dir, client_name, data=payload)
                    if verbose:
                        print(
                            f"Checkpoint saved after {len(results)} completed trials to: {client_path}"
                        )

    if client_path is not None:
        client_dir = os.path.dirname(client_path)
        client_name = os.path.basename(client_path)
        payload = {"client": client, "results": results}
        file_io("save", "joblib", client_dir, client_name, data=payload)
        if verbose:
            print(f"Saved Ax client to: {client_path}")

    if return_trial_df:
        trial_df = results_to_dataframe(results)
        return client, results, trial_df

    return client, results

def results_to_dataframe(
    results: list[dict],
    include_params: bool = True,
    include_metrics_raw: bool = True,
    include_metrics_with_sem: bool = False,
) -> pd.DataFrame:
    """
    Convert Ax optimization results into a flat DataFrame with one row per trial.

    Parameters
    ----------
    results : list of dict
        The `results` object returned by `run_ax_optimization`.
    include_params : bool, optional
        Whether to expand the parameter dictionary into columns.
    include_metrics_raw : bool, optional
        Whether to expand raw metric means (from `metrics_raw`) into columns.
    include_metrics_with_sem : bool, optional
        Whether to also expand SEM-aware Ax metric payloads from `metrics`
        into paired columns: `<metric>_mean` and `<metric>_sem`.

    Returns
    -------
    pandas.DataFrame
        Flat table with one row per completed trial.
    """
    rows = []

    for item in results:
        row = {
            "trial_index": item.get("trial_index"),
        }

        if include_params:
            params = item.get("params", {}) or {}
            row.update(params)

        if include_metrics_raw:
            metrics_raw = item.get("metrics_raw", {}) or {}
            for key, value in metrics_raw.items():
                row[key] = value

        if include_metrics_with_sem:
            metrics = item.get("metrics", {}) or {}
            for key, value in metrics.items():
                if isinstance(value, (tuple, list)) and len(value) >= 2:
                    row[f"{key}_mean"] = value[0]
                    row[f"{key}_sem"] = value[1]
                else:
                    row[f"{key}_mean"] = value
                    row[f"{key}_sem"] = float("nan")

        rows.append(row)

    df = pd.DataFrame(rows)

    if "trial_index" in df.columns:
        df["trial_index"] = pd.to_numeric(df["trial_index"], errors="coerce")

    return df

def _as_mean_sem(value):
    """Normalize Ax metric payloads into a (mean, sem) tuple."""
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    if isinstance(value, Number):
        return float(value), float("nan")
    return float("nan"), float("nan")


def _normalize_frontier_item(item):
    """Handle different Ax return shapes for Pareto frontier entries."""
    if isinstance(item, tuple):
        params = item[0] if len(item) > 0 else {}
        metrics = item[1] if len(item) > 1 else {}
        trial_index = item[2] if len(item) > 2 else None
        return params or {}, metrics or {}, trial_index

    if isinstance(item, dict):
        params = item.get("parameterization") or item.get("parameters") or item.get("params") or {}
        metrics = item.get("metrics") or item.get("means") or {}
        trial_index = item.get("trial_index")
        return params, metrics, trial_index

    return {}, {}, None


def get_optimization_recommendations(client, optimization_mode="scalarized", top_k=5):
    """Return optimization recommendations for scalarized or multiobjective Ax runs."""
    mode = optimization_mode.strip().lower()
    if mode not in {"scalarized", "multiobjective"}:
        raise ValueError(f"optimization_mode must be 'scalarized' or 'multiobjective', got {optimization_mode}")

    with pd.option_context("future.no_silent_downcasting", False), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Downcasting behavior in `replace` is deprecated.*",
            category=FutureWarning,
            module=r"ax\.adapter\.transforms\.choice_encode",
        )

        if mode == "scalarized":
            try:
                best_result = client.get_best_parameterization(use_model_predictions=True)
            except TypeError:
                best_result = client.get_best_parameterization(use_model_predictions=False)

            # Ax versions may return either (params, metrics) or (params, metrics, trial_index).
            best_params = best_result[0]
            best_metrics = best_result[1]
            trial_index = best_result[2] if len(best_result) > 2 else None
            recommendations = {
                "mode": mode,
                "best": {
                    "params": best_params,
                    "metrics": best_metrics,
                    "trial_index": trial_index,
                },
            }

            best = recommendations["best"]
            best_params = best["params"]
            best_metrics = best["metrics"]

            print("\nBest parameterization found:")
            for k, v in best_params.items():
                print(f"  {k}: {v}")

            print("\nAssociated metrics (mean, sem):")
            for metric_name, raw_value in best_metrics.items():
                mean, sem = _as_mean_sem(raw_value)
                print(f"  {metric_name}: mean={mean:.4f}, sem={sem:.4f}")

            return recommendations

        # Multiobjective: query Pareto frontier instead of a single best point.
        try:
            frontier = client.get_pareto_frontier(use_model_predictions=True)
        except TypeError:
            frontier = client.get_pareto_frontier()

        frontier_points = []
        for item in frontier:
            params, metrics, trial_index = _normalize_frontier_item(item)
            frontier_points.append(
                {
                    "params": params,
                    "metrics": metrics,
                    "trial_index": trial_index,
                }
            )

        recommendations = {
            "mode": mode,
            "pareto_frontier": frontier_points[:top_k],
            "pareto_frontier_size": len(frontier_points),
        }

        pareto_points = recommendations["pareto_frontier"]
        print(f"\nPareto frontier size: {recommendations['pareto_frontier_size']}")
        print(f"Showing first {len(pareto_points)} Pareto points:\n")
        if not pareto_points:
            print("No Pareto points found.")
        else:
            for idx, point in enumerate(pareto_points, start=1):
                print(f"Point {idx}:")
                if point["trial_index"] is not None:
                    print(f"  trial_index: {point['trial_index']}")

                print("  params:")
                for k, v in point["params"].items():
                    print(f"    {k}: {v}")

                print("  metrics (mean, sem):")
                for metric_name, raw_value in point["metrics"].items():
                    mean, sem = _as_mean_sem(raw_value)
                    print(f"    {metric_name}: mean={mean:.4f}, sem={sem:.4f}")
                print("")

        return recommendations

def prepare_objective_history(
    trial_df: pd.DataFrame,
    objective_names,
    minimize_metrics=None,
    trial_index_column: str = "trial_index",
    aggregate_name: str = "objective_sum",
    objective_weights=None,
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

    if objective_weights is not None:
        history[aggregate_name] = sum(
            history[col] * float(objective_weights.get(col, 1.0))
            for col in objective_names
        )
    else:
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
    fig_size=(10, 6),
):
    import matplotlib.pyplot as plt
    import numpy as np

    trial_indices = history_df[trial_index_column].tolist()
    figures = {}

    for objective_name in objective_names:
        objective_values = pd.to_numeric(history_df[objective_name], errors="coerce")
        fig, ax = plt.subplots(figsize=fig_size)
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
    fig, ax = plt.subplots(figsize=fig_size)
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


def summarize_inverse_design_metrics_evolution(
    trial_df: pd.DataFrame,
    objective_names,
    display_names=None,
    minimize_metrics=None,
    trial_index_column: str = "trial_index",
    aggregate_name: str = "obj_sum",
    aggregate_display_name: str = "Sum of Objectives",
    y_positions=None,
    output_dir: str | None = None,
    fig_size=None,
    figure_size=(12, 8),
    elev: float = 15,
    azim: float = -60,
    cmap=custom_cmap,
    text_color: str = "#000000",
    line_color: str = "#7F7F7F",
    grid_color: str = "#B0B0B0",
    marker_size: float = 3,
    title: str = "Inverse Design Metrics Evolution",
    save_name: str = "inverse_design_metrics_evolution.png",
    objective_weights=None,
):
    import matplotlib.pyplot as plt
    import numpy as np
    import textwrap
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection

    history_df = prepare_objective_history(
        trial_df=trial_df,
        objective_names=objective_names,
        minimize_metrics=minimize_metrics,
        trial_index_column=trial_index_column,
        aggregate_name=aggregate_name,
        objective_weights=objective_weights,
    )

    objective_names = list(objective_names)
    if display_names is None:
        display_names = list(objective_names)
    else:
        display_names = list(display_names)

    if len(display_names) != len(objective_names):
        raise ValueError(
            "display_names must have the same length as objective_names "
            f"({len(display_names)} != {len(objective_names)})"
        )

    rename_map = dict(zip(objective_names, display_names))
    plot_df = history_df.rename(columns=rename_map).copy()
    plot_df[aggregate_display_name] = pd.to_numeric(history_df[aggregate_name], errors="coerce")

    plot_columns = display_names + [aggregate_display_name]
    num_metrics = len(plot_columns)

    # Wrap long metric labels so crowded objective sets remain readable.
    label_wrap_width = 24 if num_metrics <= 6 else (18 if num_metrics <= 10 else 14)
    wrapped_plot_columns = [
        "\n".join(
            textwrap.wrap(
                str(column_name),
                width=label_wrap_width,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
        or str(column_name)
        for column_name in plot_columns
    ]
    max_label_lines = max(label.count("\n") + 1 for label in wrapped_plot_columns)

    if fig_size is not None:
        figure_size = fig_size

    default_figure_size = (12, 8)
    if tuple(figure_size) == default_figure_size:
        dynamic_width = min(18.0, max(12.0, 10.5 + 0.4 * (max_label_lines - 1)))
        dynamic_height = min(24.0, max(8.5, 4.8 + 0.95 * num_metrics + 0.35 * (max_label_lines - 1)))
        figure_size = (dynamic_width, dynamic_height)

    if y_positions is None:
        base_spacing = max(1.1, 1.0 + 0.15 * max_label_lines + 0.03 * max(0, num_metrics - 4))
        y_positions = np.arange(num_metrics, dtype=float) * base_spacing
    else:
        y_positions = np.asarray(y_positions, dtype=float)
        if len(y_positions) != len(plot_columns):
            raise ValueError(
                "y_positions must have the same length as the plotted metrics "
                f"({len(y_positions)} != {len(plot_columns)})"
            )

    trial_indices = pd.to_numeric(history_df[trial_index_column], errors="coerce").to_numpy(dtype=float)
    if np.isnan(trial_indices).any():
        trial_indices = np.arange(len(history_df), dtype=float)

    if cmap is None:
        cmap = custom_cmap
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    color_positions = np.linspace(0.0, 1.0, len(plot_columns), endpoint=False)
    colors = [cmap(position) for position in color_positions]

    def darken_color(color, factor=0.6):
        return tuple(np.clip(np.array(to_rgb(color)) * factor, 0, 1))

    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False
    left_margin = min(0.30, 0.10 + 0.02 * max(0, max_label_lines - 1) + 0.008 * max(0, num_metrics - 5))
    right_margin = 0.96 if num_metrics >= 8 else 0.94
    top_margin = 0.90 if num_metrics >= 10 else 0.94
    fig.subplots_adjust(left=left_margin, right=right_margin, bottom=0.08, top=top_margin)

    positive_layers = []
    for index, column_name in enumerate(plot_columns):
        values = pd.to_numeric(plot_df[column_name], errors="coerce").to_numpy(dtype=float)
        ys = np.full_like(trial_indices, y_positions[index], dtype=float)
        color = colors[index]
        outline_color = darken_color(color)

        values_neg = np.minimum(values, 0.0)
        values_pos = np.maximum(values, 0.0)
        has_neg = np.any(values_neg < 0)
        has_pos = np.any(values_pos > 0)

        if has_neg:
            verts_neg = [[trial_indices[0], ys[0], 0.0]]
            for point_index in range(len(trial_indices)):
                verts_neg.append([trial_indices[point_index], ys[point_index], values_neg[point_index]])
            verts_neg.append([trial_indices[-1], ys[-1], 0.0])

            poly_neg = Poly3DCollection(
                [verts_neg],
                alpha=0.3,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
            )
            poly_neg.set_zorder(1)
            ax.add_collection3d(poly_neg)

            label_neg = column_name if not has_pos else None
            ax.plot(
                trial_indices,
                ys,
                values_neg,
                color=color,
                linewidth=1,
                marker="o",
                markersize=marker_size,
                markeredgecolor=outline_color,
                markeredgewidth=0.8,
                alpha=1.0,
                label=label_neg,
                zorder=1,
            )

        positive_layers.append((column_name, trial_indices.copy(), ys.copy(), values_pos.copy(), color, outline_color, has_pos))

    x_plane = np.linspace(np.min(trial_indices), np.max(trial_indices), 2)
    y_plane = np.linspace(np.min(y_positions), np.max(y_positions), 2)
    X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
    Z_plane = np.zeros_like(X_plane)
    ax.plot_surface(
        X_plane,
        Y_plane,
        Z_plane,
        color="#9E9E9E",
        alpha=0.5,
        edgecolor="none",
        linewidth=0,
        antialiased=False,
        shade=True,
        zorder=2,
    )

    count = len(positive_layers)
    for column_name, xs_pos, ys_pos, values_pos, color, outline_color, has_pos in positive_layers:
        if not has_pos:
            continue

        verts_pos = [[xs_pos[0], ys_pos[0], 0.0]]
        for point_index in range(len(xs_pos)):
            verts_pos.append([xs_pos[point_index], ys_pos[point_index], values_pos[point_index]])
        verts_pos.append([xs_pos[-1], ys_pos[-1], 0.0])

        poly_pos = Poly3DCollection(
            [verts_pos],
            alpha=0.5,
            facecolor=color,
            edgecolor="black",
            linewidth=0.5,
        )
        poly_pos.set_zorder(3)
        ax.add_collection3d(poly_pos)
        count += 1
        ax.plot(
            xs_pos,
            ys_pos,
            values_pos,
            color=color,
            linewidth=1,
            marker="o",
            markersize=marker_size,
            markeredgecolor=outline_color,
            markeredgewidth=0.8,
            alpha=1.0,
            label=column_name,
            zorder=3 + count,
        )

    z_values = pd.to_numeric(plot_df[plot_columns].stack(), errors="coerce").to_numpy(dtype=float)
    z_values = z_values[np.isfinite(z_values)]
    if z_values.size > 0:
        z_min = float(np.min(z_values))
        z_max = float(np.max(z_values))
        z_span = max(z_max - z_min, 1e-6)
        z_pad = 0.08 * z_span
        ax.set_zlim(z_min - z_pad, z_max + z_pad)

    z_label_pad = min(35, 12 + 2 * max_label_lines)

    # Keep axis title slots empty and render labels as in-box 3D annotations instead.
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("Objective Value", fontsize=14, labelpad=z_label_pad, color=text_color, fontfamily="Arial", rotation=90)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(wrapped_plot_columns)
    ax.tick_params(axis="y", pad=2)
    ax.view_init(elev=elev, azim=azim)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor((0, 0, 0, 0))
    ax.yaxis.pane.set_edgecolor((0, 0, 0, 0))
    ax.zaxis.pane.set_edgecolor((0, 0, 0, 0))
    ax.xaxis.pane.set_linewidth(0)
    ax.yaxis.pane.set_linewidth(0)
    ax.zaxis.pane.set_linewidth(0)
    ax.xaxis.line.set_color((0, 0, 0, 0))
    ax.yaxis.line.set_color((0, 0, 0, 0))
    ax.zaxis.line.set_color((0, 0, 0, 0))
    ax.xaxis.line.set_linewidth(1)
    ax.yaxis.line.set_linewidth(1)
    ax.zaxis.line.set_linewidth(1)

    ax.tick_params(axis="x", colors=text_color, width=5, pad=0)
    ax.tick_params(axis="y", colors=text_color, width=5, pad=2)
    ax.tick_params(axis="z", colors=text_color, width=5, pad=6)

    for label in ax.get_yticklabels():
        label.set_horizontalalignment("left")
        label.set_verticalalignment("center")
        label.set_rotation(0)

    ax.xaxis._axinfo["grid"]["color"] = grid_color
    ax.yaxis._axinfo["grid"]["color"] = grid_color
    ax.zaxis._axinfo["grid"]["color"] = grid_color
    ax.xaxis._axinfo["grid"]["linewidth"] = 1
    ax.yaxis._axinfo["grid"]["linewidth"] = 1
    ax.zaxis._axinfo["grid"]["linewidth"] = 1
    ax.grid(False)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    x_span = max(xlim[1] - xlim[0], 1e-9)
    y_span = max(ylim[1] - ylim[0], 1e-9)
    z_span = max(zlim[1] - zlim[0], 1e-9)

    # Place x/y axis titles inside the 3D box and keep them parallel to their axes.
    ax.text(
        xlim[0] + 0.50 * x_span,
        ylim[0] + 0.01 * y_span,
        zlim[0] + 0.01 * z_span,
        "Trial Index",
        color=text_color,
        fontsize=12,
        fontfamily="Arial",
        ha="center",
        va="bottom",
        rotation=0,
        zdir="z",
        zorder=1200,
    )
    ax.text(
        xlim[1] - 0.01 * x_span,
        ylim[0] + 0.5 * y_span,
        zlim[0] + 0.01 * z_span,
        "Objective Functions",
        color=text_color,
        fontsize=12,
        fontfamily="Arial",
        ha="right",
        va="center",
        rotation=90,
        zdir="z",
        zorder=1200,
    )

    front_edges = [
        ([xlim[0], xlim[1]], [ylim[0], ylim[0]], [zlim[1], zlim[1]]),
        ([xlim[1], xlim[1]], [ylim[0], ylim[0]], [zlim[0], zlim[1]]),
        ([xlim[1], xlim[1]], [ylim[0], ylim[1]], [zlim[1], zlim[1]]),
    ]

    back_edges = [
        ([xlim[0], xlim[1]], [ylim[1], ylim[1]], [zlim[0], zlim[0]]),
        ([xlim[0], xlim[1]], [ylim[1], ylim[1]], [zlim[1], zlim[1]]),
        ([xlim[0], xlim[0]], [ylim[0], ylim[1]], [zlim[0], zlim[0]]),
        ([xlim[0], xlim[0]], [ylim[0], ylim[1]], [zlim[1], zlim[1]]),
        ([xlim[0], xlim[0]], [ylim[1], ylim[1]], [zlim[0], zlim[1]]),
        ([xlim[1], xlim[1]], [ylim[1], ylim[1]], [zlim[0], zlim[1]]),
        ([xlim[0], xlim[0]], [ylim[0], ylim[0]], [zlim[0], zlim[1]]),
        ([xlim[1], xlim[1]], [ylim[0], ylim[1]], [zlim[0], zlim[0]]),
        ([xlim[0], xlim[1]], [ylim[0], ylim[0]], [zlim[0], zlim[0]]),
    ]

    for edge in back_edges:
        line = Line3D(edge[0], edge[1], edge[2], color=line_color, linewidth=1.2, alpha=1.0, zorder=0)
        line.set_clip_on(False)
        ax.add_line(line)

    for edge in front_edges:
        line = Line3D(edge[0], edge[1], edge[2], color=line_color, linewidth=1.2, alpha=0.5, zorder=1000)
        line.set_clip_on(False)
        ax.add_line(line)

    title_pad = 18 + 2 * max(0, max_label_lines - 1)
    ax.set_title(title, pad=title_pad)

    legend_kwargs = {}
    if num_metrics >= 8:
        legend_kwargs = {"bbox_to_anchor": (1.02, 1.0), "borderaxespad": 0.0}
    ax.legend(loc="upper left", fontsize=14, framealpha=0.9, prop={"family": "Arial"}, **legend_kwargs)

    if output_dir is not None:
        fig.savefig(os.path.join(output_dir, save_name), dpi=300, bbox_inches="tight")

    return {
        "history_df": history_df,
        "plot_df": plot_df,
        "figure": fig,
        "axes": ax,
    }


def build_frontier_dataframe(client, use_model_predictions: bool = False) -> pd.DataFrame:
    try:
        frontier = client.get_pareto_frontier(use_model_predictions=use_model_predictions)
    except TypeError:
        frontier = client.get_pareto_frontier()

    rows = []
    for item in frontier:
        if not isinstance(item, (tuple, dict)):
            continue

        params, metrics, _ = _normalize_frontier_item(item)

        flattened_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (tuple, list)) and len(value) >= 1:
                flattened_metrics[key] = value[0]
            else:
                flattened_metrics[key] = value
        rows.append({**params, **flattened_metrics})

    return pd.DataFrame(rows)

def _prepare_objective_frame(
    df: pd.DataFrame,
    objective_names,
    minimize_metrics=None,
) -> pd.DataFrame:
    """
    Return a copy of the objective columns where all objectives are oriented
    so that larger values are better.
    """
    objective_names = list(objective_names)
    minimize_metrics = set(minimize_metrics or [])

    missing = [col for col in objective_names if col not in df.columns]
    if missing:
        raise KeyError(f"Missing objective columns: {missing}")

    out = df.copy()
    for col in objective_names:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if col in minimize_metrics:
            out[col] = -out[col]
    return out


def _pareto_efficient_mask(
    values: pd.DataFrame | list | tuple,
) -> pd.Series:
    """
    Compute Pareto-efficient mask for a 2D numeric table where larger is better
    for every objective.
    """
    import numpy as np

    if isinstance(values, pd.DataFrame):
        arr = values.to_numpy(dtype=float)
        index = values.index
    else:
        arr = np.asarray(values, dtype=float)
        index = pd.RangeIndex(len(arr))

    n = arr.shape[0]
    is_efficient = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_efficient[i]:
            continue

        candidate = arr[i]
        active = np.where(is_efficient)[0]
        active_points = arr[active]

        dominates_candidate = np.all(active_points >= candidate, axis=1) & np.any(active_points > candidate, axis=1)
        if np.any(dominates_candidate):
            is_efficient[i] = False
            continue

        candidate_dominates = np.all(candidate >= active_points, axis=1) & np.any(candidate > active_points, axis=1)
        dominated_indices = active[candidate_dominates]
        is_efficient[dominated_indices] = False
        is_efficient[i] = True

    return pd.Series(is_efficient, index=index)


def plot_pareto_scatter_2d(
    trial_df: pd.DataFrame,
    objective_names,
    minimize_metrics=None,
    display_names=None,
    frontier_df: pd.DataFrame | None = None,
    color_by: str | None = "trial_index",
    cmap: str = "viridis",
    trial_color: str = "#BDBDBD",
    frontier_color: str = "#D62728",
    marker_size: int = 45,
    frontier_marker_size: int = 90,
    alpha: float = 0.8,
    annotate_frontier: bool = False,
    output_path: str | None = None,
    fig_size=(8, 6),
):
    """
    Plot a true 2-objective Pareto scatter from trial_df. If frontier_df is not
    provided, the Pareto frontier is computed directly from trial_df.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    objective_names = list(objective_names)
    if len(objective_names) != 2:
        raise ValueError("plot_pareto_scatter_2d requires exactly 2 objective names")

    x_col, y_col = objective_names
    display_names = list(display_names) if display_names is not None else [x_col, y_col]

    oriented_df = _prepare_objective_frame(
        trial_df,
        objective_names=objective_names,
        minimize_metrics=minimize_metrics,
    ).copy()

    plot_df = oriented_df.dropna(subset=[x_col, y_col]).copy()

    if frontier_df is not None and not frontier_df.empty:
        frontier_oriented = _prepare_objective_frame(
            frontier_df,
            objective_names=objective_names,
            minimize_metrics=minimize_metrics,
        ).dropna(subset=[x_col, y_col]).copy()
        frontier_plot_df = frontier_oriented
    else:
        mask = _pareto_efficient_mask(plot_df[[x_col, y_col]])
        frontier_plot_df = plot_df.loc[mask].copy()

    frontier_plot_df = frontier_plot_df.sort_values(by=[x_col, y_col])

    fig, ax = plt.subplots(figsize=fig_size)

    if color_by is not None and color_by in plot_df.columns:
        scatter = ax.scatter(
            plot_df[x_col],
            plot_df[y_col],
            c=pd.to_numeric(plot_df[color_by], errors="coerce"),
            cmap=cmap,
            s=marker_size,
            alpha=alpha,
            edgecolors="white",
            linewidths=0.6,
            label="Trials",
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(color_by)
    else:
        ax.scatter(
            plot_df[x_col],
            plot_df[y_col],
            c=trial_color,
            s=marker_size,
            alpha=alpha,
            edgecolors="white",
            linewidths=0.6,
            label="Trials",
        )

    ax.plot(
        frontier_plot_df[x_col],
        frontier_plot_df[y_col],
        color=frontier_color,
        linewidth=1.5,
        alpha=0.9,
        zorder=4,
    )

    ax.scatter(
        frontier_plot_df[x_col],
        frontier_plot_df[y_col],
        c=frontier_color,
        s=frontier_marker_size,
        marker="^",
        edgecolors=[_darker_color(frontier_color)],
        linewidths=0.9,
        label="Pareto Frontier",
        zorder=5,
    )

    if annotate_frontier and "trial_index" in frontier_plot_df.columns:
        for _, row in frontier_plot_df.iterrows():
            ax.annotate(
                str(int(row["trial_index"])) if pd.notna(row["trial_index"]) else "",
                (row[x_col], row[y_col]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

    ax.set_xlabel(display_names[0])
    ax.set_ylabel(display_names[1])
    ax.set_title("Pareto Scatter Plot")
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.25)

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return {
        "figure": fig,
        "axes": ax,
        "trial_df_oriented": plot_df,
        "frontier_df_oriented": frontier_plot_df,
    }


def plot_parallel_coordinates_plotly(
    trial_df: pd.DataFrame,
    objective_names,
    minimize_metrics=None,
    color_by: str | None = None,
    highlight_pareto: bool = True,
    frontier_df: pd.DataFrame | None = None,
    title: str = "Parallel Coordinates of Objective Values",
    normalize: bool = True,
):
    """
    Create an interactive Plotly parallel-coordinates plot for >3 objectives.
    All objectives are oriented so that larger values are better.
    """
    objective_names = list(objective_names)
    if len(objective_names) < 4:
        raise ValueError("plot_parallel_coordinates_plotly is intended for 4 or more objectives")

    try:
        import plotly.express as px
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for plot_parallel_coordinates_plotly. "
            "Install plotly first."
        ) from exc

    plot_df = _prepare_objective_frame(
        trial_df,
        objective_names=objective_names,
        minimize_metrics=minimize_metrics,
    ).copy()

    plot_df = plot_df.dropna(subset=objective_names).copy()

    if normalize:
        for col in objective_names:
            col_min = plot_df[col].min()
            col_max = plot_df[col].max()
            if pd.notna(col_min) and pd.notna(col_max) and col_max > col_min:
                plot_df[col] = (plot_df[col] - col_min) / (col_max - col_min)
            else:
                plot_df[col] = 0.0

    if highlight_pareto:
        if frontier_df is not None and not frontier_df.empty:
            frontier_oriented = _prepare_objective_frame(
                frontier_df,
                objective_names=objective_names,
                minimize_metrics=minimize_metrics,
            ).dropna(subset=objective_names).copy()

            if normalize:
                for col in objective_names:
                    ref_min = plot_df[col].min()
                    ref_max = plot_df[col].max()
                    original = frontier_oriented[col]
                    if pd.notna(ref_min) and pd.notna(ref_max) and ref_max > ref_min:
                        frontier_oriented[col] = (original - original.min()) / (original.max() - original.min()) if original.max() > original.min() else 0.0

            frontier_keys = set(map(tuple, frontier_oriented[objective_names].round(10).to_numpy()))
            plot_df["is_pareto"] = [
                tuple(row.round(10)) in frontier_keys
                for row in plot_df[objective_names].to_numpy()
            ]
        else:
            mask = _pareto_efficient_mask(plot_df[objective_names])
            plot_df["is_pareto"] = mask.to_numpy()
    else:
        plot_df["is_pareto"] = False

    if color_by is not None and color_by in plot_df.columns:
        color_series = pd.to_numeric(plot_df[color_by], errors="coerce")
    else:
        color_series = plot_df["is_pareto"].astype(int)

    fig = px.parallel_coordinates(
        plot_df,
        dimensions=objective_names,
        color=color_series,
        color_continuous_scale="Viridis",
        title=title,
    )

    return {
        "figure": fig,
        "plot_df": plot_df,
    }


def plot_objective_response_maps(
    trial_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    frontier_df: pd.DataFrame | None = None,
    grid_size: int = 100,
    smoothing: float = 0.0001,
    kernel: str = "thin_plate_spline",
    y_padding: float = 0.0,
    surface_cmap: str = "viridis",
    contour_cmap: str = "viridis",
    trial_color: str = "r",
    frontier_color: str = "b",
    surface_marker_size: int = 100,
    contour_marker_size: int = 20,
    elev: float = 30,
    azim: float = -45,
    fig_size=(10, 8),
):
    """
    Reuse the existing RBF/surface/contour pipeline to visualize one objective
    over two selected design variables, optionally overlaying Pareto points.
    """
    surface_data = build_rbf_surface(
        data=trial_df,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        grid_size=grid_size,
        smoothing=smoothing,
        kernel=kernel,
        y_padding=y_padding,
    )

    fig_surface, ax_surface = plot_inverse_design_surface(
        surface_data=surface_data,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        frontier_df=frontier_df,
        cmap=surface_cmap,
        trial_color=trial_color,
        frontier_color=frontier_color,
        marker_size=surface_marker_size,
        elev=elev,
        azim=azim,
        y_padding=y_padding,
        fig_size=fig_size,
    )

    fig_contour, ax_contour = plot_inverse_design_contour(
        surface_data=surface_data,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        frontier_df=frontier_df,
        cmap=contour_cmap,
        trial_color=trial_color,
        frontier_color=frontier_color,
        marker_size=contour_marker_size,
        y_padding=y_padding,
        fig_size=fig_size,
    )

    return {
        "surface_data": surface_data,
        "surface_figure": fig_surface,
        "surface_axes": ax_surface,
        "contour_figure": fig_contour,
        "contour_axes": ax_contour,
    }

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
    fig_size=(10, 8),
):
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=fig_size)
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
    fig_size=(10, 8),
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=fig_size)
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
        frontier_clean = frontier_df.dropna(subset=[x_col, y_col]).sort_values(by=[x_col, y_col])
        if len(frontier_clean) >= 2:
            ax.plot(
                frontier_clean[x_col],
                frontier_clean[y_col],
                color=frontier_color,
                linestyle="-",
                linewidth=1.8,
                alpha=0.9,
                zorder=3,
            )
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

def plot_ax_results(
    trial_df: pd.DataFrame,
    objective_names,
    optimization_mode: str = "scalarized",
    minimize_metrics=None,
    trial_index_column: str = "trial_index",
    aggregate_name: str = "objective_sum",
    aggregate_display_name: str = "Sum of Objectives",
    display_names=None,
    client=None,
    use_model_predictions: bool = False,
    output_dir: str | None = None,
    xtick_step: int = 10,
    fig_size=None,
    figure_size=(12, 8),
    title: str = "Inverse Design Metrics Evolution",
    save_name: str = "inverse_design_metrics_evolution.png",
    make_parallel_coordinates: bool = False,
    parallel_coords_color_by: str | None = None,
    response_map_specs: list[dict] | None = None,
    objective_weights=None,
):
    """
    Dispatch visualization of Ax optimization results based on objective count.

    Parameters
    ----------
    response_map_specs : list of dict, optional
        Each dict should contain at least:
        {"x_col": ..., "y_col": ..., "z_col": ...}
        and may also include optional kwargs for plot_objective_response_maps().
    """
    requested_objective_names = list(objective_names)
    if fig_size is not None:
        figure_size = fig_size
    if aggregate_name in requested_objective_names:
        objective_names = [name for name in requested_objective_names if name != aggregate_name]
        if display_names is not None and len(display_names) == len(requested_objective_names):
            display_names = [
                name
                for index, name in enumerate(display_names)
                if requested_objective_names[index] != aggregate_name
            ]
    else:
        objective_names = requested_objective_names

    mode = _normalize_optimization_mode(optimization_mode)
    n_objectives = len(objective_names)

    if n_objectives == 0:
        raise ValueError("objective_names must contain at least one objective column")

    missing = [col for col in objective_names if col not in trial_df.columns]
    if missing:
        raise KeyError(f"Missing objective columns in trial_df: {missing}")

    history_df = prepare_objective_history(
        trial_df=trial_df,
        objective_names=objective_names,
        minimize_metrics=minimize_metrics,
        trial_index_column=trial_index_column,
        aggregate_name=aggregate_name,
        objective_weights=objective_weights,
    )

    plot_trial_df = trial_df.copy()
    if aggregate_name not in plot_trial_df.columns:
        plot_trial_df[aggregate_name] = history_df[aggregate_name].values

    output = {
        "mode": mode,
        "objective_names": objective_names,
        "trial_df": plot_trial_df.copy(),
        "history_df": history_df,
        "figures": {},
        "summary": None,
        "frontier_df": None,
        "pareto_scatter": None,
        "parallel_coordinates": None,
        "response_maps": [],
    }

    frontier_df = None
    if mode == "multiobjective" and client is not None:
        frontier_df = build_frontier_dataframe(
            client,
            use_model_predictions=use_model_predictions,
        )
        output["frontier_df"] = frontier_df

    if n_objectives == 1:
        figures = plot_objective_evolution(
            history_df=history_df,
            objective_names=objective_names,
            trial_index_column=trial_index_column,
            aggregate_name=aggregate_name,
            xtick_step=xtick_step,
            output_dir=output_dir,
            aggregate_title=f"Evolution of {aggregate_display_name}",
            fig_size=figure_size,
        )
        output["figures"].update(figures)

    elif n_objectives >= 2:
        summary = summarize_inverse_design_metrics_evolution(
            trial_df=trial_df,
            objective_names=objective_names,
            display_names=display_names,
            minimize_metrics=minimize_metrics,
            trial_index_column=trial_index_column,
            aggregate_name=aggregate_name,
            aggregate_display_name=aggregate_display_name,
            output_dir=output_dir,
            fig_size=figure_size,
            figure_size=figure_size,
            title=title,
            save_name=save_name,
            objective_weights=objective_weights,
        )
        output["summary"] = summary
        output["figures"]["summary_figure"] = summary["figure"]

        if make_parallel_coordinates and n_objectives > 3:
            parallel_coordinates = plot_parallel_coordinates_plotly(
                trial_df=plot_trial_df,
                objective_names=objective_names,
                minimize_metrics=minimize_metrics,
                color_by=parallel_coords_color_by,
                highlight_pareto=True,
                frontier_df=frontier_df,
                title="Parallel Coordinates of Objective Values",
            )
            output["parallel_coordinates"] = parallel_coordinates

    if response_map_specs:
        for spec in response_map_specs:
            spec = dict(spec)
            x_col = spec.pop("x_col")
            y_col = spec.pop("y_col")
            z_col = spec.pop("z_col")

            response_map = plot_objective_response_maps(
                trial_df=plot_trial_df,
                x_col=x_col,
                y_col=y_col,
                z_col=z_col,
                frontier_df=frontier_df,
                **spec,
            )
            output["response_maps"].append(response_map)

    return output