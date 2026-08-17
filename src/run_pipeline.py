"""One-command workflow entry point for the ML-MEM package.

Use from the command line
-------------------------
    python -m run_pipeline                          # whole workflow, config.yaml in cwd
    python -m run_pipeline --config path/to/config.yaml
    python -m run_pipeline --stage model --force-retrain

Use as a library in your own project
------------------------------------
    from run_pipeline import run_pipeline

    run_pipeline(
        config_path="./my_project/config.yaml",
        project_root="./my_project",
        stages="all",
    )

Everything (dataset, targets, features, models, optimization) is read from
the config file, so the same package can be reused on any tabular ML
dataset without code changes.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402

from utl_config_loader import (  # noqa: E402
    get_data_paths,
    get_figures_dir,
    get_model_dir,
    get_optimization_config,
    load_config,
    print_dataset_report,
    resolve_features_to_model,
    resolve_targets,
)
from utl_eda import compute_and_plot_pearson_heatmap, plot_target_distributions  # noqa: E402
from utl_feateng import feature_engineering  # noqa: E402
from utl_fileIO import file_io  # noqa: E402
from utl_modelling import evaluate_and_plot_results, train_model  # noqa: E402

STAGES = ["data", "eda", "model", "shap", "optimize"]


def _safe_target(target: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(target)).strip("_")


def load_data(config):
    """Load the dataset, print a compatibility report, and enrich it."""
    data_paths = get_data_paths(config)
    dataset_path = data_paths["dataset"]
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. Put your file in data/raw/ and "
            "set data.dataset in your config file."
        )

    sheet = config.get("data", {}).get("sheet") or None
    if str(dataset_path).lower().endswith(".csv"):
        df = pd.read_csv(dataset_path)
    else:
        df = pd.read_excel(dataset_path, sheet_name=sheet)

    print(f"\nLoaded dataset: {os.path.basename(dataset_path)}")
    print_dataset_report(config, df)

    from utl_config_loader import check_dataset_compatibility

    report = check_dataset_compatibility(config, df)
    if report["targets_missing"]:
        missing = ", ".join(report["targets_missing"])
        raise ValueError(
            f"Configured targets not found in the dataset: {missing}. "
            "Fix data.targets in your config file."
        )

    df_enriched = feature_engineering(df, verbose=False, config=config)
    return df, df_enriched


def run_eda(config, df_enriched, targets, figures_dir):
    """Summary target plots + Pearson correlation heatmap."""
    fig, _ = plot_target_distributions(
        df_enriched, target_columns=targets, config=config
    )
    file_io("save", "figure", figures_dir, "target_distributions.png", data=fig)

    numeric_df = df_enriched.select_dtypes(include=["number"])
    if numeric_df.shape[1] >= 2:
        _, heatmap_fig = compute_and_plot_pearson_heatmap(numeric_df, config=config)
        file_io("save", "figure", figures_dir, "pearson_heatmap.png", data=heatmap_fig)
        plt.close(heatmap_fig)
    else:
        print("Skipping Pearson heatmap: fewer than two numeric columns.")

    plt.close(fig)
    print("EDA figures saved to:", figures_dir)


def run_modeling(config, df_enriched, targets, figures_dir, force_retrain):
    """Train one model per target and save evaluation plots."""
    for target in targets:
        print(f"\n=== Modeling target: {target} ===")
        pipe, X_train, X_test, y_train, y_test = train_model(
            df=df_enriched,
            target=target,
            config_path=config["_config_path"],
            force_retrain=force_retrain,
        )

        y_pred_train = pipe.predict(X_train)
        y_pred_test = pipe.predict(X_test)

        fig, resolved_task = evaluate_and_plot_results(
            target=target,
            y_train=y_train,
            y_pred_train=y_pred_train,
            y_test=y_test,
            y_pred_test=y_pred_test,
            task="auto",
            config=config,
        )
        plot_name = (
            f"confusion_{_safe_target(target)}.png"
            if resolved_task == "classification"
            else f"parity_{_safe_target(target)}.png"
        )
        file_io("save", "figure", figures_dir, plot_name, data=fig)
        plt.close(fig)
    print(f"Modeling completed for {len(targets)} targets")


def run_shap(config, df_enriched, targets, figures_dir):
    """SHAP interpretation for each trained model."""
    import utl_shap

    for target in targets:
        shap_values, feature_names, summary_fig, bar_fig = utl_shap.shap_explain_target(
            target=target,
            df_enriched=df_enriched,
            features_to_model=resolve_features_to_model(config, df_enriched),
            config=config,
            max_samples=1000,
        )
        if summary_fig is None or bar_fig is None:
            print(f"SHAP skipped for target: {target}")
            continue
        file_io(
            "save",
            "figure",
            figures_dir,
            f"shap_summary_{_safe_target(target)}.png",
            data=summary_fig,
        )
        file_io(
            "save",
            "figure",
            figures_dir,
            f"shap_importance_{_safe_target(target)}.png",
            data=bar_fig,
        )
        plt.close(summary_fig)
        plt.close(bar_fig)
    print(f"SHAP analysis completed for {len(targets)} targets")


def _optimization_is_configured(config) -> bool:
    """True only when the optimization block can actually run."""
    opt_cfg = get_optimization_config(config)
    if not opt_cfg.get("objectives") or not opt_cfg.get("search_space"):
        return False
    try:
        configured_targets = set(str(t) for t in resolve_targets(config))
    except ValueError:
        return False
    for objective_cfg in opt_cfg.get("objectives", []) or []:
        metric_key = objective_cfg.get("original_name") or objective_cfg.get("name")
        if metric_key not in configured_targets:
            return False
    return True


def run_optimization(config, figures_dir, force_retrain, trial_error=0.01):
    """Bayesian optimization / inverse design (Ax). Requires the ax package."""
    try:
        from ax.api.client import Client  # noqa: F401
        from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig

        from utl_Ax import (
            build_ax_search_space,
            get_optimization_recommendations,
            oracle_from_params,
            plot_ax_results,
            results_to_dataframe,
            run_ax_optimization,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Skipping inverse design: Ax is not available ({exc}).")
        return

    if not _optimization_is_configured(config):
        print(
            "Skipping inverse design: add optimization.objectives and "
            "optimization.search_space to your config (objectives must name "
            "columns in data.targets) to enable it."
        )
        return

    opt_cfg = get_optimization_config(config)
    search_space_parameters = build_ax_search_space(
        opt_cfg, ChoiceParameterConfig, RangeParameterConfig
    )
    client_path = os.path.join(
        get_model_dir(config), opt_cfg.get("client_cache_file", "ax_client.pkl")
    )

    print(f"\n=== Inverse design ({opt_cfg.get('mode')}) ===")
    try:
        client, results = run_ax_optimization(
            search_space_parameters=search_space_parameters,
            oracle_from_params=oracle_from_params,
            config=config,
            sobol_init=opt_cfg.get("sobol_init", 5),
            n_trials=opt_cfg.get("n_trials", 20),
            trial_error=trial_error,
            client_path=client_path,
            load_client=True,
            force_rebuild=force_retrain,
            random_seed=config.get("modeling", {}).get("random_state"),
            save_every_trials=opt_cfg.get("save_checkpoint_every", 5),
        )
    except FileNotFoundError as exc:
        print(f"Skipping inverse design: {exc}")
        return
    except KeyError as exc:
        print(f"Skipping inverse design: {exc}")
        return

    get_optimization_recommendations(
        client=client,
        optimization_mode=opt_cfg.get("mode", "multiobjective"),
        top_k=5,
    )

    trial_df = results_to_dataframe(results)
    viz = plot_ax_results(trial_df=trial_df, config=config, client=client)

    summary_figure = (
        viz.get("summary", {}).get("figure") if isinstance(viz.get("summary"), dict) else None
    )
    if summary_figure is None and viz.get("figures"):
        summary_figure = viz["figures"].get("summary_figure")
    if summary_figure is not None:
        file_io("save", "figure", figures_dir, "inverse_design_summary.png", data=summary_figure)

    for index, response_map in enumerate(viz.get("response_maps", []), start=1):
        if response_map.get("surface_figure") is not None:
            file_io(
                "save",
                "figure",
                figures_dir,
                f"response_map_{index}_surface.png",
                data=response_map["surface_figure"],
            )
        if response_map.get("contour_figure") is not None:
            file_io(
                "save",
                "figure",
                figures_dir,
                f"response_map_{index}_contour.png",
                data=response_map["contour_figure"],
            )
    print(f"Inverse design completed: {len(results)} trials")


def run_pipeline(
    config_path=None,
    project_root=None,
    stages="all",
    force_retrain=False,
    trial_error=0.01,
):
    """Run the whole ML-MEM workflow for a user-supplied configuration.

    Parameters
    ----------
    config_path : str, optional
        Path to the config file. If None, uses ``<project_root>/config.yaml``.
    project_root : str, optional
        Base directory for all relative paths in the config. If None, it is
        inferred from ``config_path`` (or the current directory).
    stages : str or list[str], optional
        Which stages to run: subset of {"data", "eda", "model", "shap",
        "optimize"}, or "all" (default).
    force_retrain : bool, optional
        Ignore saved models and retrain / restart inverse design.
    trial_error : float, optional
        Bit-flip noise added to oracle predictions during optimization.

    Returns
    -------
    dict
        Loaded config used for the run.
    """
    if config_path is not None:
        config_path = str(Path(config_path).expanduser().resolve())
    if project_root is None and config_path is not None:
        project_root = str(Path(config_path).parent)

    config = load_config(config_path=config_path, project_root=project_root)
    targets = resolve_targets(config)
    figures_dir = get_figures_dir(config)
    os.makedirs(figures_dir, exist_ok=True)

    _, df_enriched = load_data(config)

    if isinstance(stages, str):
        stage_list = STAGES if stages == "all" else [stages]
    else:
        stage_list = STAGES if stages == "all" else list(stages)

    if "eda" in stage_list:
        run_eda(config, df_enriched, targets, figures_dir)
    if "model" in stage_list:
        run_modeling(config, df_enriched, targets, figures_dir, force_retrain)
    if "shap" in stage_list:
        run_shap(config, df_enriched, targets, figures_dir)
    if "optimize" in stage_list:
        run_optimization(config, figures_dir, force_retrain, trial_error)

    print("\nPipeline finished. Figures are in:", figures_dir)
    return config


def main():
    parser = argparse.ArgumentParser(
        prog="run_pipeline", description="ML-MEM membrane ML workflow"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config file (default: <project_root>/config.yaml)",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Base directory for relative paths in the config",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all"] + STAGES,
        help="Which stage(s) to run (default: all)",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain models (and restart inverse design) even if saved results exist",
    )
    args = parser.parse_args()

    warnings.simplefilter("default")
    run_pipeline(
        config_path=args.config,
        project_root=args.project_root,
        stages=args.stage,
        force_retrain=args.force_retrain,
    )


if __name__ == "__main__":
    main()