import matplotlib
import numpy as np
import pandas as pd
import pytest

from utl_Ax import (
    _apply_ax_computed_parameters,
    _build_multiobjective_thresholds,
    _normalize,
    _normalize_optimization_mode,
    plot_ax_results,
    prepare_objective_history,
)


def test_normalize_value_within_range():
    assert _normalize(5.0, {"k": (0.0, 10.0)}, "k") == pytest.approx(0.5)


def test_normalize_optimization_mode():
    assert _normalize_optimization_mode("multiobjective") == "multiobjective"
    assert _normalize_optimization_mode("scalarized") == "scalarized"
    with pytest.raises(ValueError):
        _normalize_optimization_mode("unknown_mode")


def test_build_multiobjective_thresholds():
    cfg = {
        "objectives": [
            {"name": "o1", "direction": "maximize", "threshold": 0.5},
            {"name": "o2", "direction": "maximize", "threshold": ">=0.75"},
        ]
    }
    assert _build_multiobjective_thresholds(cfg) == ["o1>=0.5", "o2>=0.75"]


def test_apply_ax_computed_parameters_clamps_min_value():
    cfg = {
        "computed_parameters": [
            {"name": "b", "equation": "a * 2", "min_value": 4.0}
        ]
    }
    result = _apply_ax_computed_parameters({"a": 1.0}, cfg)
    assert result["a"] == 1.0
    assert result["b"] == pytest.approx(4.0)


def test_prepare_objective_history_sum():
    trial_df = pd.DataFrame(
        {"trial_index": [0, 1, 2], "o1": [0.2, 0.5, 1.0], "o2": [0.8, 0.4, 0.1]}
    )
    history = prepare_objective_history(
        trial_df, ["o1", "o2"], aggregate_name="obj_sum"
    )
    assert list(history["obj_sum"]) == pytest.approx([1.0, 0.9, 1.1])


def test_plot_ax_results_one_objective_sets_summary_figure():
    matplotlib.use("Agg")
    config = {
        "optimization": {
            "mode": "scalarized",
            "objectives": [
                {"name": "kpi", "original_name": "kpi", "direction": "maximize"}
            ],
            "inverse_design": {"aggregate_name": "objective_sum"},
        },
        "_project_root": ".",
    }
    trial_df = pd.DataFrame({"trial_index": [0, 1, 2], "kpi": [0.1, 0.4, 0.3]})
    viz = plot_ax_results(trial_df, config=config)
    assert viz["figures"].get("summary_figure") is not None
    assert viz.get("summary", {}).get("figure") is not None


def test_prepare_objective_history_minimize_flips_sign():
    trial_df = pd.DataFrame(
        {"trial_index": [0, 1], "o1": [0.2, 0.5]}
    )
    history = prepare_objective_history(
        trial_df, ["o1"], minimize_metrics=["o1"], aggregate_name="obj_sum"
    )
    assert list(history["obj_sum"]) == pytest.approx([-0.2, -0.5])


def test_oracle_metric_to_target_uses_config_not_hardcoded(tmp_path):
    """Surrogate path uses model_target / oracle_model_targets, not membrane hardcodes."""
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import ExtraTreesRegressor

    from utl_Ax import _load_required_oracle_pipelines
    from utl_modelling import _resolve_model_path

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    target = "my_kpi"
    path = _resolve_model_path(
        str(model_dir),
        "{target}_{model}_{task}.pkl",
        target=target,
        task="regression",
        model="extratrees",
    )
    joblib.dump(Pipeline([("model", ExtraTreesRegressor(n_estimators=5))]), path)

    config = {
        "modeling": {
            "model_selection": "extratrees",
            "model_filename_pattern": "{target}_{model}_{task}.pkl",
            "oracle_model_targets": {"metric_a": target},
        },
        "optimization": {
            "objectives": [
                {"name": "norm_a", "original_name": "metric_a", "model_target": target}
            ]
        },
    }
    loaded = _load_required_oracle_pipelines(
        model_path=str(model_dir),
        config=config,
        required_metric_keys=["metric_a"],
    )
    assert "metric_a" in loaded