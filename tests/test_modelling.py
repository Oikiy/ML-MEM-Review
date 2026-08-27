import pandas as pd
import pytest

from utl_model_arch import build_pipeline_from_modeling_config
from utl_modelling import (
    _drop_low_information_columns,
    _resolve_model_path,
    _safe_target,
    _task_to_filename_label,
    scaled_plot_fonts,
    train_model,
)


def test_scaled_plot_fonts_shrinks_with_more_items():
    title_few, _, tick_few, _ = scaled_plot_fonts(4, base=20)
    title_many, _, tick_many, _ = scaled_plot_fonts(24, base=20)
    assert title_few >= title_many
    assert tick_few >= tick_many
    assert tick_many >= 6


def test_task_to_filename_label():
    assert _task_to_filename_label("regression") == "regressor"
    assert _task_to_filename_label("classification") == "classifier"
    assert _task_to_filename_label("custom") == "custom"


def test_resolve_model_path_with_task_pattern():
    path = _resolve_model_path(
        model_dir=r"C:\models",
        model_filename_pattern="{target}_{model}_{task}.pkl",
        target="kpi_1",
        task="regression",
        model="extratrees",
    )
    assert path == r"C:\models\kpi_1_extratrees_regressor.pkl"


def test_resolve_model_path_appends_task_label():
    path = _resolve_model_path(
        model_dir=r"C:\models",
        model_filename_pattern="{target}_{model}.pkl",
        target="kpi_1",
        task="regression",
        model="extratrees",
    )
    assert path == r"C:\models\kpi_1_extratrees_regressor.pkl"


def test_resolve_model_path_sanitizes_special_characters():
    target = "Pure_Water_Permeability_{Lm^{-2}^h{-1}bar^{-1}}"
    safe = _safe_target(target)
    assert "{" not in safe
    assert "}" not in safe
    path = _resolve_model_path(
        model_dir=r"C:\models",
        model_filename_pattern="{target}_{model}_{task}.pkl",
        target=target,
        task="regression",
        model="extratrees",
    )
    assert path == rf"C:\models\{safe}_extratrees_regressor.pkl"


def test_train_model_reads_missing_threshold_from_features(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
project:
  name: threshold_test
  paths:
    models: models
data:
  dataset: x.csv
  targets: [y]
features:
  mode: auto
  missing_value_threshold: 0.55
feature_engineering:
  missing_value_threshold: 0.01
modeling:
  model_selection: extratrees
  task: regression
  use_hyperparam_tuning: false
  n_estimators: 10
  random_state: 0
""",
        encoding="utf-8",
    )
    n = 20
    half_missing = [None if i < 8 else float(i) for i in range(n)]
    df = pd.DataFrame(
        {
            "kept": [float(i) for i in range(n)],
            "half_missing": half_missing,
            "y": [float(i) for i in range(n)],
        }
    )
    pipe, X_train, X_test, _, _ = train_model(
        df, target="y", config_path=str(cfg), force_retrain=True
    )
    assert "half_missing" in X_train.columns
    assert "half_missing" in X_test.columns
    assert pipe is not None


def test_drop_low_information_columns_uses_train_only():
    X_train = pd.DataFrame(
        {
            "good": [1.0, 2.0, 3.0],
            "constant": ["x", "x", "x"],
            "high_missing": [1.0, None, None],
        }
    )
    X_test = pd.DataFrame(
        {
            "good": [4.0, 5.0],
            "constant": ["x", "x"],
            "high_missing": [None, None],
        }
    )
    X_train_out, X_test_out, dropped = _drop_low_information_columns(X_train, X_test)
    assert set(dropped) == {"constant", "high_missing"}
    assert list(X_train_out.columns) == ["good"]
    assert list(X_test_out.columns) == ["good"]


def test_build_pipeline_raises_on_no_features():
    with pytest.raises(ValueError, match="No usable features"):
        build_pipeline_from_modeling_config(
            {"model_selection": "extratrees"},
            task="regression",
            categorical_cols=[],
            numerical_cols=[],
        )


def test_evaluate_and_plot_results_classification_returns_fig():
    from utl_modelling import evaluate_and_plot_results

    y_train = pd.Series(["a", "b", "a", "b", "a", "b"])
    y_pred_train = pd.Series(["a", "b", "a", "a", "a", "b"])
    y_test = pd.Series(["a", "b", "a"])
    y_pred_test = pd.Series(["a", "b", "b"])
    fig, task = evaluate_and_plot_results(
        target="label",
        y_train=y_train,
        y_pred_train=y_pred_train,
        y_test=y_test,
        y_pred_test=y_pred_test,
        task="classification",
        config={"plotting": {}, "modeling": {"task": "classification"}},
    )
    assert task == "classification"
    assert hasattr(fig, "savefig")


def test_evaluate_and_plot_results_regression_returns_fig():
    from utl_modelling import evaluate_and_plot_results

    y_train = pd.Series([1.0, 2.0, 3.0, 4.0])
    y_pred_train = pd.Series([1.1, 1.9, 3.2, 3.8])
    y_test = pd.Series([5.0, 6.0])
    y_pred_test = pd.Series([4.8, 6.1])
    fig, task = evaluate_and_plot_results(
        target="y",
        y_train=y_train,
        y_pred_train=y_pred_train,
        y_test=y_test,
        y_pred_test=y_pred_test,
        task="regression",
        config={"plotting": {}, "modeling": {"task": "regression"}},
    )
    assert task == "regression"
    assert hasattr(fig, "savefig")