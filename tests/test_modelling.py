import numpy as np
import pandas as pd
import pytest

from utl_modelling import (
    _build_default_pipeline,
    _drop_low_information_columns,
    _resolve_model_path,
    _task_to_filename_label,
)


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


def test_build_default_pipeline_raises_on_no_features():
    X_train = pd.DataFrame()
    with pytest.raises(ValueError, match="No usable features"):
        _build_default_pipeline(
            X_train=X_train,
            task="regression",
            random_state=42,
            n_estimators=50,
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