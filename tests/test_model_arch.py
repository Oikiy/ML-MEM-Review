"""Tests for switchable model architectures."""

import pytest
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier

from utl_model_arch import (
    get_default_param_dist,
    is_tree_model,
    list_available_models,
    make_estimator,
    normalize_model_key,
    resolve_param_dist,
)


def test_normalize_aliases():
    assert normalize_model_key("ExtraTrees") == "extratrees"
    assert normalize_model_key("rf") == "randomforest"
    assert normalize_model_key("random_forest") == "randomforest"
    assert normalize_model_key("logreg") == "logistic"


def test_list_available_models():
    available = list_available_models()
    assert "extratrees" in available["regression"]
    assert "logistic" in available["classification"]
    assert "logistic" not in available["regression"]


def test_make_estimator_regression_and_classification():
    et_reg = make_estimator("extratrees", "regression", n_estimators=10)
    et_clf = make_estimator("extratrees", "classification", n_estimators=10)
    rf_reg = make_estimator("randomforest", "regression", n_estimators=10)
    ridge = make_estimator("ridge", "regression")
    log = make_estimator("logistic", "classification")
    assert isinstance(et_reg, ExtraTreesRegressor)
    assert isinstance(et_clf, ExtraTreesClassifier)
    assert isinstance(rf_reg, RandomForestRegressor)
    assert isinstance(ridge, Ridge)
    assert isinstance(log, LogisticRegression)


def test_make_estimator_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model_selection"):
        make_estimator("not_a_model", "regression")


def test_make_estimator_wrong_task_raises():
    with pytest.raises(ValueError, match="Unknown model_selection"):
        make_estimator("logistic", "regression")


def test_default_param_dist_differs_by_model():
    tree = get_default_param_dist("extratrees", "regression")
    ridge = get_default_param_dist("ridge", "regression")
    assert "model__n_estimators" in tree
    assert "model__alpha" in ridge
    assert "model__n_estimators" not in ridge


def test_resolve_param_dist_falls_back_when_tree_grid_on_linear():
    tree_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt"],
    }
    resolved = resolve_param_dist("ridge", "regression", user_param_dist=tree_grid)
    assert "model__alpha" in resolved
    assert "model__n_estimators" not in resolved


def test_resolve_param_dist_keeps_explicit_user_grid():
    user = {"model__alpha": [0.5, 5.0]}
    assert resolve_param_dist("ridge", "regression", user_param_dist=user) == user


def test_is_tree_model():
    assert is_tree_model("extratrees")
    assert is_tree_model("randomforest")
    assert not is_tree_model("ridge")
    assert not is_tree_model("logistic")


def test_train_model_respects_model_selection(tmp_path):
    import pandas as pd
    from utl_modelling import train_model

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
project:
  name: arch_test
  paths:
    models: models
data:
  dataset: x.csv
  targets: [y]
features:
  mode: auto
modeling:
  model_selection: randomforest
  task: regression
  use_hyperparam_tuning: false
  n_estimators: 15
  random_state: 0
""",
        encoding="utf-8",
    )
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    pipe, *_ = train_model(
        df, target="y", config_path=str(cfg), force_retrain=True
    )
    assert isinstance(pipe.named_steps["model"], RandomForestRegressor)
    model_files = list((tmp_path / "models").glob("*.pkl"))
    assert any("randomforest" in p.name for p in model_files)
