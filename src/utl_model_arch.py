"""Configurable training pipeline and model registry.

Users switch architectures with one config key::

    modeling:
      model_selection: extratrees   # or randomforest, ridge, ...

Registries are task-aware (regression vs classification). Default hyperparameter
search spaces are provided per (model, task) so switching models does not
require hand-writing ``param_dist``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import RFE, SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)


def _make_one_hot_encoder(**kwargs):
    """Create OneHotEncoder compatible with older and newer scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, **kwargs)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, **kwargs)


# -----------------------
# Component registries
# -----------------------

CATEGORICAL_ENCODERS = {
    "onehot": lambda **kwargs: _make_one_hot_encoder(**kwargs),
    "ordinal": lambda **kwargs: OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        **kwargs,
    ),
    "passthrough": lambda **kwargs: "passthrough",
}

NUMERICAL_SCALERS = {
    "standard": lambda **kwargs: StandardScaler(**kwargs),
    "minmax": lambda **kwargs: MinMaxScaler(**kwargs),
    "robust": lambda **kwargs: RobustScaler(**kwargs),
    "passthrough": lambda **kwargs: "passthrough",
}

FEATURE_SELECTORS = {
    "passthrough": lambda **kwargs: "passthrough",
    "rfe": lambda **kwargs: RFE(**kwargs),
    "selectkbest": lambda **kwargs: SelectKBest(score_func=f_regression, **kwargs),
}

# Factory: (random_state, n_estimators, extra_params) -> estimator
EstimatorFactory = Callable[..., Any]

MODELS_REGRESSION: dict[str, EstimatorFactory] = {
    "extratrees": lambda random_state=42, n_estimators=300, **kw: ExtraTreesRegressor(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1, **kw
    ),
    "randomforest": lambda random_state=42, n_estimators=300, **kw: RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1, **kw
    ),
    "histgradientboosting": lambda random_state=42, n_estimators=300, **kw: HistGradientBoostingRegressor(
        max_iter=n_estimators, random_state=random_state, **kw
    ),
    "ridge": lambda random_state=42, n_estimators=300, **kw: Ridge(
        random_state=random_state, **{k: v for k, v in kw.items() if k != "n_estimators"}
    ),
    "elasticnet": lambda random_state=42, n_estimators=300, **kw: ElasticNet(
        random_state=random_state,
        max_iter=5000,
        **{k: v for k, v in kw.items() if k != "n_estimators"},
    ),
    "linear": lambda random_state=42, n_estimators=300, **kw: LinearRegression(
        **{k: v for k, v in kw.items() if k not in {"n_estimators", "random_state"}}
    ),
}

MODELS_CLASSIFICATION: dict[str, EstimatorFactory] = {
    "extratrees": lambda random_state=42, n_estimators=300, **kw: ExtraTreesClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1, **kw
    ),
    "randomforest": lambda random_state=42, n_estimators=300, **kw: RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1, **kw
    ),
    "histgradientboosting": lambda random_state=42, n_estimators=300, **kw: HistGradientBoostingClassifier(
        max_iter=n_estimators, random_state=random_state, **kw
    ),
    "logistic": lambda random_state=42, n_estimators=300, **kw: LogisticRegression(
        max_iter=2000,
        random_state=random_state,
        **{k: v for k, v in kw.items() if k != "n_estimators"},
    ),
    "ridge": lambda random_state=42, n_estimators=300, **kw: RidgeClassifier(
        random_state=random_state,
        **{k: v for k, v in kw.items() if k != "n_estimators"},
    ),
}

# Backward-compatible alias used by older notebook/docs (regression only).
MODELS = MODELS_REGRESSION

_TREE_MODEL_KEYS = frozenset({"extratrees", "randomforest", "histgradientboosting"})

_ALIASES = {
    "et": "extratrees",
    "extra_trees": "extratrees",
    "extra-trees": "extratrees",
    "rf": "randomforest",
    "random_forest": "randomforest",
    "random-forest": "randomforest",
    "hgb": "histgradientboosting",
    "hist_gradient_boosting": "histgradientboosting",
    "hgbr": "histgradientboosting",
    "hgbc": "histgradientboosting",
    "logreg": "logistic",
    "logistic_regression": "logistic",
    "elastic_net": "elasticnet",
    "linreg": "linear",
    "linear_regression": "linear",
    "ridge_classifier": "ridge",
}


def normalize_model_key(name: str) -> str:
    """Normalize a user-facing model name to a registry key."""
    key = str(name).strip().lower().replace(" ", "_")
    key = _ALIASES.get(key, key)
    key = key.replace("-", "_")
    # allow random_forest -> try alias already; collapse double forms
    compact = key.replace("_", "")
    compact_aliases = {
        "extratrees": "extratrees",
        "randomforest": "randomforest",
        "histgradientboosting": "histgradientboosting",
        "elasticnet": "elasticnet",
        "logisticregression": "logistic",
        "linearregression": "linear",
        "ridgeclassifier": "ridge",
    }
    return compact_aliases.get(compact, key if "_" in key else compact)


def get_model_registry(task: str) -> dict[str, EstimatorFactory]:
    task_l = str(task).strip().lower()
    if task_l == "classification":
        return MODELS_CLASSIFICATION
    if task_l == "regression":
        return MODELS_REGRESSION
    raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")


def list_available_models(task: str | None = None) -> dict[str, list[str]]:
    """Return available model keys, optionally filtered by task."""
    if task is None:
        return {
            "regression": sorted(MODELS_REGRESSION),
            "classification": sorted(MODELS_CLASSIFICATION),
        }
    return {str(task): sorted(get_model_registry(task))}


def is_tree_model(model_selection: str) -> bool:
    return normalize_model_key(model_selection) in _TREE_MODEL_KEYS


def make_estimator(
    model_selection: str,
    task: str,
    random_state: int = 42,
    n_estimators: int = 300,
    **extra_params: Any,
):
    """Instantiate an estimator for the given model key and task."""
    key = normalize_model_key(model_selection)
    registry = get_model_registry(task)
    if key not in registry:
        available = sorted(registry)
        raise ValueError(
            f"Unknown model_selection '{model_selection}' for task '{task}'. "
            f"Available: {available}"
        )
    params = dict(extra_params)
    return registry[key](random_state=random_state, n_estimators=n_estimators, **params)


# Default RandomizedSearch spaces keyed by (model_key, task)
_TREE_REG_DIST = {
    "model__n_estimators": [200, 300, 500, 800],
    "model__max_depth": [None, 10, 20, 40],
    "model__min_samples_leaf": [1, 2, 5],
    "model__max_features": ["sqrt", "log2", 0.5, 0.8],
}
_TREE_CLF_DIST = {
    **_TREE_REG_DIST,
    "model__criterion": ["gini", "entropy"],
}
_HGB_REG_DIST = {
    "model__max_iter": [100, 200, 300],
    "model__max_depth": [None, 3, 6, 10],
    "model__learning_rate": [0.05, 0.1, 0.2],
    "model__min_samples_leaf": [5, 10, 20],
}
_HGB_CLF_DIST = dict(_HGB_REG_DIST)

PARAM_DIST: dict[tuple[str, str], dict[str, Any]] = {
    ("extratrees", "regression"): dict(_TREE_REG_DIST),
    ("extratrees", "classification"): dict(_TREE_CLF_DIST),
    ("randomforest", "regression"): dict(_TREE_REG_DIST),
    ("randomforest", "classification"): dict(_TREE_CLF_DIST),
    ("histgradientboosting", "regression"): dict(_HGB_REG_DIST),
    ("histgradientboosting", "classification"): dict(_HGB_CLF_DIST),
    ("ridge", "regression"): {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
    ("ridge", "classification"): {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
    ("elasticnet", "regression"): {
        "model__alpha": [0.01, 0.1, 1.0, 10.0],
        "model__l1_ratio": [0.1, 0.5, 0.9],
    },
    ("linear", "regression"): {},
    ("logistic", "classification"): {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__penalty": ["l2"],
    },
}


def get_default_param_dist(model_selection: str, task: str) -> dict[str, Any]:
    """Return the default hyperparameter search space for model+task."""
    key = normalize_model_key(model_selection)
    task_l = str(task).strip().lower()
    dist = PARAM_DIST.get((key, task_l))
    if dist is None:
        return {}
    return dict(dist)


def resolve_param_dist(
    model_selection: str,
    task: str,
    user_param_dist: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prefer user ``param_dist`` when non-empty; else model-specific defaults.

    If a leftover tree-style grid is paired with a non-tree model (common when
    only ``model_selection`` is changed in config.yaml), fall back to the
    model-specific defaults instead of crashing RandomizedSearchCV.
    """
    defaults = get_default_param_dist(model_selection, task)
    if not isinstance(user_param_dist, dict) or len(user_param_dist) == 0:
        return defaults

    tree_only_keys = {
        "model__max_features",
        "model__min_samples_leaf",
        "model__criterion",
        "model__n_estimators",
        "model__max_depth",
    }
    if not is_tree_model(model_selection) and any(k in user_param_dist for k in tree_only_keys):
        default_keys = set(defaults)
        if default_keys and not default_keys.intersection(user_param_dist):
            return defaults
    return dict(user_param_dist)


# -----------------------
# Pipeline config / build
# -----------------------


@dataclass
class PipelineConfig:
    categorical_encoder: str = "onehot"
    categorical_encoder_params: dict[str, Any] = field(default_factory=dict)

    numerical_scaler: str = "passthrough"
    numerical_scaler_params: dict[str, Any] = field(default_factory=dict)

    categorical_imputer_strategy: str = "most_frequent"
    numerical_imputer_strategy: str = "median"

    feature_selector: str = "passthrough"
    feature_selector_params: dict[str, Any] = field(default_factory=dict)

    model: str = "extratrees"
    task: str = "regression"
    default_model_params: dict[str, Any] = field(
        default_factory=lambda: {"n_estimators": 300, "random_state": 42}
    )

    preprocess_params: dict[str, Any] = field(default_factory=dict)
    pipeline_params: dict[str, Any] = field(default_factory=dict)


def _make_component(registry: dict, name: str, params: dict):
    if name not in registry:
        raise ValueError(f"Unknown component '{name}'. Available: {list(registry)}")
    return registry[name](**params)


def build_training_pipeline(config: PipelineConfig):
    """Build a sklearn Pipeline from a PipelineConfig instance."""
    cat_encoder = _make_component(
        CATEGORICAL_ENCODERS,
        config.categorical_encoder,
        config.categorical_encoder_params,
    )
    num_scaler = _make_component(
        NUMERICAL_SCALERS,
        config.numerical_scaler,
        config.numerical_scaler_params,
    )

    selector_params = dict(config.feature_selector_params)
    if config.feature_selector == "selectkbest" and "score_func" not in selector_params:
        score = f_classif if str(config.task).lower() == "classification" else f_regression
        selector_params = {**selector_params, "score_func": score}
    selector = _make_component(FEATURE_SELECTORS, config.feature_selector, selector_params)

    model_params = dict(config.default_model_params)
    random_state = int(model_params.pop("random_state", 42))
    n_estimators = int(model_params.pop("n_estimators", 300))
    model = make_estimator(
        config.model,
        task=config.task,
        random_state=random_state,
        n_estimators=n_estimators,
        **model_params,
    )

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=config.categorical_imputer_strategy)),
            ("encoder", cat_encoder),
        ]
    )
    numerical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=config.numerical_imputer_strategy)),
            ("scaler", num_scaler),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipe, make_column_selector(dtype_include=["object", "category"])),
            ("num", numerical_pipe, make_column_selector(dtype_exclude=["object", "category"])),
        ],
        **config.preprocess_params,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("selector", selector),
            ("model", model),
        ],
        **config.pipeline_params,
    )


def build_pipeline_from_modeling_config(
    modeling_cfg: dict[str, Any],
    task: str = "regression",
) -> Pipeline:
    """Build a training pipeline from a ``modeling`` config section."""
    pipeline_cfg_dict: dict[str, Any] = dict(modeling_cfg.get("pipeline", {}))
    model_key = normalize_model_key(
        modeling_cfg.get("model_selection", pipeline_cfg_dict.get("model", "extratrees"))
    )
    pipeline_cfg_dict["model"] = model_key
    pipeline_cfg_dict["task"] = task

    if "default_model_params" not in pipeline_cfg_dict:
        pipeline_cfg_dict["default_model_params"] = {
            "n_estimators": modeling_cfg.get("n_estimators", 300),
            "random_state": modeling_cfg.get("random_state", 42),
        }
    else:
        default_model_params = dict(pipeline_cfg_dict.get("default_model_params", {}))
        default_model_params.setdefault("n_estimators", modeling_cfg.get("n_estimators", 300))
        default_model_params.setdefault("random_state", modeling_cfg.get("random_state", 42))
        pipeline_cfg_dict["default_model_params"] = default_model_params

    # Drop unknown dataclass fields if present
    allowed = {f.name for f in PipelineConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in pipeline_cfg_dict.items() if k in allowed}
    return build_training_pipeline(PipelineConfig(**filtered))


def build_model_pipeline(df_frame, feature_list: list[str], modeling_cfg: dict[str, Any]) -> Pipeline:
    """Backward-compatible wrapper used by existing notebook calls."""
    _ = df_frame, feature_list
    task = modeling_cfg.get("task", "regression")
    if str(task).lower() == "auto":
        task = "regression"
    return build_pipeline_from_modeling_config(modeling_cfg, task=task)


def get_hyperparameter_space(config: dict[str, Any], task: str = "regression") -> dict[str, Any]:
    """Return search space from config, falling back to model-specific defaults."""
    modeling_cfg = config.get("modeling", {})
    model_selection = modeling_cfg.get("model_selection", "extratrees")
    return resolve_param_dist(
        model_selection,
        task=task,
        user_param_dist=modeling_cfg.get("param_dist"),
    )