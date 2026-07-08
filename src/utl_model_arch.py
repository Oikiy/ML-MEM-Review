"""Configurable training pipeline builder.

This module exposes a registry-driven pipeline factory used by notebooks and
scripts to keep preprocessing/model choices in configuration.
"""

from dataclasses import dataclass, field
from typing import Any

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
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
# Registries
# -----------------------

CATEGORICAL_ENCODERS = {
    "onehot": lambda **kwargs: _make_one_hot_encoder(**kwargs),
    "ordinal": lambda **kwargs: OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        **kwargs
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

MODELS = {
    "extratrees": lambda **kwargs: ExtraTreesRegressor(n_jobs=-1, **kwargs),
    "randomforest": lambda **kwargs: RandomForestRegressor(n_jobs=-1, **kwargs),
    "ridge": lambda **kwargs: Ridge(**kwargs),
    "elasticnet": lambda **kwargs: ElasticNet(**kwargs),
    "linear": lambda **kwargs: LinearRegression(**kwargs),
}


# -----------------------
# Config
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
    default_model_params: dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 300,
        "random_state": 42,
    })

    preprocess_params: dict[str, Any] = field(default_factory=dict)
    pipeline_params: dict[str, Any] = field(default_factory=dict)


# -----------------------
# Builders
# -----------------------

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
    selector = _make_component(
        FEATURE_SELECTORS,
        config.feature_selector,
        config.feature_selector_params,
    )
    model = _make_component(
        MODELS,
        config.model,
        config.default_model_params,
    )

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=config.categorical_imputer_strategy)),
        ("encoder", cat_encoder),
    ])

    numerical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=config.numerical_imputer_strategy)),
        ("scaler", num_scaler),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipe, make_column_selector(dtype_include=["object", "category"])),
            ("num", numerical_pipe, make_column_selector(dtype_exclude=["object", "category"])),
        ],
        **config.preprocess_params,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("selector", selector),
            ("model", model),
        ],
        **config.pipeline_params,
    )

    return pipe


def build_pipeline_from_modeling_config(modeling_cfg: dict[str, Any]) -> Pipeline:
    """Build a training pipeline from a ``modeling`` config section."""
    pipeline_cfg_dict: dict[str, Any] = dict(modeling_cfg.get("pipeline", {}))

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

    pipeline_config = PipelineConfig(**pipeline_cfg_dict)
    return build_training_pipeline(pipeline_config)


def build_model_pipeline(df_frame, feature_list: list[str], modeling_cfg: dict[str, Any]) -> Pipeline:
    """Backward-compatible wrapper used by existing notebook calls.

    ``df_frame`` and ``feature_list`` are intentionally unused here because the
    shared builder relies on dtype selectors at fit time.
    """
    _ = df_frame, feature_list
    return build_pipeline_from_modeling_config(modeling_cfg)


def get_hyperparameter_space(config: dict[str, Any]) -> dict[str, Any]:
    """Return RandomizedSearchCV parameter space from config with defaults."""
    modeling_cfg = config.get("modeling", {})
    return modeling_cfg.get(
        "param_dist",
        {
            "model__n_estimators": [200, 300, 500, 800],
            "model__max_depth": [None, 10, 20, 40],
            "model__min_samples_leaf": [1, 2, 5],
            "model__max_features": ["sqrt", "log2", 0.5, 0.8],
        },
    )