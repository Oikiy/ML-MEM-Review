"""
Configuration loader for ML-MEM workflow.

This module provides utilities to load and manage the project configuration
from YAML, with automatic path resolution and validation.
"""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


# Default configuration values. Every key a user may omit from config.yaml is
# defined here so the workflow runs with a minimal configuration file.
DEFAULTS: Dict[str, Any] = {
    "project": {
        "name": "ML-MEM-Review",
        "paths": {
            "data_raw": "data/raw",
            "models": "models",
            "outputs_figures": "outputs/figures",
        },
    },
    "data": {
        "dataset": None,
        "sheet": "",
        "targets": [],
    },
    "features": {
        "mode": "auto",
        "include": [],
        "exclude": [],
        "missing_value_threshold": 0.3,
        "low_variance_threshold": 1e-9,
    },
    "plotting": {
        "global_figure_size": [14, 10],
        "font_size": 20,
        "dpi": 300,
        "colors": {
            "boxplot": ["#25517e", "#4c7da1", "#cd7a69"],
            "violin": ["#cedae6", "#abc5d3", "#e9c2b3"],
            "scatter": "#52687e",
            "median": "#a73b3e",
            "outlier": "#872d30",
            "edge": "#81858a",
        },
        "custom_colormap": {
            "name": "my_palette",
            "colors": ["#7FAFD9", "#FFF8EE", "#E38AAA"],
        },
    },
    "feature_engineering": {
        "lookups": [],
        "synthetic_feature": [],
    },
    "data_analysis": {
        "correlation_thresholds": [0, 0.1],
        "pearson_heatmap_show_annotations": True,
    },
    "modeling": {
        "model_selection": "extratrees",
        "task": "auto",
        "random_state": 42,
        "train_test_ratio": 0.2,
        "n_estimators": 300,
        "cv_folds": 5,
        "use_kfold": True,
        "use_hyperparam_tuning": True,
        "search_cv": "randomizedsearchcv",
        "n_iter_search": 20,
        "pipeline": {
            "categorical_encoder": "onehot",
            "categorical_encoder_params": {},
            "numerical_scaler": "passthrough",
            "numerical_scaler_params": {},
            "categorical_imputer_strategy": "most_frequent",
            "numerical_imputer_strategy": "median",
            "feature_selector": "passthrough",
            "feature_selector_params": {},
            "model": "extratrees",
            "default_model_params": {"n_estimators": 300, "random_state": 42},
            "preprocess_params": {},
            "pipeline_params": {},
        },
        # null => use built-in per-model search space at train time
        "param_dist": None,
        "model_filename_pattern": "{target}_{model}_{task}.pkl",
    },
    "optimization": {
        "mode": "scalarized",
        "sobol_init": 5,
        "n_trials": 20,
        "save_checkpoint_every": 5,
        "client_cache_file": "ax_client.pkl",
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` (override wins)."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Optional[str] = None, project_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and parse the project configuration file.
    
    Parameters
    ----------
    config_path : str, optional
        Path to the config.yaml file. If None, searches in project root.
    project_root : str, optional
        Project root directory. If None, infers from file structure.
    
    Returns
    -------
    dict
        Configuration dictionary with resolved absolute paths.
    
    Raises
    ------
    FileNotFoundError
        If config.yaml or project root cannot be found.
    """
    
    # Normalize config path first (relative paths resolve against the process CWD).
    if config_path is not None:
        config_path = str(Path(config_path).expanduser().resolve())

    # Determine project root if not provided
    if project_root is None:
        # Prefer the config file's own directory when one was given: this makes
        # the package reusable from any project without extra parameters.
        if config_path is not None:
            project_root = str(Path(config_path).parent)
        else:
            # Try to find project root by looking for config.yaml or .git
            current = Path(__file__).parent.absolute()
            while current != current.parent:
                if (current / "config.yaml").exists() or (current / ".git").exists():
                    project_root = str(current)
                    break
                current = current.parent

            if project_root is None:
                raise FileNotFoundError(
                    "Could not locate project root. Please provide project_root "
                    "parameter or ensure config.yaml or .git exists in the project "
                    "directory."
                )

    project_root = str(Path(project_root).expanduser().resolve())

    # Determine config file path
    if config_path is None:
        config_path = os.path.join(project_root, "config.yaml")
    elif not os.path.isabs(config_path):
        # If a caller passed project_root + a bare relative name, anchor it.
        candidate = Path(project_root) / config_path
        if candidate.exists():
            config_path = str(candidate.resolve())

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    if config is None:
        config = {}

    config = _deep_merge(DEFAULTS, config)

    _coerce_numeric_threshold_values(config)
    
    # Store project root and config path in config for reference
    config["_project_root"] = project_root
    config["_config_path"] = config_path
    
    # Resolve and expand all paths relative to project root
    _resolve_paths(config, project_root)
    
    return config


def _coerce_numeric_threshold_values(config: Dict[str, Any]) -> None:
    """
    Coerce numeric threshold strings to floats.

    YAML may parse scientific notation like ``1e-9`` as a string in some setups.
    This normalizes threshold values while preserving non-numeric expressions
    such as ">=0.06".
    """
    numeric_pattern = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

    def _walk(obj: Any, parent_key: Optional[str] = None) -> Any:
        if isinstance(obj, dict):
            return {key: _walk(value, key) for key, value in obj.items()}

        if isinstance(obj, list):
            return [_walk(item, parent_key) for item in obj]

        if (
            isinstance(obj, str)
            and parent_key is not None
            and "threshold" in parent_key.lower()
            and numeric_pattern.match(obj.strip())
        ):
            return float(obj)

        return obj

    normalized = _walk(config)
    config.clear()
    config.update(normalized)


def _resolve_paths(config: Dict[str, Any], project_root: str) -> None:
    """
    Recursively resolve relative paths in config to absolute paths.
    
    Modifies config in place.
    """
    for key, value in config.items():
        if isinstance(value, dict):
            _resolve_paths(value, project_root)
        elif key.endswith("_path") or (key.endswith("_dir") and isinstance(value, str)):
            # Resolve path keys
            if not os.path.isabs(value) and not value.startswith(("$", "~")):
                config[key] = os.path.join(project_root, value)
            else:
                config[key] = os.path.expandvars(os.path.expanduser(value))


def get_feature_engineering_lookup_paths(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Resolve feature engineering lookup file paths from grouped lookup config.

    Parameters
    ----------
    config : dict
        Configuration dictionary from load_config().

    Returns
    -------
    dict
        Mapping from align_column to resolved lookup file path.
    """
    project_root = config["_project_root"]
    lookups = config.get("feature_engineering", {}).get("lookups", [])
    lookup_by_align_column = {}

    for lookup in lookups:
        if not isinstance(lookup, dict):
            continue

        align_column = lookup.get("align_column")
        lookup_path = lookup.get("lookup_path")

        if not align_column or not lookup_path:
            continue

        if not os.path.isabs(lookup_path) and not lookup_path.startswith(("$", "~")):
            resolved_lookup_path = os.path.join(project_root, lookup_path)
        else:
            resolved_lookup_path = os.path.expandvars(os.path.expanduser(lookup_path))

        lookup_by_align_column[align_column] = resolved_lookup_path

    return lookup_by_align_column


def get_data_paths(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Build complete data file paths from config.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary from load_config().
    
    Returns
    -------
    dict
        Dictionary with keys:
        - dataset
        - lookup_by_align_column (mapping align_column -> resolved lookup file path)
    """
    project_root = config["_project_root"]
    data_raw = config["project"]["paths"]["data_raw"]
    data_raw_abs = os.path.join(project_root, data_raw)

    lookup_by_align_column = get_feature_engineering_lookup_paths(config)

    data_paths = {
        "dataset": os.path.join(data_raw_abs, config["data"]["dataset"]),
        "lookup_by_align_column": lookup_by_align_column,
    }

    return data_paths


def get_model_dir(config: Dict[str, Any]) -> str:
    """
    Get the absolute path to the models directory.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary from load_config().
    
    Returns
    -------
    str
        Absolute path to models directory.
    """
    project_root = config["_project_root"]
    models_dir = config["project"]["paths"]["models"]
    return os.path.join(project_root, models_dir)


def get_figures_dir(config: Dict[str, Any]) -> str:
    """
    Get the absolute path to the figures output directory.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary from load_config().
    
    Returns
    -------
    str
        Absolute path to figures directory.
    """
    project_root = config["_project_root"]
    figures_dir = config["project"]["paths"]["outputs_figures"]
    return os.path.join(project_root, figures_dir)


def get_plotting_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract plotting configuration.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary from load_config().
    
    Returns
    -------
    dict
        Plotting configuration subset.
    """
    return config.get("plotting", {})


def get_feature_engineering_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract feature engineering configuration.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary from load_config().
    
    Returns
    -------
    dict
        Feature engineering configuration subset.
    """
    return config.get("feature_engineering", {})


def get_optimization_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract optimization (Ax) configuration.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary from load_config().
    
    Returns
    -------
    dict
        Optimization configuration subset.
    """
    return config.get("optimization", {})


def get_modeling_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract modeling configuration.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary from load_config().
    
    Returns
    -------
    dict
        Modeling configuration subset.
    """
    return config.get("modeling", {})


def get_project_definition_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract project definition configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary from load_config().

    Returns
    -------
    dict
        Project definition configuration subset.
    """
    return config.get("project_definition", {})


def get_project_definition_target(config: Dict[str, Any]) -> list[str]:
    """Backward-compatible alias for :func:`resolve_targets`."""
    return resolve_targets(config)


def resolve_targets(config: Dict[str, Any]) -> list[str]:
    """
    Return the configured target (output / KPI) columns.

    New config format: ``data.targets``. Legacy fallbacks
    (``project_definition.target``) are still honoured.

    Parameters
    ----------
    config : dict
        Configuration dictionary from load_config().

    Returns
    -------
    list[str]
        List of target column names.

    Raises
    ------
    ValueError
        If no targets are configured.
    """
    targets = config.get("data", {}).get("targets", [])
    if not targets:
        targets = config.get("project_definition", {}).get("target", [])
    if not targets:
        raise ValueError(
            "No targets configured. Add your output column name(s) to config.yaml "
            "under 'data.targets' (e.g.  targets: [\"your_kpi\"])."
        )
    return list(targets)


def get_project_definition_features(config: Dict[str, Any]) -> list[str]:
    """Backward-compatible alias for the manually configured feature list."""
    features_cfg = config.get("features", {})
    include = features_cfg.get("include", [])
    if not include:
        include = config.get("features_to_model", [])
    if not include:
        include = config.get("project_definition", {}).get("features", [])
    return list(include)


def resolve_features_to_model(config: Dict[str, Any], df=None) -> list[str]:
    """
    Return the columns to use as model features.

    mode = "auto" (default):  every usable column except the targets and the
    configured exclusions. Constant, all-missing, or mostly-missing columns
    are skipped. Works with any dataset without editing config.yaml.

    mode = "manual":          exactly the columns in ``features.include``
    (with legacy fallbacks to ``features_to_model`` / ``project_definition.features``).

    Parameters
    ----------
    config : dict
        Configuration dictionary from load_config().
    df : pandas.DataFrame, optional
        Dataset used for auto mode. Required when mode is "auto".

    Returns
    -------
    list[str]
        Feature column names.
    """
    features_cfg = config.get("features", {})
    mode = str(features_cfg.get("mode", "auto")).strip().lower()
    targets = set(str(t) for t in resolve_targets(config))

    if mode != "auto":
        include = features_cfg.get("include", [])
        if not include:
            include = config.get("features_to_model", [])
        if not include:
            include = config.get("project_definition", {}).get("features", [])
        return [c for c in include if df is None or c in df.columns]

    if df is None:
        raise ValueError(
            "features.mode is 'auto' but no dataset (df) was provided to "
            "resolve_features_to_model()."
        )

    excluded = set(str(e) for e in features_cfg.get("exclude", []) or [])
    missing_threshold = float(features_cfg.get("missing_value_threshold", 0.3))

    kept = []
    for column in df.columns:
        column_str = str(column)
        if column_str in targets or column_str in excluded:
            continue
        missing_fraction = float(df[column].isna().mean())
        if missing_fraction > missing_threshold:
            continue
        non_null = df[column].dropna()
        if non_null.empty:
            continue
        if len(non_null.unique()) <= 1:
            continue
        kept.append(column)
    return kept


def check_dataset_compatibility(config: Dict[str, Any], df) -> Dict[str, Any]:
    """
    Compare the configured targets/features against a loaded dataset.

    Returns a report dict (never raises) that callers can print or display:

    - ``targets_found`` / ``targets_missing``
    - ``features_used`` / ``features_skipped`` (auto mode)
    - ``n_rows``
    """
    targets = resolve_targets(config)
    columns = [str(c) for c in df.columns]

    targets_found = [t for t in targets if t in columns]
    targets_missing = [t for t in targets if t not in columns]

    features_used = resolve_features_to_model(config, df)
    features_skipped = [
        c for c in columns
        if str(c) not in targets and str(c) not in features_used
    ]

    return {
        "n_rows": int(len(df)),
        "n_columns": int(len(columns)),
        "targets_found": targets_found,
        "targets_missing": targets_missing,
        "features_used": features_used,
        "features_skipped": features_skipped,
    }


def print_dataset_report(config: Dict[str, Any], df) -> None:
    """Print a human-readable dataset compatibility report."""
    report = check_dataset_compatibility(config, df)
    print("=" * 70)
    print("DATASET CHECK")
    print("=" * 70)
    print(f"Rows: {report['n_rows']}   Columns: {report['n_columns']}")

    print("\nTargets:")
    if report["targets_found"]:
        for target in report["targets_found"]:
            print(f"  [OK]   {target}")
    for target in report["targets_missing"]:
        print(f"  [MISSING] {target}  <- not found in your file")

    print(f"\nFeatures used ({len(report['features_used'])}):")
    for feature in report["features_used"]:
        print(f"  - {feature}")

    if report["features_skipped"]:
        print(f"\nSkipped columns ({len(report['features_skipped'])}):")
        for feature in report["features_skipped"][:25]:
            print(f"  - {feature}")
        if len(report["features_skipped"]) > 25:
            print(f"  ... and {len(report['features_skipped']) - 25} more")
    print("=" * 70)


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate that all required config sections are present.

    Parameters
    ----------
    config : dict
        Configuration dictionary from load_config().

    Raises
    ------
    ValueError
        If required sections are missing.
    """
    required_sections = ["data", "modeling", "plotting", "feature_engineering"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
