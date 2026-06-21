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
    
    # Determine project root if not provided
    if project_root is None:
        # Try to find project root by looking for config.yaml or .git
        current = Path(__file__).parent.absolute()
        while current != current.parent:
            if (current / "config.yaml").exists() or (current / ".git").exists():
                project_root = str(current)
                break
            current = current.parent
        
        if project_root is None:
            raise FileNotFoundError(
                "Could not locate project root. Please provide project_root parameter or ensure "
                "config.yaml or .git exists in the project directory."
            )
    
    project_root = str(Path(project_root).absolute())
    
    # Determine config file path
    if config_path is None:
        config_path = os.path.join(project_root, "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if config is None:
        config = {}

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
        elif isinstance(value, str) and key.endswith("_dir") or key.endswith("_path"):
            # Resolve path keys
            if not os.path.isabs(value) and not value.startswith(("$", "~")):
                config[key] = os.path.join(project_root, value)
            else:
                config[key] = os.path.expandvars(os.path.expanduser(value))


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
        Dictionary with keys: 'tutorial_dataset', 'cation_prop', 'anion_prop'
    """
    project_root = config["_project_root"]
    data_raw = config["project"]["paths"]["data_raw"]
    data_raw_abs = os.path.join(project_root, data_raw)
    
    return {
        "tutorial_dataset": os.path.join(data_raw_abs, config["data"]["tutorial_dataset"]),
        "cation_prop": os.path.join(data_raw_abs, config["data"]["cation_properties"]),
        "anion_prop": os.path.join(data_raw_abs, config["data"]["anion_properties"]),
    }


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
    required_sections = [
        "project", "data", "plotting", "feature_engineering",
        "modeling", "targets", "base_features", "optimization"
    ]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
