import pytest
import pandas as pd

from utl_config_loader import (
    _deep_merge,
    _resolve_paths,
    check_dataset_compatibility,
    load_config,
    get_figures_dir,
    get_model_dir,
    resolve_features_to_model,
    resolve_targets,
    validate_config,
)

MINIMAL_CONFIG = """\
project:
  name: "my_project"

data:
  dataset: "my_data.xlsx"
  sheet: ""
  targets: ["kpi_1"]

features:
  mode: "auto"
"""


def write_minimal(tmp_path, extra=""):
    path = tmp_path / "config.yaml"
    path.write_text(MINIMAL_CONFIG + extra, encoding="utf-8")
    return str(path)


def test_minimal_config_merges_defaults(tmp_path):
    config = load_config(config_path=write_minimal(tmp_path))
    assert config["project"]["name"] == "my_project"
    assert config["features"]["mode"] == "auto"
    assert config["modeling"]["model_selection"] == "extratrees"
    assert config["modeling"]["random_state"] == 42
    assert config["optimization"].get("objectives") is None
    assert config["feature_engineering"]["lookups"] == []
    assert config["_project_root"] == str(tmp_path)
    assert config["_config_path"] == str(tmp_path / "config.yaml")


def test_deep_merge_override_wins():
    base = {"features": {"mode": "auto", "include": ["a", "b"]}, "keep": 1}
    override = {"features": {"include": ["x"]}}
    merged = _deep_merge(base, override)
    assert merged == {"features": {"mode": "auto", "include": ["x"]}, "keep": 1}


def test_resolve_targets_from_data(tmp_path):
    config = load_config(config_path=write_minimal(tmp_path))
    assert resolve_targets(config) == ["kpi_1"]


def test_resolve_targets_legacy_fallback(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        'project_definition:\n  target: ["old_kpi"]\n', encoding="utf-8"
    )
    config = load_config(config_path=str(config_path))
    assert resolve_targets(config) == ["old_kpi"]


def test_resolve_targets_missing_raises(tmp_path):
    no_targets = "data:\n  dataset: x.csv\n"
    config = load_config(config_path=write_minimal(tmp_path, no_targets))
    with pytest.raises(ValueError):
        resolve_targets(config)


def test_resolve_features_auto_skips_target_constant_and_high_missing(tmp_path):
    config = load_config(config_path=write_minimal(tmp_path))
    df = pd.DataFrame(
        {
            "kpi_1": [0.1, 0.2, 0.3, 0.4],
            "good_1": [1.0, 2.0, 3.0, 4.0],
            "constant": ["x", "x", "x", "x"],
            "mostly_missing": [1.0, None, None, None],
            "cat_ok": ["red", "blue", "red", "green"],
        }
    )
    features = resolve_features_to_model(config, df)
    assert "kpi_1" not in features
    assert "constant" not in features
    assert "mostly_missing" not in features
    assert set(features) == {"good_1", "cat_ok"}


def test_resolve_features_manual_returns_present_columns(tmp_path):
    manual = (
        "features:\n"
        "  mode: manual\n"
        '  include: ["good_1", "kpi_1", "absent_col"]\n'
    )
    config = load_config(config_path=write_minimal(tmp_path, manual))
    df = pd.DataFrame({"kpi_1": [0.1, 0.2], "good_1": [1.0, 2.0]})
    assert resolve_features_to_model(config, df) == ["good_1", "kpi_1"]


def test_path_keys_relative_to_project_root(tmp_path):
    import os

    config = load_config(config_path=write_minimal(tmp_path))
    assert os.path.normpath(get_model_dir(config)) == os.path.normpath(
        str(tmp_path / "models")
    )
    assert os.path.normpath(get_figures_dir(config)) == os.path.normpath(
        str(tmp_path / "outputs" / "figures")
    )


def test_resolve_paths_precedence(tmp_path):
    cfg = {
        "in_dir": "x",
        "outdir": "y",
        "file_path": "z",
        "data_raw": "raw",
        "plain": "w",
    }
    _resolve_paths(cfg, str(tmp_path))
    assert cfg["in_dir"] == str(tmp_path / "x")
    assert cfg["file_path"] == str(tmp_path / "z")
    assert cfg["outdir"] == "y"
    assert cfg["data_raw"] == "raw"
    assert cfg["plain"] == "w"


def test_check_dataset_compatibility(tmp_path):
    config = load_config(config_path=write_minimal(tmp_path))
    df = pd.DataFrame({"kpi_1": [0.1, 0.2], "good_1": [1.0, 2.0]})
    report = check_dataset_compatibility(config, df)
    assert report["targets_found"] == ["kpi_1"]
    assert report["targets_missing"] == []
    assert report["n_rows"] == 2
    assert report["features_used"] == ["good_1"]


def test_validate_config_required_sections():
    with pytest.raises(ValueError, match="Missing required configuration section"):
        validate_config({"data": {}, "modeling": {}, "plotting": {}})