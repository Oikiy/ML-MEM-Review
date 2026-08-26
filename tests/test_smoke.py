"""End-to-end smokes on the example gallery (isolated copies)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from run_pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]


def _prepare_example(tmp_path: Path, name: str, n_estimators: int = 20) -> Path:
    src = ROOT / "examples" / name
    dest = tmp_path / name
    shutil.copytree(
        src,
        dest,
        ignore=shutil.ignore_patterns("Colab.ipynb", "README.md", "models", "outputs"),
    )
    config_path = dest / "config.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    modeling = payload.setdefault("modeling", {})
    modeling["use_hyperparam_tuning"] = False
    modeling["n_estimators"] = n_estimators
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


@pytest.mark.smoke
def test_example_01_minimal_regression_model_stage(tmp_path):
    config_path = _prepare_example(tmp_path, "01_minimal_regression")
    config = run_pipeline(
        config_path=str(config_path),
        stages="model",
        force_retrain=True,
    )
    figures = Path(config["_project_root"]) / "outputs" / "figures"
    models = Path(config["_project_root"]) / "models"
    assert (figures / "parity_permeance.png").is_file()
    assert any(models.glob("*permeance*regressor*.pkl"))


@pytest.mark.smoke
def test_example_02_classification_model_stage(tmp_path):
    config_path = _prepare_example(tmp_path, "02_classification")
    config = run_pipeline(
        config_path=str(config_path),
        stages="model",
        force_retrain=True,
    )
    figures = Path(config["_project_root"]) / "outputs" / "figures"
    models = Path(config["_project_root"]) / "models"
    assert (figures / "confusion_label.png").is_file()
    assert any(models.glob("*label*classifier*.pkl"))


@pytest.mark.smoke
def test_optimize_skipped_when_not_configured(tmp_path):
    config_path = _prepare_example(tmp_path, "02_classification")
    config = run_pipeline(
        config_path=str(config_path),
        stages="optimize",
        force_retrain=True,
    )
    figures = Path(config["_project_root"]) / "outputs" / "figures"
    assert not (figures / "inverse_design_summary.png").exists()
    assert config["optimization"].get("objectives") in (None, [])
