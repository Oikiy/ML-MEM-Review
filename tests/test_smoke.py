"""End-to-end smokes on the example gallery (isolated copies)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from run_pipeline import STAGES, normalize_stages, run_pipeline

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


def test_comma_separated_stages_are_parsed():
    assert normalize_stages("eda,model") == ["eda", "model"]
    assert normalize_stages("eda, model") == ["eda", "model"]
    assert normalize_stages("all") == list(STAGES)
    with pytest.raises(ValueError, match="Unknown stage"):
        run_pipeline(config_path="missing.yaml", stages="eda,not_a_stage")


def test_colab_notebooks_sync_clone_and_avoid_comma_stage_flag():
    nbs = list((ROOT / "examples").glob("*/Colab.ipynb"))
    nbs.append(ROOT / "notebooks" / "Complete_notebook" / "Ml-MEM-Tut.ipynb")
    assert len(nbs) >= 6
    for nb_path in nbs:
        src = "\n".join("".join(c.get("source") or []) for c in json.loads(
            nb_path.read_text(encoding="utf-8")
        )["cells"])
        assert '"fetch"' in src, nb_path
        assert "FETCH_HEAD" in src, nb_path
        assert ".colab_install_ok" in src, nb_path
        if nb_path.name == "Colab.ipynb":
            assert "PYTHONPATH" in src, nb_path
            assert "--stage eda,model" not in src, nb_path
            assert 'for stage in ("eda", "model"' in src, nb_path


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
