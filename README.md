# ML-MEM-Review

A reproducible ML inverse-design workflow for membrane development
(the companion repository of the membrane ML tutorial).

The same package runs on **any** tabular ML dataset — the membrane
recipe search space, feature lookups, and tutorial notebook are provided as
pre-filled examples, not hard requirements.

**Nine-step paper vs this repo.** The review uses a closed loop (1 objectives → 2 schema → 3 inspect → 4 FE/EDA → 5 surrogate → 6 SHAP → 7 inverse design → 8 lab validate → 9 refine). This repository implements the **tabular core, Steps 4–7**. Steps 1–3 and 8–9 are discussed in the teaching notebook; they are not wet-lab automation.

**Scope note:** training and evaluation support both **regression** and
**classification**. The Ax **inverse-design** stage is intentionally for
**continuous (regression) targets only** — that matches the membrane tutorial
KPIs (e.g. permeability, rejection). Classification models are not used as
optimization oracles; skip or omit the `optimization` block when your target
is discrete.

## Quickstart (5 minutes)

```bash
pip install -e .            # core ML stack (train / EDA)
# pip install -e ".[opt]"   # Ax inverse design
# pip install -e ".[shap]"  # SHAP figures
# pip install -e ".[all]"   # opt + shap + imblearn + notebook

# New project? Copy the minimal template and edit two values:
cp config.minimal.yaml my_project/config.yaml
```

Edit `my_project/config.yaml` — only the `data` block is required:

```yaml
data:
  dataset: "my_data.xlsx"      # put your file in data/raw/
  sheet: ""
  targets: ["your_kpi_column"] # your numeric output column(s)
```

Run the whole workflow:

```bash
python -m run_pipeline --config my_project/config.yaml
```

That prints a DATASET CHECK report, then trains a model per target and writes
EDA / parity (or confusion) figures into `outputs/figures/`. SHAP is skipped
if the extra is missing. Inverse design runs when `optimization` is wired to
`data.targets`; if it is wired but Ax is not installed, the pipeline **errors**
(`pip install -e ".[opt]"`). No code edits needed.

## Reusing on a completely different dataset

Feature selection is automatic by default (`features: {mode: "auto"}`): every
usable, non-constant column except `data.targets` is used as a model feature.
Regression vs classification is inferred from the target (or set
`modeling.task: classification` / `regression` explicitly).

**Model architecture** (one key):

```yaml
modeling:
  model_selection: extratrees   # default (tutorial)
  # regression:     extratrees | randomforest | histgradientboosting | ridge | elasticnet | linear
  # classification: extratrees | randomforest | histgradientboosting | logistic | ridge
```

Anything that does not apply to your data is skipped with a warning, so a
minimal config works as-is. Optional blocks you can add later by copying them
from `config.yaml`:

- `feature_engineering.lookups` / `synthetic_feature` — merge descriptor
  tables or add equation-derived columns.
- `optimization` — Bayesian inverse design for **continuous** objectives only.
  Each objective must point at a `data.targets` column via `model_target`,
  `original_name`, or `name`. Skipped if that block is missing or miswired.
  If it is wired and Ax is missing: **error** (`pip install -e ".[opt]"`).
  Not applicable to classification targets.

As a library in your own project:

```python
from run_pipeline import run_pipeline

config = run_pipeline(
    config_path="./my_project/config.yaml",  # paths resolve against this folder
)
```

## Using the tutorial example (membrane dataset)

`config.yaml` is the pre-filled membrane example (manual feature list, ion
property lookups, multiobjective / Pareto inverse design). `data/` must contain the
dataset before running. The companion notebook
`notebooks/Complete_notebook/Ml-MEM-Tut.ipynb` walks through the same pipeline
cell by cell for teaching.

## Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Oikiy/ML-MEM-Review/blob/main/notebooks/Complete_notebook/Ml-MEM-Tut.ipynb)

Full clone + install + examples: see **[`docs/COLAB.md`](docs/COLAB.md)**.  
The tutorial notebook includes a **Colab bootstrap** cell (clone repo, `pip install -e ".[all]"`, inline plots).

## Examples gallery (short recipes)

Self-contained mini-projects under [`examples/`](examples/) — each has its own
`config.yaml` and tiny synthetic data:

| Example | Teaches | Colab |
|---------|---------|-------|
| [`01_minimal_regression`](examples/01_minimal_regression/) | New continuous dataset, auto features | [Open](https://colab.research.google.com/github/Oikiy/ML-MEM-Review/blob/main/examples/01_minimal_regression/Colab.ipynb) |
| [`02_classification`](examples/02_classification/) | Discrete labels, confusion plots | [Open](https://colab.research.google.com/github/Oikiy/ML-MEM-Review/blob/main/examples/02_classification/Colab.ipynb) |
| [`03_switch_models`](examples/03_switch_models/) | `model_selection` (e.g. RandomForest) | [Open](https://colab.research.google.com/github/Oikiy/ML-MEM-Review/blob/main/examples/03_switch_models/Colab.ipynb) |
| [`04_feature_selection`](examples/04_feature_selection/) | Manual `include` vs `auto` + `exclude` | [Open](https://colab.research.google.com/github/Oikiy/ML-MEM-Review/blob/main/examples/04_feature_selection/Colab.ipynb) |
| [`05_inverse_design_controls`](examples/05_inverse_design_controls/) | Ax search space, constants, objectives | [Open](https://colab.research.google.com/github/Oikiy/ML-MEM-Review/blob/main/examples/05_inverse_design_controls/Colab.ipynb) |

```bash
python -m run_pipeline --config examples/01_minimal_regression/config.yaml --force-retrain
```

See [`examples/README.md`](examples/README.md) for the full list and notes.

## Outputs

| Stage    | Output in `outputs/figures/`                       |
| -------- | -------------------------------------------------- |
| EDA      | `target_distributions.png`, `pearson_heatmap.png`  |
| Modeling | `parity_<target>.png` (regression) or `confusion_<target>.png` (classification) |
| SHAP     | `shap_summary_<target>.png`, `shap_importance_<target>.png` |
| Inverse design | `inverse_design_summary.png`, `response_map_*.png` |

Trained models are saved to `models/` (joblib pickles); Ax optimization state
is checkpointed to `models/ax_client.pkl`. All artifacts are git-ignored.

## Development

```bash
pip install -e ".[dev]"     # pytest + ruff; enough for unit tests and smokes
pip install -e ".[all]"     # tutorial / Colab / inverse design + SHAP
python -m pytest
```

| Extra | Provides |
|-------|----------|
| `opt` | Ax, BoTorch, scikit-optimize, Plotly |
| `shap` | SHAP figures |
| `imblearn` | BorderlineSMOTE (`modeling.use_smote`) |
| `notebook` | Jupyter notebook |
| `all` | all of the above |

CI (GitHub Actions) runs `ruff check` and `pytest` on Python 3.11. For a pinned paper install see `requirements.lock.txt`.

## License & citation

MIT. If you use this work in a publication, please cite the associated
membrane-ML review article (citation details in `CITATION.cff`).