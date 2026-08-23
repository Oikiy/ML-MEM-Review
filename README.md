# ML-MEM-Review

A reproducible ML inverse-design workflow for membrane development
(the companion repository of the membrane ML tutorial).

The same package runs on **any** tabular ML dataset — the membrane
recipe search space, feature lookups, and tutorial notebook are provided as
pre-filled examples, not hard requirements.

**Scope note:** training and evaluation support both **regression** and
**classification**. The Ax **inverse-design** stage is intentionally for
**continuous (regression) targets only** — that matches the membrane tutorial
KPIs (e.g. permeability, rejection). Classification models are not used as
optimization oracles; skip or omit the `optimization` block when your target
is discrete.

## Quickstart (5 minutes)

```bash
pip install -e .            # installs the package + core ML stack

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

That prints a DATASET CHECK report, then trains a model per target, generates
EDA / parity / SHAP figures into `outputs/figures/`, and (if an optimization
block is configured) runs the inverse-design search. No code edits needed.

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
- `optimization` — Bayesian inverse design for **continuous** objectives only
  (must name columns in `data.targets` that you trained as regressors).
  Skipped with a clear message when misconfigured or when Ax is unavailable.
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
property lookups, multiobjective search space). `data/` must contain the
dataset before running. The companion notebook
`notebooks/Complete_notebook/Ml-MEM-Tut.ipynb` walks through the same pipeline
cell by cell for teaching.

## Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Oikiy/ML-MEM-Review/blob/main/notebooks/Complete_notebook/Ml-MEM-Tut.ipynb)

Full clone + install + examples: see **[`docs/COLAB.md`](docs/COLAB.md)**.  
The tutorial notebook includes a **Colab bootstrap** cell (clone repo, `pip install -e .`, inline plots).

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
pip install -e ".[dev]"
python -m pytest            # unit tests: config, features, modeling, Ax helpers
```

## License & citation

MIT. If you use this work in a publication, please cite the associated
membrane-ML review article (citation details in `CITATION.cff`).