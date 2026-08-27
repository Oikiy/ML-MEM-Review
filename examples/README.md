# Examples gallery

Short, self-contained recipes that show how to drive **ML-MEM** from
`config.yaml` only. These recipes correspond to the paper’s **Steps 4–7** (FE/EDA, surrogate, optional SHAP, optional Ax), not lab Steps 8–9. The teaching notebook numbers all nine steps and discusses 1–3 and 8–9.

The full membrane teaching notebook remains the paper
tutorial:

`notebooks/Complete_notebook/Ml-MEM-Tut.ipynb`

Each folder is a mini-project: `config.yaml` + small data under `data/raw/`.
Artifacts (`models/`, `outputs/`) are written **inside that folder** when you
pass its config path.

## Prerequisites

```bash
pip install -e .            # core (examples 01–04)
pip install -e ".[opt]"     # example 05 inverse design
# from the repository root
```

**Google Colab (easiest):** open the **Colab** link in the table below (one
notebook per example). Step-by-step notes: [`docs/COLAB.md`](../docs/COLAB.md).

## Curriculum

| # | Folder | Teaches | Colab |
|---|--------|---------|-------|
| 01 | [`01_minimal_regression/`](01_minimal_regression/) | New continuous dataset, `features.mode: auto` | [Open](https://colab.research.google.com/github/Oikiy/ML-MEM-Review/blob/main/examples/01_minimal_regression/Colab.ipynb) |
| 02 | [`02_classification/`](02_classification/) | Discrete labels, confusion plots, no inverse design | [Open](https://colab.research.google.com/github/Oikiy/ML-MEM-Review/blob/main/examples/02_classification/Colab.ipynb) |
| 03 | [`03_switch_models/`](03_switch_models/) | One-key `model_selection` switch | [Open](https://colab.research.google.com/github/Oikiy/ML-MEM-Review/blob/main/examples/03_switch_models/Colab.ipynb) |
| 04 | [`04_feature_selection/`](04_feature_selection/) | Choose features: `manual` include vs `auto` + exclude | [Open](https://colab.research.google.com/github/Oikiy/ML-MEM-Review/blob/main/examples/04_feature_selection/Colab.ipynb) |
| 05 | [`05_inverse_design_controls/`](05_inverse_design_controls/) | Search space, constants, objectives, scalarized weights | [Open](https://colab.research.google.com/github/Oikiy/ML-MEM-Review/blob/main/examples/05_inverse_design_controls/Colab.ipynb) |

Each folder has a thin **`Colab.ipynb`**: bootstrap → show config → `run_pipeline` → display figures.
Canonical truth remains `config.yaml` + CLI (below).

## Run pattern (all examples)

```bash
python -m run_pipeline --config examples/<folder>/config.yaml --force-retrain
```

Faster smoke (training only):

```bash
python -m run_pipeline --config examples/<folder>/config.yaml --stage model --force-retrain
```

## Notes

- Datasets are **tiny synthetic CSVs** (fast, license-clear). They are not
  scientific results.
- Inverse design (Ax) is for **continuous regression targets only**. Example 02
  intentionally omits `optimization`.
- After a run, check `<folder>/models/` and `<folder>/outputs/figures/`.
