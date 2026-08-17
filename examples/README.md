# Examples gallery

Short, self-contained recipes that show how to drive **ML-MEM** from
`config.yaml` only. The full membrane teaching notebook remains the paper
tutorial:

`notebooks/Complete_notebook/Ml-MEM-Tut.ipynb`

Each folder is a mini-project: `config.yaml` + small data under `data/raw/`.
Artifacts (`models/`, `outputs/`) are written **inside that folder** when you
pass its config path.

## Prerequisites

```bash
pip install -e .
# from the repository root
```

**Google Colab:** clone the full repo, `cd` to the root, then `pip install -e .`
and run the same `python -m run_pipeline --config examples/...` commands.
Step-by-step: [`docs/COLAB.md`](../docs/COLAB.md).

## Curriculum

| # | Folder | Teaches |
|---|--------|---------|
| 01 | [`01_minimal_regression/`](01_minimal_regression/) | New continuous dataset, `features.mode: auto` |
| 02 | [`02_classification/`](02_classification/) | Discrete labels, confusion plots, no inverse design |
| 03 | [`03_switch_models/`](03_switch_models/) | One-key `model_selection` switch |
| 04 | [`04_feature_selection/`](04_feature_selection/) | Choose features: `manual` include vs `auto` + exclude |
| 05 | [`05_inverse_design_controls/`](05_inverse_design_controls/) | Search space, constants, objectives, scalarized weights |

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
