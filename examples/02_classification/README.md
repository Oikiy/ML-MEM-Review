# 02 — Classification

**Goal:** Train a classifier on discrete labels (string classes) and get
confusion-matrix figures instead of parity plots.

## What’s special in `config.yaml`

- Target `label` is categorical (`pass` / `fail`)
- `modeling.task: classification` (optional; auto would also work here)
- **No `optimization` block** — Ax inverse design is continuous/regression only

## Run

From the **repository root**:

```bash
python -m run_pipeline --config examples/02_classification/config.yaml --force-retrain
```

Model-only:

```bash
python -m run_pipeline --config examples/02_classification/config.yaml --stage model --force-retrain
```

## Expected outputs

Under `examples/02_classification/`:

- `models/label_extratrees_classifier.pkl`
- `outputs/figures/confusion_label.png`

## Try next

- [`03_switch_models`](../03_switch_models/) — try `model_selection: randomforest` on regression data  
- Change `model_selection` here to `logistic` or `randomforest` and re-run
