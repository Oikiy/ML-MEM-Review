# 03 — Switch model architectures

**Goal:** Keep the same data and change only `modeling.model_selection` to
train a different algorithm (no code edits).

## What’s special in `config.yaml`

- Same style of continuous toy data as example 01
- Default here: `model_selection: randomforest`
- Hyperparameter tuning off for a fast demo

## Run

From the **repository root**:

```bash
python -m run_pipeline --config examples/03_switch_models/config.yaml --stage model --force-retrain
```

## Expected outputs

- `models/permeance_randomforest_regressor.pkl`
- `outputs/figures/parity_permeance.png`

## Try yourself

Edit one line in `config.yaml` and re-run with `--force-retrain`:

```yaml
modeling:
  model_selection: ridge          # or extratrees | histgradientboosting | elasticnet | linear
```

Check that the pickle name and the log line `estimator: ...` match your choice.

### Available keys (sklearn)

| Task | `model_selection` values |
|------|---------------------------|
| regression | `extratrees`, `randomforest`, `histgradientboosting`, `ridge`, `elasticnet`, `linear` |
| classification | `extratrees`, `randomforest`, `histgradientboosting`, `logistic`, `ridge` |
