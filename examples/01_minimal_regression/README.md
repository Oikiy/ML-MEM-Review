# 01 — Minimal regression

**Goal:** Run the full train path on a *new* continuous dataset by editing only
the data section (features are auto-selected).

## What’s special in `config.yaml`

- `data.dataset` / `data.targets` point at the toy CSV
- `features.mode: auto` — no manual feature list
- No feature-engineering lookups, no optimization block

## Run

From the **repository root**:

```bash
python -m run_pipeline --config examples/01_minimal_regression/config.yaml --force-retrain
```

Model-only smoke:

```bash
python -m run_pipeline --config examples/01_minimal_regression/config.yaml --stage model --force-retrain
```

## Expected outputs

Under `examples/01_minimal_regression/`:

- `models/permeance_extratrees_regressor.pkl`
- `outputs/figures/parity_permeance.png`
- (full run) EDA figures under `outputs/figures/`

## Try next

- [`02_classification`](../02_classification/) — discrete labels  
- [`03_switch_models`](../03_switch_models/) — change `model_selection`
