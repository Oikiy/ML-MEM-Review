# 04 — Select the feature set for modelling

**Goal:** Choose which columns become model inputs — and keep IDs, pure noise,
and free-text notes out of the surrogate.

## What’s in the toy CSV

| Column | Role |
|--------|------|
| `temperature_C`, `pressure_bar`, `additive_wt_pct`, `catalyst` | Real process drivers (use these) |
| `batch_id` | Identifier — **do not** model |
| `noise_uniform` | Random noise — **do not** model |
| `lab_note`, `unused_flag` | Metadata / unused flags — **do not** model |
| `permeance` | Target KPI |

## Two config patterns

### A. Manual allow-list (default in `config.yaml`)

```yaml
features:
  mode: "manual"
  include:
    - "temperature_C"
    - "pressure_bar"
    - "additive_wt_pct"
    - "catalyst"
```

Use this when you know the physical recipe variables and want a stable,
reproducible feature set (typical for membrane fabrication tables).

### B. Auto + exclude list

Uncomment Pattern B in `config.yaml`:

```yaml
features:
  mode: "auto"
  exclude:
    - "batch_id"
    - "noise_uniform"
    - "lab_note"
    - "unused_flag"
```

`auto` keeps every usable non-target column that is not constant / mostly
missing, then drops whatever you list under `exclude`. Good for a first pass
on a new spreadsheet; switch to `manual` once the feature set is settled.

## How to decide

1. **Can I set this in the lab or process?** → candidate feature  
2. **Is it only an ID, timestamp, or comment?** → exclude  
3. **Is it almost constant or >30% missing?** → auto already drops these  
4. After training, check **SHAP**: if a “feature” dominates for no physical
   reason, it is often leakage or a distractor — remove it and retrain.

## Run

From the **repository root**:

```bash
python -m run_pipeline --config examples/04_feature_selection/config.yaml --force-retrain
```

Model-only smoke:

```bash
python -m run_pipeline --config examples/04_feature_selection/config.yaml --stage model --force-retrain
```

On startup the pipeline prints a **DATASET CHECK** report listing
`features_used`. Confirm it matches your allow-list (manual) or your
auto−exclude result.

## Expected outputs

Under `examples/04_feature_selection/`:

- `models/permeance_extratrees_regressor.pkl`
- `outputs/figures/parity_permeance.png`
- (full run) SHAP figures for the selected features only

## Try next

- Flip to Pattern B and compare SHAP / parity  
- [`05_inverse_design_controls`](../05_inverse_design_controls/) — search space & objectives  
- Tutorial notebook: `notebooks/Complete_notebook/Ml-MEM-Tut.ipynb`
