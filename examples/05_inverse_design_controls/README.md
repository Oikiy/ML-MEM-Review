# 05 — Control the inverse-design grid and objectives

**Goal:** Drive Bayesian inverse design from `config.yaml` only — choose which
knobs Ax may vary, what stays fixed, and how competing KPIs are scored.

Requires **continuous (regression) targets**. Classification problems should
omit the `optimization` block (see example 02).

## What’s in the toy CSV

Process inputs: `temperature_C`, `pressure_bar`, `additive_wt_pct`, `monomer`,
`polymerization_time_min`, `solvent`  
Targets: `permeance`, `rejection` (synthetic trade-off surface)

## Three things you control

### 1. Search space (the “parameter grid”)

```yaml
optimization:
  search_space:
    - name: "temperature_C"
      type: "range"
      bounds: [30.0, 70.0]      # continuous interval Ax samples
    - name: "monomer"
      type: "choice"
      values: ["PIP", "PEI"]   # categorical options
      is_ordered: false
```

- **`range`** → continuous lab settings (mass, time, T, P, …)  
- **`choice`** → discrete options (monomer name, solvent brand, …)  
- Bounds should be **physically realizable** and preferably inside the training
  coverage (extrapolation is unreliable).

Names in `search_space` must match **feature column names** the surrogate was
trained on (after feature engineering).

### 2. Constants (held fixed)

```yaml
  constant_parameters:
    solvent: "hexane"
```

Use this when a column is a model input but you do **not** want Ax to change it
(fixed solvent, fixed solute, zeroed unused mass slots, …). Constants are merged
with each trial before oracle prediction.

### 3. Objectives and scoring mode

```yaml
  mode: "scalarized"   # or "multiobjective"
  objectives:
    - name: "permeance"
      original_name: "permeance"   # must match data.targets (or map via model_target)
      model_target: "permeance"
      direction: "maximize"
      normalization_min: 5.0
      normalization_max: 55.0
  scalarized_objective:
    equation: "0.3 * opt_permeance + 0.7 * opt_rejection"
```

| Knob | Effect |
|------|--------|
| `mode: scalarized` | One weighted score (`scalarized_objective.equation`) |
| `mode: multiobjective` | Pareto search; optional `threshold` per objective |
| `direction` | `maximize` or `minimize` |
| `normalization_*` | Maps raw KPI into ~[0, 1] before weighting / thresholds |
| `sobol_init` / `n_trials` | Warm-start size and total BO budget |

**Weight tip:** raise the coefficient on the KPI you care about more
(`0.7 * opt_rejection` vs `0.3 * opt_permeance` in the default config).

## Run

From the **repository root**:

```bash
python -m run_pipeline --config examples/05_inverse_design_controls/config.yaml --force-retrain
```

This trains one model per target, then runs Ax. If Ax is missing, training still
succeeds and inverse design is skipped with a clear message
(`pip install -e .` installs `ax-platform`).

Re-run inverse design only after models exist:

```bash
python -m run_pipeline --config examples/05_inverse_design_controls/config.yaml --stage optimize --force-retrain
```

## Expected outputs

Under `examples/05_inverse_design_controls/`:

- `models/permeance_extratrees_regressor.pkl`
- `models/rejection_extratrees_regressor.pkl`
- `models/ax_client.pkl` (checkpoint)
- `outputs/figures/inverse_design_summary.png`
- `outputs/figures/response_map_*.png`

## Suggested experiments

1. **Tighten the grid** — shrink `temperature_C` bounds; watch recommendations.  
2. **Reweight** — swap to `0.8 * opt_permeance + 0.2 * opt_rejection`.  
3. **Pareto mode** — set `mode: multiobjective` and add `threshold: ">=0.2"` on
   each objective; remove reliance on `scalarized_objective`.  
4. **Free another knob** — move `solvent` from `constant_parameters` into
   `search_space` as a `choice`.

## Try next

- Full membrane tutorial: `config.yaml` + `notebooks/Complete_notebook/Ml-MEM-Tut.ipynb`  
- [`04_feature_selection`](../04_feature_selection/) — clean features before optimizing  
