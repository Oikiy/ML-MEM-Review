# Run ML-MEM on Google Colab

This guide covers the **tutorial notebook** and the **examples gallery** after
cloning the full repository.

**Requirements**

- Python **3.11+** Colab runtime (check with `!python -V`)
- Network access to install PyTorch / Ax / BoTorch (first install is slow)
- Prefer a longer session for inverse design (example 05 and notebook §6)

Open-in-Colab badge (replace branch if needed):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Oikiy/ML-MEM-Review/blob/main/notebooks/Complete_notebook/Ml-MEM-Tut.ipynb)

---

## A. Tutorial notebook (recommended path)

1. Open the notebook via the badge above, **or**  
   Colab → *GitHub* → `Oikiy/ML-MEM-Review` →  
   `notebooks/Complete_notebook/Ml-MEM-Tut.ipynb`.
2. Runtime → **Change runtime type** → Python 3.11+ (GPU optional).
3. Run the **Google Colab bootstrap** cell near the top (Section 0).  
   It will:
   - detect Colab
   - clone the repo if you only opened the `.ipynb` from GitHub
   - `cd` to the repo root
   - `pip install -e .`
   - enable `%matplotlib inline`
4. Run the remaining cells top to bottom.

If bootstrap already ran and the kernel restarted, run the bootstrap cell again
(it reuses an existing clone under `/content/ML-MEM-Review`).

**Ax / inverse design:** Sections 1–5 work without Ax. Section 6 needs a
successful `pip install -e .` (pulls `ax-platform` + `botorch` + `torch`).
If Ax import fails, re-run bootstrap or:

```python
!pip install -e .
```

---

## B. Examples gallery (shell cells)

In a **fresh Colab notebook** (or after bootstrap so the repo exists):

```python
import os
os.chdir("/content/ML-MEM-Review")  # after clone; adjust if needed
```

```bash
pip install -e .
```

```bash
# Fast smoke (train only)
python -m run_pipeline --config examples/01_minimal_regression/config.yaml --stage model --force-retrain
python -m run_pipeline --config examples/02_classification/config.yaml --stage model --force-retrain
python -m run_pipeline --config examples/03_switch_models/config.yaml --stage model --force-retrain
python -m run_pipeline --config examples/04_feature_selection/config.yaml --stage model --force-retrain
```

```bash
# Full inverse-design demo (slower; needs Ax)
python -m run_pipeline --config examples/05_inverse_design_controls/config.yaml --force-retrain
```

Artifacts land under each example folder:

- `examples/<name>/models/`
- `examples/<name>/outputs/figures/`

---

## C. Clone-only recipe (no “Open notebook from GitHub”)

```python
REPO_URL = "https://github.com/Oikiy/ML-MEM-Review.git"
!git clone --depth 1 $REPO_URL
%cd ML-MEM-Review
!pip install -e .
```

Then either open `notebooks/Complete_notebook/Ml-MEM-Tut.ipynb` from the
file browser, or run example commands as in section B.

Fork users: set `REPO_URL` to your fork before cloning.

---

## D. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Could not find config.yaml` | Bootstrap did not `cd` to repo root; re-run Colab cell or `%cd /content/ML-MEM-Review` |
| `ModuleNotFoundError: utl_*` | `pip install -e .` from repo root, or ensure `src/` is on `sys.path` (notebook setup does this) |
| `requires-python >=3.11` / install fails | Runtime → Change runtime type → pick 3.11+ |
| Ax import skipped | Reinstall: `!pip install -e .` ; restart runtime; re-run bootstrap |
| Install times out | Re-run `pip install -e .`; avoid interrupting torch download |
| Free tier disconnect mid-BO | Lower `optimization.n_trials` in config; re-run with `--force-retrain` |
| Example 04/05 missing | Pull latest `main` (those folders must be on the remote) |
| Figures not shown | Ensure bootstrap ran (`%matplotlib inline`) |

---

## E. What is / isn’t Colab-specific

- **Portable:** configs, tabular data, `run_pipeline`, pathlib paths, examples 01–05.
- **Colab-only convenience:** bootstrap clone + install cell, `%matplotlib inline`.
- **Not required:** Google Drive mount (everything stays under `/content` unless you choose otherwise).
