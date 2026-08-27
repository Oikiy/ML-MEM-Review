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
   - **`git fetch` + `reset --hard` to `origin/main`** if a clone already exists
     (so the notebook and `/content/ML-MEM-Review` stay on the same commit)
   - `cd` to the repo root
   - `pip install -e ".[all]"` when the clone SHA changed
   - enable `%matplotlib inline`
4. Run the remaining cells top to bottom.

Re-run bootstrap after every GitHub push (or kernel restart). It updates the
clone under `/content/ML-MEM-Review` and reinstalls only when `HEAD` changed.
That `reset --hard` **discards local edits** in the clone; set
`ML_MEM_SKIP_PULL=1` to keep them, or `ML_MEM_FORCE_INSTALL=1` to reinstall.

**Ax / inverse design:** Sections 1–5 work with the core install. Section 6 needs
`pip install -e ".[all]"` (or at least `".[opt]"`) for `ax-platform` + `botorch`
+ `torch`. If Ax import fails, re-run bootstrap or:

```python
!pip install -e ".[all]"
```

---

## B. Examples gallery (separate Colab notebooks)

Each example has a thin **`Colab.ipynb`** (bootstrap → show config → run pipeline → show figures):

| Example | Open in Colab |
|---------|----------------|
| 01 minimal regression | [Colab](https://colab.research.google.com/github/Oikiy/ML-MEM-Review/blob/main/examples/01_minimal_regression/Colab.ipynb) |
| 02 classification | [Colab](https://colab.research.google.com/github/Oikiy/ML-MEM-Review/blob/main/examples/02_classification/Colab.ipynb) |
| 03 switch models | [Colab](https://colab.research.google.com/github/Oikiy/ML-MEM-Review/blob/main/examples/03_switch_models/Colab.ipynb) |
| 04 feature selection | [Colab](https://colab.research.google.com/github/Oikiy/ML-MEM-Review/blob/main/examples/04_feature_selection/Colab.ipynb) |
| 05 inverse-design controls | [Colab](https://colab.research.google.com/github/Oikiy/ML-MEM-Review/blob/main/examples/05_inverse_design_controls/Colab.ipynb) |

Or from a shell after clone:

```bash
cd /content/ML-MEM-Review   # after bootstrap/clone
pip install -e ".[all]"     # or "." for examples 01–04 only

python -m run_pipeline --config examples/01_minimal_regression/config.yaml --force-retrain
# … same pattern for 02–05
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
!pip install -e ".[all]"
```

Then either open `notebooks/Complete_notebook/Ml-MEM-Tut.ipynb` from the
file browser, or run example commands as in section B.

Fork users: set `REPO_URL` to your fork before cloning.

---

## D. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `run_pipeline exited 2` / argparse | `--stage` is one token (`eda` or `eda,model`). Example notebooks call `--stage eda` then `--stage model`. Re-run **bootstrap** so the clone matches this notebook. |
| `CalledProcessError` / `run_pipeline exited 1` | Scroll **above** that traceback for the pipeline error. Common: `No module named run_pipeline` (re-run bootstrap; `PYTHONPATH=src` is set in example run cells). SHAP is **skipped** if import/torch fails. If inverse design is in the YAML and Ax is missing, the pipeline **errors** (`pip install -e ".[opt]"`). Runtime must be Python **3.11+** (`!python -V`). |
| Notebook newer than clone / stale `/content` | Re-run bootstrap (it `git fetch` + `reset --hard` to `origin/main`). If the **notebook file** itself is old, Runtime → Disconnect and delete runtime, then open from GitHub again. |
| `Could not find config.yaml` | Bootstrap did not `cd` to repo root; re-run Colab cell or `%cd /content/ML-MEM-Review` |
| `ModuleNotFoundError: utl_*` | `pip install -e .` from repo root, or ensure `src/` is on `sys.path` (notebook setup does this) |
| `requires-python >=3.11` / install fails | Runtime → Change runtime type → pick 3.11+ |
| Ax import skipped | Reinstall: `!pip install -e ".[opt]"` ; restart runtime; re-run bootstrap |
| Install times out | Re-run `pip install -e ".[all]"`; avoid interrupting torch download |
| Free tier disconnect mid-BO | Lower `optimization.n_trials` in config; re-run with `--force-retrain` |
| Example 04/05 missing | Re-run bootstrap (clone must be current `main`) |
| Figures not shown | Ensure bootstrap ran (`%matplotlib inline`) |

Environment overrides (optional, set before the bootstrap cell):

| Variable | Effect |
|----------|--------|
| `ML_MEM_REPO_URL` | Fork URL instead of `Oikiy/ML-MEM-Review` |
| `ML_MEM_REPO_REF` | Branch to clone/fetch (default `main`) |
| `ML_MEM_COLAB_ROOT` | Clone path (default `/content/ML-MEM-Review`) |
| `ML_MEM_FORCE_INSTALL=1` | Re-run `pip install -e …` even if SHA unchanged |
| Example 05 / tutorial | Bootstrap extras are `opt` / `all`. After examples 01–04, re-running bootstrap **reinstalls Ax**. If inverse design is configured and Ax is missing, the pipeline **errors** instead of skipping. |
| `ML_MEM_SKIP_PULL=1` | Do not `git reset` an existing clone |

---

## E. What is / isn’t Colab-specific

- **Portable:** configs, tabular data, `run_pipeline`, pathlib paths, examples 01–05.
- **Colab-only convenience:** bootstrap clone + install cell, `%matplotlib inline`.
- **Not required:** Google Drive mount (everything stays under `/content` unless you choose otherwise).
