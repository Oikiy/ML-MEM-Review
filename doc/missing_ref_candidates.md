# Missing `(ref)` candidates — **recent, high-impact journals** (revised)

**Purpose:** Replace incomplete `(ref)` / `[ref]` / tutorial-link placeholders with **recent (≈2018–2025)** papers from **high-impact** venues, better aligned with a 2025–2026 membrane ML review.

**How this list differs from the previous one**
- Prefers *Nature* family, *Science* family, *Nat. Mach. Intell.*, *Nat. Rev. Phys.*, *Nat. Commun.*, *Adv. Mater.* / field-top journals (*J. Membr. Sci.*, *Desalination*, *Environ. Sci. Technol.*, *Water Res.*).
- Keeps **method + membrane application** balance (not only 1970s statistics classics).
- Classics (LASSO 1996, etc.) are demoted to an optional appendix.

**Rules before you insert**
1. Open the DOI and confirm the sentence is supported.
2. Rows marked **VERIFY PRIMARY** must match the *named* study in your prose.
3. Prefer **one strong recent cite** per claim; add a classic only if the editor wants foundations.

---

## A. Core shortlist (recent / high IF) — use these first

| # | Authors (short) | Year | Journal (approx. IF tier) | Title (short) | DOI | Best for which `(ref)` claim |
|---|-----------------|------|---------------------------|---------------|-----|------------------------------|
| **R1** | Butler, Davies, Cartwright, Isayev, Walsh | 2018 | ***Nature*** | Machine learning for molecular and materials science | [10.1038/s41586-018-0337-2](https://doi.org/10.1038/s41586-018-0337-2) | General ML-for-materials framing; fingerprints/descriptors; data-driven discovery culture |
| **R2** | Karniadakis et al. | 2021 | ***Nat. Rev. Phys.*** | Physics-informed machine learning | [10.1038/s42254-021-00314-5](https://doi.org/10.1038/s42254-021-00314-5) | PINN / grey-box; physics constraints in training; hybrid safeguard language |
| **R3** | Raissi, Perdikaris, Karniadakis | 2019 | ***J. Comput. Phys.*** (very highly cited) | Physics-informed neural networks… | [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045) | Technical PINN loss / AD / collocation details (pair with R2) |
| **R4** | Shields et al. (Doyle / Adams) | 2021 | ***Nature*** | Bayesian reaction optimization as a tool for chemical synthesis | [10.1038/s41586-021-03213-y](https://doi.org/10.1038/s41586-021-03213-y) | Bayesian optimization with few experiments; closed-loop optimization culture |
| **R5** | Burger et al. | 2020 | ***Nature*** | A mobile robotic chemist | [10.1038/s41586-020-2442-2](https://doi.org/10.1038/s41586-020-2442-2) | Self-driving / autonomous experimentation outlook (Step 9 / future) |
| **R6** | Rudin | 2019 | ***Nat. Mach. Intell.*** | Stop explaining black box ML models for high stakes decisions… | [10.1038/s42256-019-0048-x](https://doi.org/10.1038/s42256-019-0048-x) | Limits of post-hoc xAI; interpretability vs SHAP-as-truth (Challenges + SHAP wording) |
| **R7** | Lundberg & Lee | 2017 | NeurIPS (canonical SHAP; still standard) | A unified approach to interpreting model predictions | [10.48550/arXiv.1705.07874](https://doi.org/10.48550/arXiv.1705.07874) | SHAP method definition (if not already in your list) |
| **R8** | Angelopoulos & Bates | 2023 | *Found. Trends Mach. Learn.* (widely used) | Conformal prediction: a gentle introduction | [10.1561/2200000101](https://doi.org/10.1561/2200000101) | Uncertainty / reliable prediction sets (Challenges) |
| **R9** | Messeri & Crockett | 2024 | ***Nature*** | Artificial intelligence and illusions of understanding… | [10.1038/s41586-024-07146-0](https://doi.org/10.1038/s41586-024-07146-0) | Overconfidence in AI understanding — softens “learns physics” rhetoric |
| **R10** | Chen & Guestrin | 2016 | ***KDD*** (de facto standard; huge impact) | XGBoost… | [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785) | Tree boosting for tabular fabrication data; “trees on complex data” |
| **R11** | Baltrušaitis, Ahuja, Morency | 2019 | ***IEEE TPAMI*** | Multimodal machine learning: a survey and taxonomy | [10.1109/TPAMI.2018.2798607](https://doi.org/10.1109/TPAMI.2018.2798607) | Multimodal fusion outside membranes (medical/materials analogy) |
| **R12** | de Jonge & Ross | 2011 | ***Nat. Nanotechnol.*** | Electron microscopy of specimens in liquid | [10.1038/nnano.2011.161](https://doi.org/10.1038/nnano.2011.161) | Liquid-phase TEM / in-liquid dynamics imaging (still the field anchor; pair with a 2020+ LP-TEM+ML paper if you find one) |

---

## B. Membrane-focused recent / high field impact

Use these for **membrane-specific** `(ref)` sentences (often better than generic ML cites).

| # | Focus | Suggested venue tier | How to locate / use | Maps to |
|---|--------|----------------------|---------------------|---------|
| **M1** | OSN + SHAP / solvent effects | *Adv. Membranes* / *J. Membr. Sci.* / *Sep. Purif. Technol.* | Prefer **Ignacz / Szekely-type XAI–OSN** papers you already discuss (~2020–2023); e.g. explainable ML for solvent effects in OSN | SHAP on fabrication/solvent; case studies |
| **M2** | Fragment / inverse design TFC PA | ***J. Membr. Sci.*** (2024–2025) | Primary paper for “fragment-based ML… polyamide” (your Fig. 12a narrative) — cite **that** paper, not a substitute | Inverse design TFC; large candidate libraries |
| **M3** | BO / ML for TFC IP recipes | *J. Membr. Sci.* / *Environ. Sci. Technol.* / *Nat. Commun.* | **Gao et al.** (or equivalent) BO–TFC primary — **VERIFY year/DOI against PDF** | Case study inverse design; experiment vs prediction gap |
| **M4** | Polymer gas-separation ML screening | *Nat. Mater.* / *Sci. Adv.* / *J. Membr. Sci.* | Paper behind “~11,000 homopolymers” gas-separation ML (your Fig. 12b) | Virtual screening |
| **M5** | AEM / energy membrane ML | *Adv. Energy Mater.* / *J. Membr. Sci.* / *Energy Environ. Sci.* | Zhang et al. FCNN AEM screening primary | Energy fabrication case |
| **M6** | RO fouling / operation ML | *Water Res.* / *Desalination* / *ES&T* | High-IF fouling prediction reviews or DNN cake/pore-blocking studies (**Dansawad et al. — VERIFY PRIMARY**) | Operation/fouling `(ref)` |
| **M7** | Data scarcity / OMD / FAIR membrane data | *J. Membr. Sci.* / *Nat. Water* / database paper | **Open Membrane Database** primary publication + any 2022–2025 update note | Data infrastructure; OMD sentence |
| **M8** | Transfer learning / hybrid RO modeling | *Desalination* (2023–2025) | e.g. physics-informed / transfer learning RO papers (your Challenges already gesture here) | Transfer learning; scale-up bridge |
| **M9** | Evaluation leakage / OOD materials | *Nat. Commun.* / *ACS Cent. Sci.* / *J. Chem. Inf. Model.* | Recent OOD / reaction generalization papers (you already list some 2025 ACS Cent. Sci. / JCIM items in the biblio) | Benchmarking & leakage Challenges |
| **M10** | Multimodal process–structure (HIPMP etc.) | Dataset + analysis paper | **HIPMP** cite you already use **[121]** — keep/refresh DOI | Multimodal manufacturing + micrographs |

> **Practical tip:** For M1–M6, pull DOIs directly from the PDFs you already cite in Tables 1–3 / Fig. 12 captions. Those are usually higher-impact *and* claim-perfect matches.

---

## C. Claim → recommended recent cite (quick map)

| Manuscript claim (abbreviated) | Prefer |
|--------------------------------|--------|
| Stepwise / regularized linear methods in modern practice | **R1** + one membrane Ridge/LASSO row from your tables (not 1970 ridge unless editor asks) |
| k-NN / instance-based similarity | **R1** (materials ML context) + membrane KNN row from Table 1 |
| Naïve Bayes / categorical risk baselines | Membrane classification paper from your table (bead/defect) + optional **R1** |
| SVR small data; trees on complex data | **R10** + membrane comparison where XGB/RF beat SVR (your table) |
| Mechanistic NF fails multi-component feeds | DSPM-DE critical papers you already have **[89,90,99]** + short recent NF modeling review in *JMS/Desalination* |
| FIB-SEM / 3D morphology ML | Recent FIB-SEM+ML porous-media or membrane tomography paper (2020+ *Nat. Commun.* / *Ultramicroscopy*); methods review as backup |
| GLCM / texture | Prefer **recent SEM texture + ML membrane** paper; Haralick only if needed as origin |
| GNN pore / polymer graphs | Recent polymer/membrane GNN (your table GNN rows) + optional geometric DL survey in *Nat. Rev.* / *IEEE* |
| SEM / SEMPro / AFM–Stokes | **VERIFY PRIMARY** for SEMPro & AFM–ML papers named in text |
| Image-data scarcity | **M7** + Challenges data papers; **R1** for general small-data materials ML |
| LP-TEM video dynamics | **R12** + a 2020+ LP-TEM deep-learning paper if available |
| Multimodal medical/materials | **R11**; membrane multimodal → **M10** + VERIFY dynamic-membrane study |
| GAN + CNN prescribed strength | VERIFY PRIMARY materials/membrane generative design paper (2020+ *Nat. Commun.* / *Adv. Mater.*) |
| CV / tuning / leakage-aware evaluation | **M9** + **R6**/**R9** for overconfident metrics; avoid claiming split “learns physics” |
| RF/XGB tabular; DNN images; RNN time | **R10** + membrane time-series fouling LSTM/ANN reviews in *Water Res./Desalination* |
| BO inverse design / few experiments | **R4**; membrane BO → **M3** |
| Process-control setpoints | **M6** / operation ML in *Water Res.* |
| Prediction ≠ experiment after inverse design | **M3** experimental validation section + **R9** |
| Physics-informed hybrid constraints | **R2** + **R3** |
| Dansawad fouling DNN | **VERIFY PRIMARY** (M6) |
| Self-driving / closed-loop future | **R5** + **R4** |
| Tutorial link `[_____(link)]` | GitHub + Zenodo DOI (not a journal article) |

---

## D. Copy-paste block (recent / high IF only)

```
[R1] K. T. Butler et al., “Machine learning for molecular and materials science,”
     Nature 559, 547–555 (2018). doi: 10.1038/s41586-018-0337-2

[R2] G. E. Karniadakis et al., “Physics-informed machine learning,”
     Nat. Rev. Phys. (2021). doi: 10.1038/s42254-021-00314-5

[R3] M. Raissi, P. Perdikaris, G. E. Karniadakis, “Physics-informed neural networks…,”
     J. Comput. Phys. (2019). doi: 10.1016/j.jcp.2018.10.045

[R4] B. J. Shields et al., “Bayesian reaction optimization as a tool for chemical synthesis,”
     Nature 590, 89–96 (2021). doi: 10.1038/s41586-021-03213-y

[R5] B. Burger et al., “A mobile robotic chemist,”
     Nature 583, 237–241 (2020). doi: 10.1038/s41586-020-2442-2

[R6] C. Rudin, “Stop explaining black box machine learning models for high stakes decisions…,”
     Nat. Mach. Intell. 1, 206–215 (2019). doi: 10.1038/s42256-019-0048-x

[R7] S. M. Lundberg, S.-I. Lee, “A unified approach to interpreting model predictions,”
     NeurIPS (2017). doi: 10.48550/arXiv.1705.07874

[R8] A. N. Angelopoulos, S. Bates, “Conformal Prediction: A Gentle Introduction,”
     Found. Trends Mach. Learn. (2023). doi: 10.1561/2200000101

[R9] L. Messeri, M. J. Crockett, “Artificial intelligence and illusions of understanding
     in scientific research,” Nature (2024). doi: 10.1038/s41586-024-07146-0

[R10] T. Chen, C. Guestrin, “XGBoost: A Scalable Tree Boosting System,”
      KDD (2016). doi: 10.1145/2939672.2939785

[R11] T. Baltrušaitis, C. Ahuja, L.-P. Morency, “Multimodal Machine Learning: A Survey
      and Taxonomy,” IEEE TPAMI (2019). doi: 10.1109/TPAMI.2018.2798607

[R12] N. de Jonge, F. M. Ross, “Electron microscopy of specimens in liquid,”
      Nat. Nanotechnol. (2011). doi: 10.1038/nnano.2011.161
```

**Plus membrane primaries (fill DOIs from your PDFs / Fig. 12 captions):** Gao BO–TFC; fragment-based PA inverse design (*J. Membr. Sci.*); Ignacz/Szekely XAI–OSN; Zhang AEM FCNN; Dansawad fouling DNN; OMD database paper; HIPMP dataset paper.

---

## E. Optional classics (only if a reviewer demands foundations)

| Topic | Classic (use sparingly) |
|-------|-------------------------|
| LASSO | Tibshirani 1996, doi: 10.1111/j.2517-6161.1996.tb02080.x |
| Ridge | Hoerl & Kennard 1970, doi: 10.1080/00401706.1970.10488634 |
| Random forests | Breiman 2001, doi: 10.1023/A:1010933404324 |
| CV survey | Arlot & Celisse 2010, doi: 10.1214/09-SS054 |
| GLCM origin | Haralick 1973, doi: 10.1109/TSMC.1973.4309314 |

For a **high-IF review**, prefer **R1–R12 + M1–M10** over this appendix.

---

## F. Suggested insertion strategy (efficient)

1. **Method sentences** (PINN, BO, xAI limits, multimodal survey) → **R2–R6, R8, R11**.  
2. **Materials ML culture / descriptors** → **R1**.  
3. **Tabular SOTA trees** → **R10** + your membrane XGB/RF rows.  
4. **Every membrane-specific mechanism/case `(ref)`** → **M1–M6 / VERIFY PRIMARY** (highest scientific accuracy).  
5. **Tutorial link** → GitHub + Zenodo (code availability statement).  
6. Delete remaining bare `(ref)` after a sweep.

---

*Revised for high-impact, recent literature. Always verify DOI landing pages and claim fit before submission.*
