# Figure X — Section 4 flow diagram (PowerPoint coordinates)

**Files:**
- [`FigX_Section4_ML_Workflow.pptx`](FigX_Section4_ML_Workflow.pptx) — PowerPoint
- [`FigX_Section4_ML_Workflow.drawio`](FigX_Section4_ML_Workflow.drawio) — diagrams.net / draw.io

**Slide size:** 13.333 in × 7.5 in (16:9 widescreen)  
**draw.io page:** 1600 × 900 px (same layout; ~120 px/in)  
**Origin:** top-left of slide = (0, 0); units below = **inches** (PPT) unless noted

## Color tokens

| Token | RGB | Hex | Use |
|-------|-----|-----|-----|
| Blue | 46, 90, 136 | `#2E5A88` | Data band stroke / steps 1–4 |
| Blue fill | 214, 230, 245 | `#D6E6F5` | Step 1–4 fill |
| Purple | 91, 75, 138 | `#5B4B8A` | Model band / steps 5–7 |
| Purple fill | 232, 226, 244 | `#E8E2F4` | Step 5–7 fill |
| Green | 47, 107, 79 | `#2F6B4F` | Experiment band / steps 8–9 |
| Green fill | 217, 238, 227 | `#D9EEE3` | Step 8–9 fill |
| Orange | 196, 92, 38 | `#C45C26` | Feedback + callout border |
| Orange soft | 232, 160, 107 | `#E8A06B` | Feedback arrow fill |
| Gray | 74, 74, 74 | `#4A4A4A` | Subtitles, minor arrows |
| Callout fill | 255, 248, 238 | `#FFF8EE` | Right panel |

## Step boxes

Every step box: **width = 1.85 in**, **height = 1.15 in**, corner round ~0.15.  
Number badge: **oval 0.28 × 0.28 in** at `(box_left − 0.06, box_top − 0.06)`.

| Step | Title | Subtitle (two lines) | left | top | Band |
|------|-------|----------------------|------|-----|------|
| 1 | Objectives | KPIs, constraints / lab limits | 0.50 | 1.15 | Data |
| 2 | Data schema | Ingest recipes → tables | 2.55 | 1.15 | Data |
| 3 | Inspect | Missing, dups, outliers | 4.60 | 1.15 | Data |
| 4 | FE + EDA | Descriptors, correlations | 6.65 | 1.15 | Data |
| 5 | Surrogate | Train / CV / parity | 0.50 | 3.50 | Model |
| 6 | xAI (SHAP) | Process drivers, plausibility | 2.55 | 3.50 | Model |
| 7 | Inverse design | Screen + BO search space | 4.60 | 3.50 | Model |
| 8 | Validate | Fabricate & test KPIs | 0.50 | 5.75 | Experiment |
| 9 | Refine | Update data & retrain | 2.55 | 5.75 | Experiment |

**Horizontal pitch:** centers every **2.05 in** (1.85 box + 0.20 gap).

## Bands (background)

| Band | left | top | width | height |
|------|------|-----|-------|--------|
| Data foundation (1–4) | 0.30 | 0.65 | 8.55 | 2.15 |
| Model & in silico design (5–7) | 0.30 | 3.05 | 8.55 | 2.00 |
| Experiment & closed loop (8–9) | 0.30 | 5.30 | 8.55 | 1.75 |

## Connectors

| Link | Shape | left | top | w | h |
|------|-------|------|-----|---|---|
| 1→2, 2→3, 3→4 | RIGHT_ARROW | see pptx | box_mid_y ≈ 1.65 | 0.18 | 0.14 |
| 5→6, 6→7 | RIGHT_ARROW | | mid_y ≈ 4.00 | 0.18 | 0.14 |
| 8→9 | RIGHT_ARROW | | mid_y ≈ 6.25 | 0.18 | 0.14 |
| Into row 2 | DOWN_ARROW | 1.20 | 2.85 | 0.20 | 0.35 |
| From step 4 | DOWN_ARROW | 7.30 | 2.35 | 0.22 | 0.55 |
| 7→8 region | DOWN_ARROW | 5.35 | 4.75 | 0.22 | 0.40 |
| **Feedback 9→2** | UP_ARROW | **4.55** | **1.40** | **0.28** | **4.90** |

**Feedback label** at left=4.85, top=3.80, w=1.70, h=0.90:  
`FEEDBACK / new recipes, KPIs, failures → schema (2)`

## Right callout

| Element | left | top | width | height |
|---------|------|-----|-------|--------|
| Panel | 9.10 | 0.65 | 3.90 | 5.40 |
| Title | 9.25 | 0.80 | 3.60 | 0.35 |
| Body | 9.25 | 1.25 | 3.60 | 4.50 |

Body content (edit freely):

```
COMPANION CODE (tabular core)
Maps primarily to Steps 4–7:
• config.yaml (targets, features, search_space)
• run_pipeline: EDA → train → SHAP → Ax
• Inputs: recipe–KPI tables (+ lookups)
• Not default: image/multimodal (§3 only)
```

## Fonts

| Element | Font | Size | Style |
|---------|------|------|-------|
| Slide title | Calibri | 16 pt | Bold |
| Band label | Calibri | 10 pt | Bold, band color |
| Step title | Calibri | 11 pt | Bold, band color |
| Step subtitle | Calibri | 9 pt | Regular, gray |
| Callout title | Calibri | 12 pt | Bold, orange |
| Callout body | Calibri | 11 pt | Regular |
| Footer | Calibri | 9 pt | Gray |

## Build tips in PowerPoint

1. Open `FigX_Section4_ML_Workflow.pptx` (slide 1 = art; slide 2 = this table).
2. Replace crude vertical feedback with an **elbow connector**: 9 right → up → left into 2.
3. Optional dashed connectors: 6→4 and 6→7 (Format → Dash type).
4. For journal: **File → Export → PDF** or copy shapes into Illustrator.
5. If the journal wants single-column width, scale uniformly to ~3.5 in width and bump font +1 pt after scale.

## Caption (paste into Word)

*Figure X. Nine-step closed-loop workflow for ML-assisted membrane development. Blue: objectives and tabular data curation (Steps 1–4). Purple: surrogate modeling, explainable AI, and inverse design (Steps 5–7). Green: experimental validation and model updating (Steps 8–9). Orange arrow: feedback of validated and failed experiments into the data schema. Right panel: scope of the companion code (tabular pipeline).*
