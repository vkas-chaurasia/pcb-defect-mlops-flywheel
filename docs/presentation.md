---
marp: true
theme: default
paginate: true
style: |
  @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;600&family=Inter:wght@400;500;600&display=swap');

  section {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background: #F2F2F0;
    color: #1a1a1a;
    padding: 48px 60px;
  }

  /* ── Title slide ── */
  section.title {
    background: #5B9A3E;
    color: #ffffff;
    padding: 0;
    display: grid;
    grid-template-columns: 55% 45%;
  }
  section.title .left {
    padding: 60px 50px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  section.title h1 {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 2.6rem;
    color: #ffffff;
    border: none;
    line-height: 1.25;
    margin: 0 0 0.4em 0;
  }
  section.title h2 {
    font-size: 1rem;
    font-weight: 400;
    color: #d4ecc8;
    border: none;
    margin: 0 0 0.3em 0;
  }
  section.title .names {
    font-size: 0.9rem;
    color: #d4ecc8;
    margin-top: auto;
  }

  /* ── Content slides ── */
  h1 {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 1.75rem;
    font-weight: 600;
    color: #3A7D28;
    border: none;
    border-bottom: 3px solid #3A7D28;
    padding-bottom: 0.25em;
    margin-bottom: 0.6em;
  }
  h2 { font-size: 1.1rem; color: #2C5F1E; margin-bottom: 0.3em; }
  h3 { font-size: 0.95rem; color: #3A7D28; margin-bottom: 0.2em; }

  /* ── Tables ── */
  table { width: 100%; font-size: 0.74rem; border-collapse: collapse; }
  th { background: #3A7D28; color: #fff; padding: 6px 10px; text-align: left; }
  td { padding: 5px 10px; border-bottom: 1px solid #dde8d8; }
  tr:nth-child(even) td { background: #eef5ea; }

  /* ── Code ── */
  code { background: #e8f2e4; color: #2C5F1E; border-radius: 3px; padding: 0.05em 0.35em; font-size: 0.82em; }
  pre  { background: #263326; color: #b8d8b0; border-radius: 6px; padding: 0.9em 1.1em; font-size: 0.7rem; line-height: 1.5; }

  /* ── Problem / Solution boxes ── */
  .box { border: 1.5px solid #3A7D28; border-radius: 6px; padding: 16px 20px; margin: 0; }
  .box-problem { border-color: #D96B2A; }
  .box-label { font-weight: 700; font-size: 0.8rem; letter-spacing: 0.05em; margin-bottom: 8px; }
  .box-label.problem { color: #D96B2A; }
  .box-label.solution { color: #3A7D28; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }

  /* ── Callout / quote ── */
  blockquote { background: #eef5ea; border-left: 4px solid #3A7D28; padding: 10px 16px; margin: 12px 0 0; font-size: 0.82rem; color: #2C5F1E; border-radius: 0 4px 4px 0; }
  blockquote p { margin: 0; }

  /* ── Human gate badge ── */
  .human { display:inline-block; background:#E8A030; color:#fff; border-radius:3px; padding:1px 7px; font-size:0.72rem; font-weight:600; }

  /* ── Final dark slide ── */
  section.dark { background: #263326; color: #ffffff; }
  section.dark h1 { color: #7DC85A; border-bottom-color: #7DC85A; }
  section.dark li { color: #d4ecc8; }
  section.dark strong { color: #ffffff; }

  /* ── Pagination ── */
  section::after { color: #888; font-size: 0.7rem; }
---

<!-- _class: title -->

<div class="left">

# PCB Defect Detection System

## Machine Learning and Data in Operations
## Spring, 2026

<div class="names">Bhatia Isha · Chaurasia Vikas · Duss Karin · Müller Jonathan</div>

</div>

---

# Why Automate PCB Inspection?

<div class="two-col">

<div class="box box-problem">
<div class="box-label problem">THE PROBLEM</div>

- Manual inspection is slow and labor-intensive
- Human error rates reach **20–30%** in high-volume production
- Defects cause costly product recalls and field failures
- No scalable real-time feedback loop for quality control

</div>

<div class="box">
<div class="box-label solution">OUR SOLUTION</div>

- Automated, sub-second visual defect detection
- YOLOv8 replacing human variance
- End-to-end MLOps pipeline for versioning and reproducibility
- Active Learning data flywheel

</div>

</div>

---

# Concept

| | |
|:---|:---|
| **Input** | High-resolution PCB images — from production-line cameras or manual uploads |
| **Core Task** | Detect and classify 6 defect types: `open` · `short` · `mousebite` · `spur` · `spurious_copper` · `pin_hole` |
| **Output** | Defect class, bounding box location, confidence score, and PASS / FAIL decision |

<br>

> **Business value** — Faster inspection, fewer escaped defects, reduced labor cost, and more consistent quality than manual inspection

---

# Tools & Tech Stack

| Stage | Tools | Purpose |
| :--- | :--- | :--- |
| **Data** | DVC · RustFS (S3) | Version datasets, reproducible pulls |
| **Model Development** | YOLOv8 · ONNX · Python | Train & export hardware-agnostic detection model |
| **Experiment Tracking** | MLflow (dual instance) | Sandbox dev runs vs official CI/CD runs |
| **CI / CD** | GitHub Actions · Self-hosted Runner · CML | Automated training on PRs, visual reports |
| **Deployment** | FastAPI · Docker Compose | ONNX inference API with prediction logging |
| **Frontend** | Streamlit | Interactive defect sandbox |
| **Labeling** | Label Studio | Human-in-the-loop annotation |
| **Orchestration** | Airflow | Governance eval · label sync · drift monitoring |
| **Drift Detection** | Evidently AI | Statistical monitoring of production predictions |

---

# End-to-End MLOps Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '11px', 'fontFamily': 'Inter'}}}%%
graph LR

classDef infra fill:#1C2B3A,stroke:#0d1821,color:#fff,font-weight:bold
classDef ml fill:#1A4731,stroke:#0d2419,color:#fff,font-weight:bold
classDef registry fill:#1E6B3C,stroke:#0f3a21,color:#fff,font-weight:bold
classDef decision fill:#C47A1E,stroke:#8B5615,color:#fff,font-weight:bold
classDef human fill:#FFD700,stroke:#8B6914,color:#1a1a1a,font-weight:bold

A["1. Training Pipeline\nGit · DVC · YOLOv8\nGitHub Actions · CML"]:::ml
H_PR["HUMAN APPROVAL\nReview CML report\n& merge PR"]:::human
B["2. MLflow Registry\nTracking · Versioning"]:::registry
C["3. Governance Evaluation\nAirflow model_governance_eval"]:::infra
H_GOV["HUMAN APPROVAL\nGovernance PR\n→ promote @champion"]:::human
E["4. Production Serving\nFastAPI · ONNX · Streamlit"]:::infra
F{"5. Drift Detected?\nAirflow + Evidently AI\n① sync_labels  ② drift_monitor"}:::decision
H_RETRAIN["HUMAN APPROVAL\nReview retraining PR\n& merge"]:::human
G["6. Label Studio\nAnnotation"]:::ml
H_ANNOT["HUMAN\nAnnotate uncertain\ndefect images"]:::human

A -->|train + validate| H_PR
H_PR -->|"merge to main · register @staging"| B
B -->|trigger governance| C
C -->|"eval passed · open PR"| H_GOV
H_GOV -.->|set @champion alias| B
B -.->|hot reload| E
E -->|prediction_log.csv| F
F -->|No — keep serving| E
F -->|"Yes — open retraining PR"| H_RETRAIN
H_RETRAIN -.->|merge triggers CI/CD| A
E -->|"low confidence detection"| G
G --> H_ANNOT
H_ANNOT -.->|"Airflow sync_labels → data/raw"| A
```

---

# Phase 1 — Training & CI/CD

**Triggered by:** Pull Request opened or updated

- GitHub Actions (self-hosted runner) runs the full pipeline:
  `dvc pull` → `dvc repro` → preprocess → YOLOv8 train → evaluate
- Metrics logged to **MLflow Official** (:5556)
- **CML** posts confusion matrix + F1 curves as a PR comment

<span class="human">👤 HUMAN GATE</span> &nbsp;Review CML report → merge to main → `@staging` registered in MLflow

| Mode | Where it logs | When to use |
|:---|:---|:---|
| `uv run python src/training/train.py` | Sandbox MLflow :5555 | Fast local iteration |
| `dvc repro` + open PR | Official MLflow :5556 | Before any merge |

> CI/CD runs **strictly from `params.yaml`** — command-line flags used locally are ignored by the pipeline

---

# Phase 2 — Model Governance

**Triggered by:** push to `main` after PR merge

```
@staging in MLflow
       │
       ▼
Airflow: model_governance_eval DAG
  evaluate_staging_model.py  ──► golden validation set
       │
  ┌────┴────┐
 FAIL      PASS
  │          │
tag:failed  tag:passed → create_governance_pr()
skip                         │
                    GitHub PR: update production_version.json
                             │
```

<span class="human">👤 HUMAN GATE</span> &nbsp;Review Governance PR → merge → `deploy.yml` sets `@champion` alias

> FastAPI polls MLflow every 30 s and hot-reloads when a new `@champion` is detected

---

# Phase 3 — Serving & Drift Monitoring

**FastAPI** serves `@champion` via ONNX · logs every prediction to `monitoring/prediction_log.csv`

**Airflow `data_sync_and_drift_check` DAG** — runs daily:

| Step | What it does |
| :--- | :--- |
| ① `sync_labels` | Pull completed annotations from Label Studio → `data/raw/active_learning/` |
| ② `drift_monitor` | Read last 100 rows of prediction log · run Evidently AI report |

**Features monitored by Evidently AI:**

| Feature | What a shift means |
|:---|:---|
| `avg_confidence` | Model seeing unfamiliar patterns |
| `avg_bbox_area` | Camera setup or PCB layout changed |
| `pass_fail` | Population-level failure rate shifted |

<span class="human">👤 HUMAN GATE</span> &nbsp;If ≥ 50% features drifted → Airflow opens retraining PR → human merges to trigger CI/CD

---

# Phase 4 — Active Learning

**Closing the flywheel:** low-confidence predictions become tomorrow's training data

```
FastAPI: defect detected but low confidence
                 │
                 ▼
         Label Studio (annotation queue)
```

<span class="human">👤 HUMAN GATE</span> &nbsp;Annotate bounding boxes for 6 defect classes in Label Studio

```
Airflow sync_labels DAG (daily)
  ① Download completed annotations via Label Studio API
  ② Convert to YOLO format  (cx, cy, w, h  normalised)
  ③ Write to  data/raw/active_learning/
                 │
                 ▼
  dvc add data/raw  →  dvc push  →  open PR  →  Phase 1
```

> Every uncertain prediction eventually improves the model — the flywheel is self-reinforcing

---

# Human-in-the-Loop Gates

Four checkpoints — **no model reaches production without a human merge**

| Gate | Where | Decision |
| :--- | :--- | :--- |
| **H1 — Annotation** | Label Studio | Draw bounding boxes on low-confidence PCB images |
| **H2 — Training PR** | GitHub PR + CML report | Accept or reject model: review confusion matrix and F1 vs main |
| **H3 — Governance PR** | GitHub Governance PR | Final safety gate before production — merge promotes `@champion` |
| **H4 — Retraining PR** | GitHub Drift PR | Decide whether detected drift warrants a full retrain cycle |

<br>

> No retraining happens without a human decision.
> No champion is set without a human merge.

---

# Demo

**Services running locally via Docker Compose:**

| Service | URL |
|:---|:---|
| Streamlit — interactive defect sandbox | `http://localhost:8501` |
| FastAPI — inference API + Swagger docs | `http://localhost:8000/docs` |
| Airflow — DAG orchestration | `http://localhost:8085` |
| MLflow Official — experiment tracking | `http://localhost:5556` |
| Evidently AI — drift dashboard | `http://localhost:8005` |
| Label Studio — annotation UI | `http://localhost:8080` |
| RustFS S3 — data & model storage | `http://localhost:9001` |

---

<!-- _class: dark -->

# Key Takeaways

- **Flywheel design** — drift → annotation → retrain → better model → less drift
- **Three Airflow DAGs** — governance eval, label sync + drift check, deploy
- **Four human gates** — no model reaches production unchecked
- **ONNX format** — eliminates Mac vs Linux inference discrepancies
- **Dual MLflow** — keeps sandbox experiments separate from official CI/CD runs
- **Self-hosted runner** — bridges GitHub Actions with local Docker infrastructure

<br>

*Bhatia Isha · Chaurasia Vikas · Duss Karin · Müller Jonathan*
*ZHAW — Machine Learning and Data in Operations · Spring 2026*
