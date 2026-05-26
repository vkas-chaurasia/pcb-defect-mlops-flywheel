---
marp: true
theme: default
paginate: true
style: |
  :root {
    --color-navy:  #1C2B3A;
    --color-green: #1A4731;
    --color-reg:   #1E6B3C;
    --color-gold:  #FFD700;
    --color-orange:#C47A1E;
  }
  section {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background: #ffffff;
    color: #1a1a1a;
  }
  section.title {
    background: #1C2B3A;
    color: #ffffff;
    text-align: center;
    justify-content: center;
  }
  section.title h1 { font-size: 2.4rem; color: #ffffff; margin-bottom: 0.3em; }
  section.title p  { color: #a0b4c8; font-size: 1rem; }
  section.dark {
    background: #1C2B3A;
    color: #ffffff;
  }
  section.dark h2 { color: #FFD700; }
  h1 { color: #1C2B3A; font-size: 1.9rem; border-bottom: 3px solid #1E6B3C; padding-bottom: 0.2em; }
  h2 { color: #1C2B3A; font-size: 1.5rem; }
  table { width: 100%; font-size: 0.78rem; }
  th { background: #1C2B3A; color: #fff; }
  tr:nth-child(even) { background: #f4f7f4; }
  code { background: #f0f4f0; border-radius: 4px; padding: 0.1em 0.4em; font-size: 0.85em; }
  pre  { background: #1C2B3A; color: #a8d8a8; border-radius: 8px; padding: 1em; font-size: 0.78rem; }
  .badge { display:inline-block; background:#1E6B3C; color:#fff; border-radius:4px; padding:2px 8px; font-size:0.75rem; margin-right:4px; }
  .human { display:inline-block; background:#FFD700; color:#1a1a1a; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:bold; }
---

<!-- _class: title -->

# PCB Defect Detection
## The MLOps Flywheel

Project in Machine Learning Operations
ZHAW · Spring 2026

---

# Project Goal

**Automate PCB quality control** using a self-reinforcing MLOps pipeline.

- Detect **6 defect types** on printed circuit boards:
  `open` · `short` · `mousebite` · `spur` · `spurious_copper` · `pin_hole`
- Build a **flywheel** — every production run generates data that improves the next model
- Full **human-in-the-loop** governance: no model reaches production without human approval

### Why this matters
> Manual PCB inspection is slow, expensive, and error-prone.
> An automated, self-improving system reduces defect escape rates over time.

---

# System Architecture

```
Training Pipeline ──► MLflow Registry ──► Governance Eval ──► Human Approval
      ▲                      │                                       │
      │                      └──────────► FastAPI (ONNX) ◄──────────┘
      │                                       │
      │                              prediction_log.csv
      │                                       │
      │                            Airflow + Evidently AI
      │                           /                      \
      │                    No drift                  Drift detected
      │                  keep serving            open retraining PR ──► Human
      │
      └──── Airflow sync_labels ◄── Human annotates ◄── Label Studio
                                     (low-confidence detections)
```

> Gold gates = human decisions · Dashed = automated flywheel loops

---

# Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Code & CI/CD** | Git · GitHub · GitHub Actions + Self-hosted Runner | Version control, automated training on PRs |
| **Detection** | YOLOv8 (ONNX export) | Object detection — 6 defect classes |
| **Data Versioning** | DVC + RustFS (S3) | Reproducible datasets, local S3-compatible storage |
| **Experiment Tracking** | MLflow (dual instance) | Sandbox :5555 for dev · Official :5556 for CI/CD |
| **Workflow Orchestration** | Airflow | Governance eval · label sync · drift monitoring |
| **Drift Detection** | Evidently AI | Statistical drift on confidence, bbox area, pass/fail |
| **Annotation** | Label Studio | Human labelling of low-confidence images |
| **Inference API** | FastAPI | ONNX serving · prediction logging · hot reload |
| **Frontend** | Streamlit | Interactive defect sandbox |
| **Infrastructure** | Docker Compose | Unified local infrastructure management |
| **PR Reporting** | CML | Confusion matrix + F1 curves posted on every PR |

---

# Phase 1 — Training Pipeline

**Triggered by:** opening or updating a Pull Request

```
Developer opens PR
       │
       ▼
GitHub Actions (self-hosted runner)
   dvc pull  ──► data/raw from RustFS S3
   dvc repro ──► preprocess ──► YOLOv8 train ──► evaluate
       │
       ▼
MLflow Official (:5556) — log metrics, confusion matrix, F1
       │
       ▼
CML posts visual report as PR comment
       │
       ▼
 👤 HUMAN reviews CML report ──► merge to main ──► register @staging in MLflow
```

- **Fast iteration**: `uv run python src/training/train.py` → logs to Sandbox MLflow (:5555)
- **CI/CD runs strictly from `params.yaml`** — command-line flags are ignored by the pipeline
- ONNX export happens automatically on merge to `main`

---

# Phase 2 — Model Governance

**Triggered by:** CI/CD push to `main` (after merge)

```
@staging registered in MLflow
         │
         ▼
  Airflow DAG: model_governance_eval
         │
    evaluate_staging_model()
    runs src/utils/evaluate_staging_model.py
    against golden validation set
         │
    ┌────┴────┐
  FAIL       PASS
    │           │
  tag:failed  tag:passed
  skip        create_governance_pr()
                │
                ▼
       GitHub PR: update deployment-repo/production_version.json
                │
                ▼
       👤 HUMAN approves and merges Governance PR
                │
                ▼
       deploy.yml ──► set @champion alias in MLflow
                │
                ▼
       FastAPI hot-reloads @champion within 30 s
```

---

# Phase 3 — Production Serving & Drift Monitoring

**FastAPI** loads `@champion` from MLflow · polls every 30 s for new version

```
POST /predict  ──►  YOLOv8 ONNX inference
                         │
                    prediction_logger.log()
                         │
                   monitoring/prediction_log.csv
```

**Airflow `data_sync_and_drift_check` DAG** — runs daily:

| Step | What it does |
| :--- | :--- |
| ① `sync_labels` | Pull completed annotations from Label Studio → `data/raw/active_learning/` |
| ② `drift_monitor` | Read last 100 rows of `prediction_log.csv` · run Evidently AI |

**Drift features monitored:**
- `avg_confidence` — model certainty dropping signals out-of-distribution input
- `avg_bbox_area` — shift in defect scale (camera/PCB layout change)
- `pass_fail` — population-level failure rate shift

**If ≥ 50% features drifted:** Airflow opens a retraining PR via GitHub API

---

# Phase 4 — Active Learning

**Closing the flywheel:** uncertain predictions become tomorrow's training data

```
FastAPI inference
       │
  defect detected
  but low confidence
       │
       ▼
 Label Studio ──► 👤 HUMAN annotates bounding boxes
                         │
                         ▼
              Airflow sync_labels DAG (daily)
              Downloads from Label Studio API
              Converts to YOLO format (cx, cy, w, h)
              Writes to data/raw/active_learning/
                         │
                         ▼
              Developer: dvc add data/raw  ──► dvc push
                         │
                         ▼
              Open Pull Request ──► loops back to Phase 1
```

> Airflow also opens a retraining PR automatically when drift is detected —
> the human only needs to review and merge it.

---

# CI/CD Workflow

**Self-hosted runner** is required — it runs on the same machine as Docker, giving the pipeline direct access to MLflow, RustFS, and Airflow.

```
Pull Request opened / updated
       │
       ▼
ci.yml  ──►  dvc pull (from RustFS)
         ──►  dvc repro (preprocess + YOLOv8 + eval)
         ──►  log to MLflow Official (:5556)
         ──►  CML: post confusion matrix + F1 curves to PR

Merge to main
       │
       ▼
ci.yml  ──►  export best.pt → best.onnx
         ──►  register @staging in MLflow Model Registry
         ──►  curl Airflow to trigger model_governance_eval DAG

Governance PR merged
       │
       ▼
deploy.yml  ──►  set @champion alias in MLflow Registry
```

---

# Human-in-the-Loop Gates

Four checkpoints where a human must approve before the system advances:

| Gate | Where | Decision |
| :--- | :--- | :--- |
| **H1 — Annotation** | Label Studio | Draw bounding boxes on low-confidence PCB images |
| **H2 — Training PR** | GitHub PR + CML report | Accept or reject model by reviewing confusion matrix and F1 vs main |
| **H3 — Governance PR** | GitHub Governance PR | Final safety gate before model reaches production — approving promotes `@champion` |
| **H4 — Retraining PR** | GitHub Drift PR | Decide whether detected drift warrants a retrain cycle |

> No model reaches production without a human merge.
> No retraining happens without a human decision.

---

<!-- _class: title -->

# Demo

**Streamlit UI** · `http://localhost:8501`
Upload a PCB image → get defect bounding boxes + PASS / FAIL

**FastAPI Docs** · `http://localhost:8000/docs`

**Airflow** · `http://localhost:8085`

**MLflow Official** · `http://localhost:5556`

**Evidently AI** · `http://localhost:8005`

**Label Studio** · `http://localhost:8080`

---

<!-- _class: dark -->

# Key Takeaways

- **Flywheel design**: drift → annotation → retrain → better model → less drift
- **Three Airflow DAGs** orchestrate all automation: governance, sync+drift, deploy
- **Four human gates** ensure no model reaches production unchecked
- **ONNX format** eliminates Mac vs Linux inference discrepancies
- **Dual MLflow** keeps sandbox experiments separate from official CI/CD runs
- **Self-hosted runner** bridges GitHub Actions with local Docker infrastructure

---
*PCB Defect Detection MLOps Flywheel · ZHAW Spring 2026*
