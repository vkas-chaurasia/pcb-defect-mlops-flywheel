---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    font-size: 1.25rem;
    background: #F2F2F0;
    color: #1a1a1a;
    padding: 40px 56px;
  }
  section.title {
    background: #5B9A3E;
    color: #ffffff;
    text-align: center;
    justify-content: center;
    padding: 60px;
  }
  section.title h1 { font-size: 2.2rem; color: #ffffff; border: none; margin-bottom: 0.3em; }
  section.title h2 { font-size: 1rem; color: #d4ecc8; border: none; font-weight: 400; }
  section.title p  { color: #d4ecc8; font-size: 0.95rem; }
  section.dark { background: #263326; color: #ffffff; }
  section.dark h1 { color: #7DC85A; border-bottom-color: #7DC85A; }
  section.dark li { color: #d4ecc8; }
  h1 { color: #3A7D28; font-size: 2rem; border-bottom: 3px solid #3A7D28; padding-bottom: 0.15em; margin-bottom: 0.4em; margin-top: 0; }
  h2 { color: #2C5F1E; font-size: 1.4rem; margin: 0.3em 0; }
  h3 { color: #3A7D28; font-size: 1.15rem; margin: 0.3em 0; }
  p  { margin: 0.35em 0; }
  ul, ol { margin: 0.35em 0; padding-left: 1.5em; }
  li { margin: 0.3em 0; }
  table { width: 100%; font-size: 0.88rem; border-collapse: collapse; }
  th { background: #3A7D28; color: #fff; padding: 7px 12px; text-align: left; }
  td { padding: 6px 12px; border-bottom: 1px solid #dde8d8; }
  tr:nth-child(even) td { background: #eef5ea; }
  code { background: #e8f2e4; color: #2C5F1E; border-radius: 3px; padding: 0.05em 0.35em; font-size: 0.84em; }
  pre  { background: #263326; color: #b8d8b0; border-radius: 6px; padding: 0.8em 1.1em; font-size: 0.75rem; line-height: 1.5; margin: 0.4em 0; }
  blockquote { background: #eef5ea; border-left: 4px solid #3A7D28; padding: 10px 16px; margin: 0.5em 0; font-size: 0.95rem; color: #2C5F1E; border-radius: 0 4px 4px 0; }
  blockquote p { margin: 0; }
  img { display: block; margin: auto; }
  section::after { color: #888; font-size: 0.7rem; }
---

<!-- _class: title -->

# PCB Defect Detection
## The MLOps Flywheel

Project in Machine Learning Operations
ZHAW · Spring 2026

Bhatia Isha · Chaurasia Vikas · Duss Karin · Müller Jonathan

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

![w:1100](diagrams/architecture.svg)

---

# Tech Stack

| Stage | Tools | Purpose |
| :--- | :--- | :--- |
| **Data & Versioning** | DVC · RustFS (S3) · Git · GitHub | Dataset versioning, reproducible pulls |
| **Model Training** | YOLOv8 · ONNX · Python | Train and export hardware-agnostic model |
| **CI/CD & Reporting** | GitHub Actions · Self-hosted Runner · CML | Automated training on PRs, visual reports |
| **Experiment Tracking** | MLflow (dual instance) | Sandbox :5555 dev · Official :5556 CI/CD |
| **Orchestration** | Airflow | Governance eval · label sync · drift check |
| **Drift Detection** | Evidently AI | Statistical monitoring of production predictions |
| **Annotation** | Label Studio | Human labelling of low-confidence images |
| **Serving & Frontend** | FastAPI · Streamlit · Docker Compose | ONNX inference API + interactive sandbox |

---

# Phase 1 — Training Pipeline

![w:1100](diagrams/phase1.svg)

> CI/CD runs strictly from `params.yaml` — local flags ignored · ONNX export on merge to `main`

---

# Phase 2 — Model Governance

![w:1100](diagrams/phase2.svg)

> FastAPI polls MLflow every 30 s and hot-reloads when a new `@champion` is detected

---

# Phase 3 — Serving & Drift Monitoring

![w:1100](diagrams/phase3.svg)

> Airflow DAG runs daily: ① sync_labels from Label Studio · ② drift_monitor with Evidently AI

---

# Phase 4 — Active Learning

![w:1100](diagrams/phase4.svg)

> Every low-confidence prediction eventually improves the model — the flywheel is self-reinforcing

---

# CI/CD Workflow

![w:1100](diagrams/cicd.svg)

> Self-hosted runner bridges GitHub Actions with local Docker infrastructure (MLflow, RustFS, Airflow)

---

# Human-in-the-Loop Gates

Four checkpoints — **no model reaches production without a human merge**

| Gate | Where | Decision |
| :--- | :--- | :--- |
| **H1 — Annotation** | Label Studio | Draw bounding boxes on low-confidence images |
| **H2 — Training PR** | GitHub PR + CML report | Review confusion matrix & F1 · approve or reject |
| **H3 — Governance PR** | GitHub PR | Safety gate — merge promotes `@champion` |
| **H4 — Retraining PR** | GitHub Drift PR | Decide if detected drift warrants a retrain |

<br>

> No champion is set without a human merge.
> No retraining happens without a human decision.

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

- **Flywheel design**: drift → annotation → retrain → better model → less drift
- **Three Airflow DAGs** orchestrate all automation: governance, sync+drift, deploy
- **Four human gates** ensure no model reaches production unchecked
- **ONNX format** eliminates Mac vs Linux inference discrepancies
- **Dual MLflow** keeps sandbox experiments separate from official CI/CD runs
- **Self-hosted runner** bridges GitHub Actions with local Docker infrastructure

<br>

*Bhatia Isha · Chaurasia Vikas · Duss Karin · Müller Jonathan*
*ZHAW — Machine Learning and Data in Operations · Spring 2026*
