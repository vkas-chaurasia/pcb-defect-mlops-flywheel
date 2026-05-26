# PCB Defect Detection — MLOps Architecture & Flow Diagrams

---

## Diagram 1: System Architecture

> Two-pipeline design: CI/CD training loop (top) feeds the serving and monitoring loop (bottom). The flywheel closes when drift triggers retraining.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '14px'}}}%%
graph TB
    classDef infra    fill:#1C2B3A,stroke:#0d1821,color:#fff,font-weight:bold
    classDef mltools  fill:#1A4731,stroke:#0d2419,color:#fff,font-weight:bold
    classDef registry fill:#1E6B3C,stroke:#0f3a21,color:#fff,font-weight:bold
    classDef decision fill:#C47A1E,stroke:#8B5615,color:#fff,font-weight:bold
    classDef human    fill:#FFD700,stroke:#8B6914,color:#1a1a1a,font-weight:bold

    subgraph TRAIN["  Training & CI/CD  "]
        direction LR
        n1["1. ML Repository\nGit · DVC · RustFS"]:::mltools
        n2["2. GitHub Actions\ndvc repro · YOLOv8 · CML"]:::infra
        n3["3. MLflow\nTracking · Registry"]:::registry
        n1 -->|dvc pull| n2
        n2 -->|log metrics| n3
    end

    subgraph GOV["  Model Governance  "]
        direction LR
        n4["4. Airflow\nmodel_governance_eval"]:::infra
        H_GOV["HUMAN APPROVAL\nReview Governance PR\n& promote @champion"]:::human
        n4 -->|"eval passed · open PR"| H_GOV
    end

    subgraph SERVE["  Serving · Monitoring · Active Learning  "]
        direction LR
        n5["5. FastAPI\nONNX Inference"]:::infra
        n6["6. Streamlit\nFrontend"]:::mltools
        n7["7. Airflow\nDrift Monitor"]:::infra
        n8{"8. Evidently AI\nDrift Detected?"}:::decision
        H_ANNOT["HUMAN\nAnnotate in\nLabel Studio"]:::human
        n9["9. Airflow\nsync_labels"]:::infra
        n6 -->|REST API| n5
        n5 -->|prediction log| n7
        n7 -->|drift check| n8
        n8 -->|"Yes — retrain"| H_ANNOT
        n8 -->|"No — keep serving"| n5
        H_ANNOT -->|annotated labels| n9
    end

    H_PR["HUMAN APPROVAL\nReview CML report\n& merge PR"]:::human

    n2 --- H_PR
    H_PR -->|"merge to main\nregister @staging"| n4
    H_GOV -.->|set @champion alias| n3
    n3 -.->|hot reload| n5
    n9 -.->|sync to data/raw · loop back to 1| n1
```

---

## Diagram 2: MLOps Flywheel — End-to-End Flow

> The four-phase production cycle. Gold nodes = human decisions that gate automation.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '13px'}}}%%
flowchart TD
    classDef human    fill:#FFD700,stroke:#8B6914,color:#1a1a1a,font-weight:bold
    classDef auto     fill:#DBEAFE,stroke:#2563EB,color:#1E3A5F
    classDef decision fill:#FEF9C3,stroke:#CA8A04,color:#451A03
    classDef phase    fill:#F0FDF4,stroke:#16A34A,color:#14532D,font-weight:bold
    classDef terminal fill:#F0FDF4,stroke:#16A34A,color:#14532D

    FLYWHEEL([" ♻ MLOps Flywheel — entry point\n(new data  or  drift detected) "]):::terminal

    %% ─────────────────────────────────────────────────────────────────────
    subgraph P1["Phase 1 — Train & Validate"]
        direction TB
        P1_LABEL["[ CI/CD Pipeline — GitHub Actions + self-hosted runner ]"]:::phase
        PR_OPEN["Developer opens Pull Request\n(with updated params.yaml or data)"]:::auto
        CI_TRAIN["CI/CD: dvc pull from RustFS\n→ dvc repro  (preprocess + train + eval)\n→ log metrics to MLflow Official :5556"]:::auto
        CML_RPT["Post CML report to PR:\nConfusion Matrix, F1, PR-curve\n+ MLflow deep links"]:::auto
        H2["👤 HUMAN IN THE LOOP\nReview CML report & metrics diff vs main\nRequest changes  OR  approve merge"]:::human
        MERGE_DEC{"Approve\nmerge?"}:::decision
        MERGE["Merge to main branch"]:::auto
        REG_STAGING["CI/CD (on push to main):\nExport ONNX → register in MLflow\nApply @staging alias"]:::auto
    end

    %% ─────────────────────────────────────────────────────────────────────
    subgraph P2["Phase 2 — Model Governance (Airflow)"]
        direction TB
        P2_LABEL["[ Airflow DAG: model_governance_eval ]"]:::phase
        EVAL_DAG["Airflow triggered by CI/CD\nafter successful push to main"]:::auto
        EVAL_RUN["Fetch @staging from MLflow\nRun evaluate_staging_model.py\nagainst golden validation set"]:::auto
        EVAL_DEC{"Passes\nevaluation?"}:::decision
        GOV_PR["Airflow creates Governance PR\nUpdates deployment-repo/production_version.json\nvia GitHub API (no git needed in container)"]:::auto
        H3["👤 HUMAN IN THE LOOP\nReview Governance PR\nVerify safety, accuracy, compliance\nApprove promotion to Production"]:::human
        GOV_DEC{"Approve\npromotion?"}:::decision
        CHAMPION["deploy.yml triggered on merge:\nSet @champion alias in MLflow Registry\nFastAPI hot-reloads within 30 s"]:::auto
    end

    %% ─────────────────────────────────────────────────────────────────────
    subgraph P3["Phase 3 — Serve & Monitor"]
        direction TB
        P3_LABEL["[ FastAPI :8000  +  Airflow daily DAG ]"]:::phase
        SERVE["FastAPI loads @champion from MLflow\nStreamlit UI :8501 for interactive inference"]:::auto
        INFER["Production inference:\nPCB image → YOLO defect detection\nPASS / FAIL decision returned"]:::auto
        LOG["Log per-request metrics to prediction_log.csv:\navg_confidence, avg_bbox_area, pass_fail"]:::auto
        DRIFT_DAG["Airflow: daily  data_sync_and_drift_check DAG\nRead last 100 rows of prediction_log.csv"]:::auto
        EV_RUN["Evidently AI:\nCompute Data Drift Report\nvs reference_predictions.csv baseline\nFeatures: avg_confidence, avg_bbox_area, pass_fail"]:::auto
        EV_SAVE["Save HTML + JSON drift report\nRegister snapshot in Evidently Workspace\n→ visible in Evidently UI :8005"]:::auto
        DRIFT_DEC{"≥50% features\ndrifted?"}:::decision
        DRIFT_PR["Airflow creates Drift PR  (retrain/drift-*)\nvia GitHub API\nIncludes Evidently HTML report in branch"]:::auto
        H4["👤 HUMAN IN THE LOOP\nInspect Evidently drift report in PR\nReview avg_confidence trend and failure rate shift\nDecide: retrain now  or  continue monitoring"]:::human
        RETRAIN_DEC{"Approve\nretrain?"}:::decision
    end

    %% ─────────────────────────────────────────────────────────────────────
    subgraph P4["Phase 4 — Active Learning & Data Refinement"]
        direction TB
        P4_LABEL["[ Label Studio :8080  +  Airflow sync_labels DAG ]"]:::phase
        COLLECT["Collect unseen / drifted PCB images\nfrom production or simulation directory"]:::auto
        UPLOAD["Upload images to Label Studio\nor use Streamlit active-learning loop\n(identify high-uncertainty cases)"]:::auto
        H1["👤 HUMAN IN THE LOOP\nAnnotate defect bounding boxes\nin Label Studio UI\n(open, short, mousebite, spur, pin_hole, spurious_copper)"]:::human
        SYNC_DAG["Airflow sync_labels DAG:\nDownload annotations from Label Studio API\nConvert to YOLO format (normalized cx,cy,w,h)\nMark tasks synced  (idempotent)"]:::auto
        DVC_PUSH["Developer:\ndvc add data/raw  →  dvc push to RustFS S3\ngit commit dvc.lock  →  open Pull Request"]:::auto
    end

    %% ─────────────────────────────────────────────────────────────────────
    %% Connections
    FLYWHEEL --> PR_OPEN
    PR_OPEN --> CI_TRAIN --> CML_RPT --> H2 --> MERGE_DEC
    MERGE_DEC -- "❌ Request changes" --> PR_OPEN
    MERGE_DEC -- "✅ Approve" --> MERGE --> REG_STAGING

    REG_STAGING --> EVAL_DAG --> EVAL_RUN --> EVAL_DEC
    EVAL_DEC -- "❌ Fail — tag 'failed'\n    skip governance" --> EVAL_DAG
    EVAL_DEC -- "✅ Pass — tag 'passed'" --> GOV_PR --> H3 --> GOV_DEC
    GOV_DEC -- "❌ Reject" --> EVAL_DAG
    GOV_DEC -- "✅ Approve & merge" --> CHAMPION

    CHAMPION --> SERVE --> INFER --> LOG
    LOG --> DRIFT_DAG --> EV_RUN --> EV_SAVE --> DRIFT_DEC
    DRIFT_DEC -- "No drift\n→ keep serving" --> INFER
    DRIFT_DEC -- "Drift\ndetected" --> DRIFT_PR --> H4 --> RETRAIN_DEC
    RETRAIN_DEC -- "Not now\n→ keep monitoring" --> INFER
    RETRAIN_DEC -- "Yes:\nretrain" --> COLLECT

    COLLECT --> UPLOAD --> H1 --> SYNC_DAG --> DVC_PUSH --> PR_OPEN
```

---

### Human-in-the-Loop Summary

| Gate | Where | What the human decides |
|:--|:--|:--|
| **H1 — Annotation** | Label Studio :8080 | Draw bounding boxes on raw PCB images; quality of labels directly determines model quality |
| **H2 — Training PR review** | GitHub Pull Request | Accept or reject model performance by reviewing CML confusion matrix and F1 vs main |
| **H3 — Governance approval** | GitHub Governance PR | Final safety gate before a model reaches production; approving this merge promotes @champion |
| **H4 — Drift retraining decision** | GitHub Drift PR | Decide whether detected drift warrants a retrain cycle or if monitoring should continue |
