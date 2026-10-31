# Two-Group Pipeline Restructure & Report Writer Design Specification

**Date:** 2026-10-31  
**Status:** Draft for User Review

---

## 1. Overview & Motivation

The `colab-data-scientist-skill` notebook workflow previously executed a flat sequence of phases where Exploratory Data Analysis (`eda-agent`) ran before Data Cleaning (`data-cleaning-agent`), and ended immediately after `critic-agent` without a dedicated results synthesis and retrospective phase.

This specification restructures the notebook workflow into **two explicit macro phase groups**:

1. **Group 1 — Data Pipeline (`agents/notebook/data-pipeline/`)**:
   - Explicitly named and treated as a cohesive **pipeline** by both the notebook code and the Main Orchestrator.
   - Always executes the core sequence: `env-setup` → `extract` → `clean` → `transform` → `analyze/plot` (EDA), audited at its exit boundary by `critic-agent` (`scope: "post_pipeline"`).
2. **Group 2 — Model Selection & Evaluation (`agents/notebook/model-training-evaluation/`)**:
   - Consumes the verified **Data Pipeline Handoff** to execute `model-training` (`baseline` → `post_baseline` critic gate → `candidates & tuning`) → `inference` → `critic` (`final` gate) → `report-writer` (`## Results & Debrief` + standalone `report.md`).

---

## 2. Directory Structure & Subagent Roster

`agents/notebook/` is reorganized into two subdirectories corresponding to the two macro phase groups:

```text
agents/
├── support/
│   └── dataset-discovery-agent.md        # Notebook-independent dataset search & metadata extraction
└── notebook/
    ├── data-pipeline/                    # Group 1: Data Pipeline
    │   ├── env-setup-agent.md            # Runtime probe (!pip list), missing installs, secrets, seeds
    │   ├── data-cleaning-agent.md        # Extract -> Clean -> Transform composable pipeline
    │   └── eda-agent.md                  # Analyze & Plot (EDA + Feature Glossary on clean/transformed data)
    └── model-training-evaluation/        # Group 2: Model Selection & Evaluation
        ├── model-training-agent.md       # Library-First baseline -> candidate selection, tuning & XAI
        ├── inference-agent.md            # Memory-safe load, train/serve skew checks & prediction persistence
        ├── critic-agent.md               # Scoped quality auditor (post_pipeline, post_baseline, final)
        └── report-writer-agent.md        # Final results report, error analysis & debrief (no new training)
```

- **Relative Link Resolution**: Because notebook subagent specs move one directory deeper (`agents/notebook/<group>/*.md`), all internal relative links to `best-practices/` and `references/` inside those files use `../../../best-practices/...` and `../../../references/...`.

---

## 3. Main Orchestrator Pipeline Contract (`SKILL.md`)

### 3.1 Two-Group Task & State Tracking
- `session-configuration.md` records workflow mode (`Quick Exploration` default vs. `Polished Deliverable`), dataframe engine, compute budget, Google Drive sync policy, and active **Pipeline Scope** (`data_pipeline_only`, `full_workflow`, or `model_evaluation_resume`).
- `task.md` groups phases explicitly under **Group 1: Data Pipeline** and **Group 2: Model Selection & Evaluation**:
  - **Group 1 (`data-pipeline`)**: `env-setup` → `data-cleaning: extract` → `data-cleaning: clean & transform` → `eda: analyze & plot` → `critic: post_pipeline`
  - **Group 2 (`model-training-evaluation`)**: `model-training: baseline` → `critic: post_baseline` → `model-training: candidates & tuning` → `inference` (when applicable) → `critic: final` → `report-writer`

### 3.2 Modular Execution & Checkpoint Boundary
Treating Group 1 as a first-class pipeline enforces three operational guarantees:
1. **Cohesive Unit & Boundary Gate**: Group 1 runs through `extract` → `clean` → `transform` → `analyze/plot` and is audited once as a complete pipeline at `critic-agent` (`scope: "post_pipeline"`) before Group 2 starts.
2. **Standalone or Resumable Execution**:
   - *Data Pipeline Only*: For ETL, data preparation, or EDA-only tasks, the Orchestrator executes Group 1 + `post_pipeline` critic gate, exports the notebook/checkpoints, and stops cleanly.
   - *Group 2 Resume / Standalone*: When attaching to an existing notebook or persisted Group 1 checkpoints (`train.parquet`, `val.parquet`, `test.parquet`, and serialized pipeline objects), the Orchestrator skips Group 1 and starts directly at Group 2.
3. **Composable Pipeline Code in the Notebook**: `extract`, `clean`, and `transform` are structured as reusable pipeline functions and `Pipeline`/`ColumnTransformer` objects (or modular `scripts/pipeline_XX_*.py` scripts in `Polished Deliverable` mode) rather than disconnected ad-hoc mutations.

### 3.3 Conciseness Constraint
- `SKILL.md` remains `<= 140` lines (enforced by `tests/test_skill_integrity.py`).

---

## 4. Group 1 Specification: Data Pipeline (`agents/notebook/data-pipeline/`)

### 4.1 `env-setup-agent`
- Retains its two-step protocol (`STEP: "PROBE"` via `!pip list` → `STEP: "SETUP"` installing only missing packages), setting `SEED = 42`, retrieving `google.colab.userdata` secrets, checking hardware accelerators, and executing safe kernel restarts when binary wheels require it.

### 4.2 `data-cleaning-agent` (`extract` → `clean` → `transform`)
Because `data-cleaning-agent` now runs **before** `eda-agent`, it takes ownership of raw data acquisition and schema probing so cleaning rules are never guessed blindly.

- **Inputs**:
  - Required: `task_modality`, `dataset_source` (or `SELECTED_DATASET` handoff from `dataset-discovery-agent`), `dataframe_engine`, `artifact_dir`.
  - Optional: `target_column`, `workflow_mode` (`"quick"` default or `"polished"`), `split_preferences`, `intended_model_family`.
- **Two-Phase Operational Flow**:
  1. **Phase 1 — `PHASE: "EXTRACT"`**:
     - Implements `extract_data(...)` to acquire and load raw data into an immutable variable (`raw_df`, `raw_view`, `raw_sdf`, or `raw_dataset`).
     - Records pinned data provenance (HuggingFace `revision`, Kaggle version string, chunked SHA256 hash for files `< 100 MB`, or exact byte size + mtime + 1,000-row sample fingerprint for files `>= 100 MB`).
     - Prints raw shape, column names, dtypes, null counts, suspected sentinel values (`-999`, `"?"`, `"N/A"`, etc.), exact duplicate row count, ID/group/time column candidates, and 3–5 sample rows.
     - Returns cells to the Orchestrator; live stdout is looped back via `send_message`.
  2. **Phase 2 — `PHASE: "CLEAN_AND_TRANSFORM"`**:
     - Driven strictly by verified `EXTRACT` stdout (cites observed raw schema metrics in `data-cleaning-agent_tasks.md` rather than an upstream EDA file).
     - Enforces destructive-operation guardrails (`STATUS: "NEEDS_USER_CONFIRMATION"` via `ask_question` before dropping `> 5%` rows, dropping non-trivial feature columns, collapsing any class below 5 samples, or dropping rows with missing targets).
     - **Clean Stage (`clean_data`)**: Executes ordered hygiene rules (sentinel-to-null → dtype/datetime casting → category normalization → pre-split deduplication → domain bounds → non-leaking missingness/outlier rules) and emits a **Per-Column Modification Log**.
     - **Leak-Free Split & Checkpoints**: Executes stratified, group-aware (`GroupShuffleSplit`/`GroupKFold`), or chronological temporal splits; asserts zero cross-split ID/group overlap; saves `train.parquet`, `val.parquet`, and `test.parquet`.
     - **Transform Stage (`transform_data`)**: Applies deterministic row-wise domain feature engineering and constructs:
       - `unfitted_pipeline` (`ColumnTransformer` / `Pipeline` or modality equivalent) for leak-free CV fold fitting in Group 2.
       - `preprocessor` fit strictly on `train` (`X_train_transformed`, `X_val_transformed`) so both `eda-agent` and `model-training-agent` can inspect/consume transformed features without leakage.
     - **Mode Ceremony**: Clean inline `extract_data` / `clean_data` / `transform_data` functions and `Pipeline` objects in `Quick Exploration` mode; modular `%%writefile scripts/pipeline_XX_<stage>.py` scripts + `DATA_CLEANING_LOG.md` in `Polished Deliverable` mode.

### 4.3 `eda-agent` (`analyze/plot`)
Because raw loading, provenance, cleaning, splitting, and feature transformation are already complete, `eda-agent` executes in a **single focused pass** (`PHASE: "ANALYZE_AND_PLOT"`).

- **Inputs**:
  - Required: `task_modality`, `dataframe_engine`, `artifact_dir`, `pipeline_variables` (`raw_df`, `clean_df`, `X_train_raw`, `X_train_transformed`, `y_train`, split names), `schema_and_cleaning_summary`.
  - Optional: `target_column`, `workflow_mode`, `dataset_metadata` (collector, publication date, source overview).
- **Responsibilities**:
  1. **Mandatory Dataset Overview & Feature Glossary Text Cell**:
     - Documents Dataset Overview & Purpose, Collector/Source, Publication Date, Sample Counts (both raw extracted count and post-cleaning/split counts), and the **4-column Feature Glossary** (`Column | Type | Domain Context | Meaning` grouped under subheaders, max ~60 chars/cell, ~100-char width, `"Features notes"` list below) covering cleaned and engineered/transformed features.
  2. **Post-Transform Exploratory Analysis & Visualizations**:
     - Computes distributions, target skewness/class balance, feature-target relationships, correlations (capped at top $K \le 20$ numerical features), temporal ACF/PACF, or CV/NLP sample diagnostics on the cleaned/transformed data.
     - **Strict Zero-Leakage Invariant**: All target-aware statistics, correlations, and feature-target plots are computed **strictly on the `train` partition** (with optional `train` vs. `val` covariate-drift comparisons using max 2 panels with `sharey=True` and `figsize=(12, 4)`).
     - Enforces single-figure plot discipline (`figsize=(8, 4)`, explicit title/axes/units, no overview subplot grids, `plt.show()` + `plt.close()`).
     - Flags any residual proxy/leakage suspects (e.g., engineered or retained feature with $|r| > 0.98$ with target).
  3. **Downstream Modeling Guidance (`eda-agent_tasks.md`)**:
     - Writes `## Findings & Modeling Recommendations` (confirmed post-cleaning shapes, class imbalance ratios, recommended primary/secondary evaluation metrics, candidate model family hints, and any residual leakage warnings).

### 4.4 `DATA_PIPELINE_HANDOFF` Contract
Upon passing Critic Gate 1 (`scope: "post_pipeline"`), the Orchestrator passes this structured context to Group 2:
- `split_objects` (`X_train_raw`, `X_val_raw`, `X_test_raw`, `y_train`, `y_val`, `y_test`, or modality loaders)
- `parquet_checkpoints` (`train.parquet`, `val.parquet`, `test.parquet`)
- `unfitted_pipeline` & `fitted_preprocessor`
- `per_column_changes` summary from `data-cleaning-agent`
- `eda_findings` from `eda-agent_tasks.md`

---

## 5. Group 2 Specification: Model Selection & Evaluation (`agents/notebook/model-training-evaluation/`)

### 5.1 `model-training-agent`
- Consumes `DATA_PIPELINE_HANDOFF` (or loads Group 1 Parquet + pipeline checkpoints directly when resuming).
- Enforces the **Mandatory Library-First Principle** (zero hand-rolled models).
- Executes in 2 phases:
  - `PHASE: "BASELINE"`: Fast domain baseline with `%%time` profiling → audited by `critic-agent` (`scope: "post_baseline"`).
  - `PHASE: "CANDIDATES_AND_TUNING"`: Bounded candidate selection & tuning within `training_quota_seconds`, recording per-model `train_metric`, `val_metric` (and `test_metric` / CV fold std when evaluated), `overfitting_gap`, `train_time_seconds`, hyperparameters, and artifact size so `report-writer-agent` has complete structured evidence.
- Exports end-to-end `SELECTED_MODEL` pipeline artifact.

### 5.2 `inference-agent`
- Invoked when holdout/test inference or deployment verification is part of the workflow.
- Performs RAM/VRAM cleanup, resilient loading (in-memory → `SELECTED_MODEL` disk checkpoint fallback), train/serve schema skew assertions, modality-specific inference, optional p50/p95 latency benchmarking, and saves predictions to `PREDICTIONS_PATH`.

### 5.3 `critic-agent` (3 Scoped Gates)
Located at `agents/notebook/model-training-evaluation/critic-agent.md` and invoked across three gates:
1. **`post_pipeline` Gate** (replaces `post_cleaning`, runs at the end of Group 1 after `eda-agent`):
   - Audits the complete Data Pipeline (`extract` → `clean` → `transform` → `analyze/plot`): pinned provenance, composable pipeline functions/objects, pre-split deduplication, zero cross-split overlap, `train`-only transformer fitting and target-EDA computation, `> 5%` row-drop confirmation, Parquet checkpoints, 4-column Feature Glossary formatting, empirical fact-checking of Markdown claims against live stdout, and single-figure plot rules.
2. **`post_baseline` Gate** (mid Group 2):
   - Audits Library-First baseline compliance, runtime duration sanity, and the "too good" leakage tripwire (`STATUS: "SUSPECTED_LEAKAGE"` routes back to Group 1 `data-cleaning-agent`).
3. **`final` Gate** (Group 2, after `model-training-agent` and `inference-agent`, **before** `report-writer-agent`):
   - Audits candidate model Library-First integrity, `Quick` vs. `Polished` mode compliance, evaluation/XAI plot formatting, and inference resiliency/persistence.
   - Applies `SAFE_AUTO_FIXES` (`< 5s` leaf/display fixes) and escalates `ESCALATED_FIXES` via `ask_question` before `report-writer-agent` runs; any unresolved critic issues are passed explicitly into `report-writer-agent`.

### 5.4 `report-writer-agent` (Final Notebook Subagent)
Located at `agents/notebook/model-training-evaluation/report-writer-agent.md`. Runs **last** in Group 2 after `model-training-agent`, `inference-agent` (if run), and `critic-agent` (`final` gate) have finished.

- **`define_subagent` Configuration**:
  - `name`: `report-writer-agent`
  - `enable_mcp_tools`: `false` (Main Orchestrator executes all `colab-mcp` calls)
  - `enable_write_tools`: `true` (Used strictly to write `report-writer-agent_tasks.md` and standalone `report.md` in `artifact_dir`)
  - `enable_subagent_tools`: `false`
- **Best-Practices Mandate**:
  - MUST adhere to [`best-practices/notebook-best-practices.md`](../../../best-practices/notebook-best-practices.md) (active `workflow_mode`: `quick` default vs. `polished`) and [`best-practices/data-science-libraries.md`](../../../best-practices/data-science-libraries.md):
    - **Colab Markdown Standards**: No unsupported GitHub alert blocks (`> [!WARNING]`, `> [!NOTE]`); use `> **Warning:** ...` or `> **Note:** ...`. Zero emojis in Markdown, comments, or outputs. No numbered lists (`1.`, `2.`) inside notebook Markdown cells—use unordered `-` bullet lists or `###` / `####` subheadings instead.
    - **Table Formatting**: Clean, concise single-line cells (no `<br>` or bullets inside cells; backticks only for identifiers/names).
    - **Plot Quality & Subplot Avoidance**: One plot per figure (`figsize=(8, 4)`), **no subplots**, explicit descriptive title, labeled axes with units, readable legends, `plt.show()` followed by `plt.close()`, plus a one-line Markdown caption per plot stating what it shows and why it matters.
    - **Mode Ceremony**: Concise section narrative in `Quick Exploration` mode; Markdown rationale before every plot/code cell and `printmd()` highlights in `Polished Deliverable` mode.
- **Inputs (Provided by Main Orchestrator)**:
  - Task definition (`task_modality`, `target_column`, business/academic goal, `primary_metric`, `workflow_mode`, `artifact_dir`)
  - Dataset summary (`dataset-discovery-agent`) and key EDA findings (`eda-agent_tasks.md`)
  - Cleaning & transformation summary (`data-cleaning-agent_tasks.md` + `PER_COLUMN_CHANGES`)
  - Per-model results (`model-training-agent`: validation/test metrics, CV fold std if any, hyperparameters, training time, model size, `SELECTED_MODEL`)
  - Inference outputs (`inference-agent`: `PREDICTIONS_PATH`, sample predictions, p50/p95 latency if measured)
  - Critic audit report (`critic-agent_tasks.md`: issues found, severity, resolved vs. unresolved blocking issues)
  - Paths to saved artifacts in Colab runtime / local artifact directory
  - *Missing Input Rule*: Never invent missing metrics or facts; record any unverifiable items under **Missing information** and proceed with verified evidence.
- **Core Responsibilities**:
  1. **Results Report (`## Results & Debrief`)**:
     - **Model Comparison Table**: One row per evaluated model (including baseline), sorted by `primary_metric`, reporting primary/secondary metrics, training time, and relevant constraints (size, latency).
     - **Recommended Model Justification (2–4 sentences)**: Weighs margin over runner-up relative to variance (CV std or bootstrap on saved predictions; prefers simpler/cheaper model when within noise), margin over baseline (flags if complex models barely beat baseline), and validation-vs-test/train overfitting gap. If a blocking critic issue remains unresolved, states it prominently and qualifies affected conclusions.
     - **Evidence Plots**: Only plots supporting a specific conclusion (e.g., metric comparison bar chart, confusion matrix or residual distribution of recommended model, feature importance)—strictly one plot per figure, no subplots, each paired with a one-line caption.
     - **Error Analysis**: Analyzes where the recommended model fails (classes, value ranges, segments) using existing saved predictions/outputs without re-training or re-running model inference.
  2. **Debrief**:
     - **What worked**: Specific pipeline/feature/model choices backed by metric deltas.
     - **What didn't**: Underperforming or low-ROI attempts and likely causes.
     - **Open risks**: Unresolved critic findings, dataset limitations, or production distribution-shift assumptions.
     - **Next steps**: 3–5 concrete, prioritized actions formatted as `-` bullets in Colab Markdown, each naming the target specialist subagent (`-> data-cleaning-agent`, `-> model-training-agent`, etc.).
     - **Missing information**: Anything the report could not verify from execution outputs or artifacts.
- **Strict Non-Experimentation Rules**:
  - Never train, tune, refit, or re-run model evaluation. Derived statistics (metric deltas, ratios, bootstrap intervals, segment error breakdowns) may be computed strictly on already-persisted predictions/metrics.
  - Every cited number must trace to an executed cell output or saved artifact without rank-altering rounding.
  - Use calibrated language ("X outperforms Y" only when the margin exceeds measured variance; otherwise "X and Y are comparable").
- **Outputs**:
  1. `CELLS` & `CELLS_TO_EXECUTE`: Notebook cells appending `## Results & Debrief` Markdown and inline single-figure evidence/error-analysis plot cells.
  2. `REPORT_PATH`: Standalone `report.md` written to `artifact_dir` (same content, referencing saved plot paths; synced to Google Drive companion artifacts folder when full artifact sync is enabled).
  3. `TASKS_FILE`: `report-writer-agent_tasks.md`.
  4. `ORCHESTRATOR_SUMMARY`: Concise summary ($\le 10$ lines) with recommended model, primary metric score, top risk, top next step, and blocking critic issue status.

---

## 6. Updates Across Documentation, References & Tests

1. **`SKILL.md`**:
   - Update Subagent Roster table with `Group 1: Data Pipeline` (`agents/notebook/data-pipeline/*`) and `Group 2: Model Selection & Evaluation` (`agents/notebook/model-training-evaluation/*`), including `report-writer-agent`.
   - Rewrite Section 2 around the two-group workflow (`Group 1: Data Pipeline` → `post_pipeline` critic gate → `Group 2: Model Selection & Evaluation` → `report-writer-agent`), standalone/checkpoint boundaries, and `<= 140` line budget.
2. **`README.md`**:
   - Update architecture overview, Mermaid workflow diagram (two subgraphs inside Notebook Execution: `Group 1: Data Pipeline` and `Group 2: Model Selection & Evaluation`), step-by-step walkthrough, and Specialist Subagents directory listing.
3. **`best-practices/notebook-best-practices.md`**:
   - Include `report-writer-agent` in the Notebook Execution Subagents roster; standardize `Polished Deliverable` modular script naming on `scripts/pipeline_XX_<stage>.py` and align final results/debrief ownership with `report-writer-agent`.
4. **`references/colab-mcp-and-recovery.md` & `references/drive-sync-and-export.md`**:
   - Update existing-notebook phase inventory and mid-session checkpoint recovery to reflect the Group 1 (`Data Pipeline`) → Group 2 (`Model Selection & Evaluation`) boundary.
   - Include `report.md` and pipeline scripts in optional companion Drive sync artifacts.
5. **`tests/test_skill_integrity.py` & Skill Behavioral Verification**:
   - Update path assertions for `agents/notebook/data-pipeline/{env-setup-agent,data-cleaning-agent,eda-agent}.md` and `agents/notebook/model-training-evaluation/{model-training-agent,inference-agent,critic-agent,report-writer-agent}.md`.
   - Verify `CELLS_TO_EXECUTE` contract across all execution subagents (including `report-writer-agent`), `post_pipeline`/`post_baseline`/`final` scopes in `critic-agent.md`, internal markdown link resolution (`../../../best-practices/...`), and `SKILL.md` line count (`<= 140`).
   - Run subagent scenario verification (`writing-skills` TDD cycle) plus `pytest` and `ruff check`.
