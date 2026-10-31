---
name: colab-data-scientist-skill
description: Use when exploring data science domains or discovering datasets across Kaggle, HuggingFace, and Google Dataset Search, OR when building, editing, and running data science, machine learning, deep learning, EDA, or model inference notebooks in Google Colab and syncing with Google Drive.
---

# Colab Data Scientist Skill (Two-Group Pipeline Orchestration)

## Overview

Multi-agent data science workflow in which the **Main Orchestrator** coordinates support discovery and **two macro notebook phase groups** (only the Main Orchestrator calls `colab-mcp`, `ask_question`, and `gdrive` — *because* concurrent subagent MCP calls over one browser WebSocket cause race conditions):
1. **Orchestrator Support (`agents/support/`)**: Notebook-independent dataset and domain discovery.
2. **Group 1 — Data Pipeline (`agents/notebook/data-pipeline/`)**: Explicitly named and orchestrated as a cohesive **pipeline** (`env-setup` $\rightarrow$ `extract` $\rightarrow$ `clean` $\rightarrow$ `transform` $\rightarrow$ `analyze/plot`), audited at its exit boundary via `critic-agent` (`scope: "post_pipeline"`).
3. **Group 2 — Model Selection & Evaluation (`agents/notebook/model-training-evaluation/`)**: Consumes the verified `DATA_PIPELINE_HANDOFF` (or persisted Group 1 checkpoints) to run `model-training` $\rightarrow$ `inference` $\rightarrow$ `critic` (`final`) $\rightarrow$ `report-writer`.

## Subagent Roster (`agents/`)

| Group / Tier | Subagent Name | Spec File | Role & When to Invoke |
|---|---|---|---|
| **Support** | `dataset-discovery-agent` | [`agents/support/dataset-discovery-agent.md`](agents/support/dataset-discovery-agent.md) | Searches hubs (1–2 query fast-path for named datasets; capped 5-query multi-hub search for open domains), extracts verified metadata/snippets, and hands `SELECTED_DATASET` to Group 1. |
| **Group 1: Data Pipeline** | `env-setup-agent` | [`agents/notebook/data-pipeline/env-setup-agent.md`](agents/notebook/data-pipeline/env-setup-agent.md) | Assumes pre-installed Colab stack, runs `!pip list` probe, installs only missing packages, detects hardware/restart needs, loads `userdata` secrets, and sets `SEED = 42`. |
| **Group 1: Data Pipeline** | `data-cleaning-agent` | [`agents/notebook/data-pipeline/data-cleaning-agent.md`](agents/notebook/data-pipeline/data-cleaning-agent.md) | Owns `extract -> clean -> transform`: `EXTRACT` probes raw schema & logs pinned provenance; `CLEAN_AND_TRANSFORM` builds composable `extract_data`/`clean_data`/`transform_data` functions, deduplicates pre-split, saves Parquet splits, and exports `unfitted_pipeline` + train-fitted `preprocessor`. |
| **Group 1: Data Pipeline** | `eda-agent` | [`agents/notebook/data-pipeline/eda-agent.md`](agents/notebook/data-pipeline/eda-agent.md) | Owns `analyze/plot` in a single pass (`ANALYZE_AND_PLOT`) on cleaned/transformed splits: emits the Dataset Overview & 4-column Feature Glossary Text Cell, runs `train`-only target diagnostics and single-figure (`8x4`) plots, and writes modeling recommendations. |
| **Group 2: Model & Eval** | `model-training-agent` | [`agents/notebook/model-training-evaluation/model-training-agent.md`](agents/notebook/model-training-evaluation/model-training-agent.md) | Consumes `DATA_PIPELINE_HANDOFF`; executes Library-First two-phase training (`BASELINE` $\rightarrow$ `CANDIDATES_AND_TUNING`), checks overfitting/leakage tripwires, logs comparison metrics, and exports `SELECTED_MODEL`. |
| **Group 2: Model & Eval** | `inference-agent` | [`agents/notebook/model-training-evaluation/inference-agent.md`](agents/notebook/model-training-evaluation/inference-agent.md) | Executes memory cleanup, resilient loading (memory + checkpoint fallback), train/serve skew assertions, modality inference, optional latency benchmarks, and persists predictions (`PREDICTIONS_PATH`). |
| **Group 2: Model & Eval** | `critic-agent` | [`agents/notebook/model-training-evaluation/critic-agent.md`](agents/notebook/model-training-evaluation/critic-agent.md) | Audits across 3 scoped gates (`post_pipeline`, `post_baseline`, `final`), fact-checks Markdown claims against live outputs, and separates `< 5s` leaf `SAFE_AUTO_FIXES` from user-confirmed `ESCALATED_FIXES`. |
| **Group 2: Model & Eval** | `report-writer-agent` | [`agents/notebook/model-training-evaluation/report-writer-agent.md`](agents/notebook/model-training-evaluation/report-writer-agent.md) | Final subagent after `critic-agent` (`final`): treats executed notebook `cells`+`outputs` as single source of truth, follows [`notebook-best-practices.md`](best-practices/notebook-best-practices.md), appends `## Results & Debrief` + evidence plots, writes `report.md`, and never runs new experiments. |

> **Subagent Definition Rules**: Define (`define_subagent`) and invoke (`invoke_subagent` with `Model: "inherit"`) only the subagents needed for the active scope. Set `enable_mcp_tools: false` for all `agents/notebook/**` subagents (`true` only for `dataset-discovery-agent`), `enable_write_tools: true` (strictly for `<subagent_name>_tasks.md`, plus `report.md` in `report-writer-agent`), and `enable_subagent_tools: false`. Each execution subagent returns executable notebook cells in `CELLS_TO_EXECUTE`.

---

## Main Orchestrator Workflow Contract

### 0. Discovery, Pipeline Scopes & Existing Notebooks
- **Pre-Notebook Discovery**: When resolving datasets, invoke `dataset-discovery-agent`. Handle `NEEDS_INPUT` / `NO_VIABLE_CANDIDATES` via `ask_question`. Pass `SELECTED_DATASET` directly to `data-cleaning-agent` (`PHASE: "EXTRACT"`).
- **Pipeline Scope Selection**: Treat Group 1 and Group 2 as modular pipeline stages:
  - **`data_pipeline_only`**: For ETL, cleaning, feature engineering, or EDA-only requests, run Group 1 + `critic-agent` (`scope: "post_pipeline"`), export/sync, and stop cleanly without invoking Group 2.
  - **`full_workflow`**: Run Group 1 $\rightarrow$ `post_pipeline` critic gate $\rightarrow$ `DATA_PIPELINE_HANDOFF` $\rightarrow$ Group 2 (ending with `report-writer-agent`).
  - **`model_evaluation_resume`**: When Group 1 is already complete in an existing notebook or persisted on disk (`train.parquet`, `val.parquet`, `test.parquet` + serialized pipeline), skip Group 1 and start directly at Group 2.
- **Out-of-Pipeline & Existing Notebooks**: For `[Out-of-Pipeline]` domains in [`best-practices/data-science-libraries.md`](best-practices/data-science-libraries.md) (§8, §17, §27, §28), clarify via `ask_question` rather than forcing standard ETL/EDA. On existing notebooks, read [`references/colab-mcp-and-recovery.md`](references/colab-mcp-and-recovery.md) first and inventory via `get_cells(includeOutputs=False)`.

### 1. Upfront Session Configuration (`session-configuration.md`) & Two-Group Checklist (`task.md`)
Consolidate unspecified preferences into a **single** `ask_question` call before starting:
1. **Workflow Mode**: **`Quick Exploration` (Default)** (concise inline pipeline functions, essential headers, minimal ceremony) vs. **`Polished Deliverable`** (full rigor per [`best-practices/notebook-best-practices.md`](best-practices/notebook-best-practices.md): Markdown before every code cell, modular `scripts/pipeline_XX_<stage>.py`, `DATA_CLEANING_LOG.md`, Parquet checkpoints, `%%time`, `printmd()`).
2. **Dataframe Engine** (tabular): `Polars`, `Pandas`, `DuckDB`, or `PySpark`.
3. **Compute Budget & Once-Per-Session Google Drive Sync Policy**: Training quota (default: 120s in `quick` mode) and once-per-session Drive sync consent.
- **Persist State**: Write all user requirements, mode, engine, quota, Drive policy, and active **Pipeline Scope** (`data_pipeline_only` / `full_workflow` / `model_evaluation_resume`) to `session-configuration.md`. Write `task.md` grouped explicitly under **Group 1: Data Pipeline** and **Group 2: Model Selection & Evaluation**, updating `[ ]` $\rightarrow$ `[/]` $\rightarrow$ `[x]`.

### 2. Group 1 Orchestration — Data Pipeline (`env-setup` $\rightarrow$ `extract` $\rightarrow$ `clean` $\rightarrow$ `transform` $\rightarrow$ `analyze/plot`)
Read [`references/colab-mcp-and-recovery.md`](references/colab-mcp-and-recovery.md) for Colab MCP mechanics and retries (up to 2 `send_message` retries on runtime/schema errors before `ask_question`).
1. **Targeted Library Lookup & Library-First Policy**: Read only the **Section-to-Line Index Map** (lines 35–68) of [`best-practices/data-science-libraries.md`](best-practices/data-science-libraries.md), then `view_file` on the matching section. Inject $\le 5$ compatible library tips (**User preference > Colab Runtime > Orchestrator**); strictly forbid hand-rolled models.
2. **Environment Setup (`env-setup-agent`)**: Run `STEP: "PROBE"` (`!pip list`), loop stdout via `send_message`, and execute `STEP: "PLAN"` (missing packages only). Handle `INCOMPATIBLE` (halt) and `HARDWARE_MISMATCH` (`ask_question`). If `REQUIRES_RESTART: true`, run install cell at `index: 0`, restart runtime, warn user, and run smoke-test cell.
3. **Composable Extract, Clean & Transform (`data-cleaning-agent`)**:
   - Pass `task_modality`, `dataset_source` (`SELECTED_DATASET`), `dataframe_engine`, `artifact_dir` (+ optional `target_column`, `workflow_mode`, `intended_model_family`).
   - Run `PHASE: "EXTRACT"` (`extract_data()` immutable load, pinned provenance, raw schema/null/sentinel/duplicate probe) and loop live stdout back via `send_message`.
   - Run `PHASE: "CLEAN_AND_TRANSFORM"` (`clean_data()` ordered hygiene + Per-Column Log $\rightarrow$ pre-split dedup & leak-free split + `train.parquet`/`val.parquet`/`test.parquet` $\rightarrow$ `transform_data()` + `unfitted_pipeline` + train-fitted `preprocessor`). Prompt via `ask_question` if `STATUS: "NEEDS_USER_CONFIRMATION"` (`> 5%` row drop or non-ID column drop).
4. **Post-Transform Analysis & Plotting (`eda-agent`)**: Pass `pipeline_variables` and `schema_and_cleaning_summary`. Execute single-pass `PHASE: "ANALYZE_AND_PLOT"` (mandatory Dataset Overview & 4-column Feature Glossary Text Cell, strict `train`-only target diagnostics, `8x4` single-figure plots, and `## Findings & Modeling Recommendations` in `eda-agent_tasks.md`).
5. **Group 1 Boundary Audit & `DATA_PIPELINE_HANDOFF`**: Invoke `critic-agent` with `scope: "post_pipeline"`. Apply `< 5s` leaf `SAFE_AUTO_FIXES`; confirm `ESCALATED_FIXES` via `ask_question`. Package `DATA_PIPELINE_HANDOFF` (`split_objects`, `parquet_checkpoints`, `unfitted_pipeline`, `fitted_preprocessor`, `per_column_changes`, `eda_findings`). Stop and export if scope is `data_pipeline_only`.

### 3. Group 2 Orchestration — Model Selection & Evaluation
1. **Phased Model Selection & Leakage Tripwire (`model-training-agent`)**:
   - Pass `DATA_PIPELINE_HANDOFF` (or resumed Group 1 checkpoints); install any `ADDITIONAL_PACKAGES` in the top setup cell.
   - Run `PHASE: "BASELINE"` and invoke `critic-agent` (`scope: "post_baseline"`). If `STATUS: "SUSPECTED_LEAKAGE"` triggers, halt Group 2 and route back to Group 1 (`data-cleaning-agent`).
   - Loop baseline timing/metrics back via `send_message` to run `PHASE: "CANDIDATES_AND_TUNING"` within compute quota, logging full comparison metrics and exporting `SELECTED_MODEL`.
2. **Memory-Safe Inference (`inference-agent`, when applicable)**: Confirm `inference_data` if ambiguous (`ask_question`), run VRAM/RAM cleanup, resilient loading, train/serve skew assertions, modality inference, and save predictions to `PREDICTIONS_PATH`.
3. **Final Process Audit (`critic-agent`, `scope: "final"`)**: Run **before** `report-writer-agent` (passing cells, head/tail/signal truncated outputs with `[PLOT_RENDERED: ...]` descriptors, `workflow_mode`, and task files). Auto-apply `SAFE_AUTO_FIXES`; escalate `ESCALATED_FIXES` via `ask_question`; record any unresolved blocking issues in `critic-agent_tasks.md`.
4. **Final Results Report & Debrief (`report-writer-agent`)**: Pass ordered executed notebook `cells` and live `outputs` (**authoritative source of truth** over `<subagent>_tasks.md`) + `session-configuration.md` + task summaries (including `critic-agent_tasks.md`). If it emits `PHASE: "EVIDENCE_AND_ERROR_ANALYSIS"`, execute read-only slice/plot cells and loop stdout back via `send_message`; then execute `PHASE: "WRITE_REPORT"` appending `## Results & Debrief`, saving standalone `report.md`, and surfacing `ORCHESTRATOR_SUMMARY`.

### 4. Notebook Export & Google Drive Sync
Follow [`references/drive-sync-and-export.md`](references/drive-sync-and-export.md) to assemble `.ipynb` via `scripts/export_to_ipynb.py`. If Drive sync is enabled, prompt via `ask_question` with a recommended path (`Colab Notebooks/<NotebookName>.ipynb`); user-provided paths or hints strictly take precedence.
