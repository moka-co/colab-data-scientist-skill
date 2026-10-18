---
name: colab-data-scientist-skill
description: Use when exploring data science domains or discovering datasets across Kaggle, HuggingFace, and Google Dataset Search, OR when building, editing, and running data science, machine learning, deep learning, EDA, or model inference notebooks in Google Colab and syncing with Google Drive.
---

# Colab Data Scientist Skill (Multi-Subagent Orchestration)

## Overview

Multi-agent data science workflow in which the **Main Orchestrator** coordinates two separate tiers of subagents:
1. **Orchestrator Support Subagents (`agents/support/`)**: Independent of notebooks and Google Colab. Invoked standalone for domain/dataset research or pre-notebook discovery.
2. **Notebook Execution Subagents (`agents/notebook/`)**: Invoked sequentially when creating or editing a Google Colab notebook. Only the Main Orchestrator calls `colab-mcp`, `ask_question`, and `gdrive` tools — *because* concurrent subagent MCP calls over a single browser WebSocket cause race conditions.

## Subagent Roster (`agents/`)

| Tier | Subagent Name | Spec File | Role & When to Invoke |
|---|---|---|---|
| **Support** | `dataset-discovery-agent` | [`agents/support/dataset-discovery-agent.md`](agents/support/dataset-discovery-agent.md) | Searches dataset hubs (API-first with search fallback), deduplicates mirrors, extracts verified metadata (license, size, auth), and returns ranked candidates or clarification. |
| **Notebook** | `env-setup-agent` | [`agents/notebook/env-setup-agent.md`](agents/notebook/env-setup-agent.md) | Reads repo manifests, probes live Colab environment, pins added packages while protecting core stack, detects hardware mismatches and C-extension restarts, retrieves `userdata` secrets, sets seeds, and runs smoke tests. |
| **Notebook** | `eda-agent` | [`agents/notebook/eda-agent.md`](agents/notebook/eda-agent.md) | Runs task-adaptive EDA across modalities in a two-phase flow (probe & inspect $\rightarrow$ schema-aware diagnostics), logs provenance, and flags leakage/anomalies. |
| **Notebook** | `data-cleaning-agent` | [`agents/notebook/data-cleaning-agent.md`](agents/notebook/data-cleaning-agent.md) | Executes ordered cleaning, pre-split deduplication, leak-free splitting, and exports unfitted preprocessing pipelines. |
| **Notebook** | `model-training-agent` | [`agents/notebook/model-training-agent.md`](agents/notebook/model-training-agent.md) | Trains models in a two-phase flow (baseline $\rightarrow$ tuned candidate), checks overfitting/leakage tripwires, and exports complete inference pipelines. |
| **Notebook** | `inference-agent` | [`agents/notebook/inference-agent.md`](agents/notebook/inference-agent.md) | Executes inference with memory-safe loading, train/serve skew checks, modality pipelines, and persisted predictions. |
| **Notebook** | `critic-agent` | [`agents/notebook/critic-agent.md`](agents/notebook/critic-agent.md) | Audits notebook quality across 3 scoped gates (`post_cleaning`, `post_baseline`, `final`); separates fast leaf fixes (`SAFE_AUTO_FIXES`) from training/data changes (`ESCALATED_FIXES`). |

> **Subagent Definition Rules**: Define (`define_subagent`) and invoke (`invoke_subagent` with `Model: "inherit"`) only the subagents needed for the user's task. Set `enable_mcp_tools: false` for all `agents/notebook/*` subagents (`true` only for `dataset-discovery-agent` web search), `enable_write_tools: true` (strictly for writing `<subagent_name>_tasks.md`), and `enable_subagent_tools: false`. Notebook subagents use built-in read tools to inspect repository manifests and code entrypoints.

---

## Main Orchestrator Workflow Contract

### 0. Standalone Discovery vs. Existing Notebook vs. New Notebook
- **Standalone vs. Pre-Notebook Discovery**: If exploring a domain or discovering datasets, invoke `dataset-discovery-agent` with `task`, `modality`, and `runtime_constraints`. If it returns `STATUS: "NEEDS_INPUT"`, pass `CLARIFICATION_QUESTIONS` to `ask_question`. If `STATUS: "NO_VIABLE_CANDIDATES"`, present broadening options. When candidate is picked (`STATUS: "CANDIDATES_FOUND"`), pass the structured `SELECTED_DATASET` payload to `eda-agent`.
- **Out-of-Pipeline Scope Check**: If the user's task centers on an `[Out-of-Pipeline]` domain in [`best-practices/data-science-libraries.md`](best-practices/data-science-libraries.md) (§8 Optimization/SAT/SMT/Planning, §17 Simulation/Physics, §27 Federated Learning, §28 Model Serving/APIs), do **not** force standard tabular EDA/Cleaning; clarify workflow expectations via `ask_question`.
- **Existing Notebook Entry Flow**: If editing an existing notebook, read [`references/colab-mcp-and-recovery.md`](references/colab-mcp-and-recovery.md) first: call `get_cells(includeOutputs=False)` to inventory existing imports, engine, and completed phases, and run **only** the phases needed for the requested change.

### 1. Upfront Session Configuration & Task Checklist (`task.md`)
Before starting a new notebook pipeline (or after dataset selection via `dataset-discovery-agent`), consolidate any unspecified preferences into a **single** `ask_question` call and record them in `task.md`:
1. **Workflow Mode**:
   - **`Quick Exploration` (Default)**: Concise inline code cells, leak-free in-memory transformations, essential Markdown headers, and minimal ceremony (no external `.py` script extraction or intermediate disk checkpoints unless requested).
   - **`Polished Deliverable`**: Full engineering rigor per [`best-practices/notebook-best-practices.md`](best-practices/notebook-best-practices.md) (Markdown before every code cell, modular `scripts/clean_XX_*.py` extraction, `DATA_CLEANING_LOG.md`, Parquet checkpoints, `%%time`, and `printmd()`).
2. **Dataframe Engine** (for tabular tasks when not specified by user or existing notebook): `Polars`, `Pandas`, `DuckDB`, or `PySpark`.
3. **Compute Budget & Once-Per-Session Google Drive Sync Policy**: Confirm optional training time quota (default: 120s in `quick` mode) and ask **once** whether to sync the `.ipynb` notebook (and any optional scripts/checkpoints) to Google Drive at workflow completion or keep exports local only.

Write the tailored phase checklist to `task.md` (e.g., *Full Pipeline*: `env-setup` $\rightarrow$ `eda` $\rightarrow$ `data-cleaning` $\rightarrow$ `model-training` $\rightarrow$ `inference` $\rightarrow$ `critic`) and update `[ ]` $\rightarrow$ `[/]` $\rightarrow$ `[x]` as phases complete.

### 2. Colab Connection, Iterative Subagent Execution & Error Recovery
Read [`references/colab-mcp-and-recovery.md`](references/colab-mcp-and-recovery.md) when connecting to Colab or handling errors. For each required phase in `task.md`:
1. **Line-Range Library Lookup (Optional Advisory Tips)**:
   - Never read all of [`best-practices/data-science-libraries.md`](best-practices/data-science-libraries.md). Read only its **Section-to-Line Index Map** (lines 35–68), then call `view_file` with `StartLine` and `EndLine` for the relevant domain section.
   - Inject **at most 5 advisory library tips** (name + `pip` package only), filtered by `Engine Compatibility` (`polars` / `pandas` / `both`). Priority hierarchy: **User preference > Colab Runtime constraints > Orchestrator suggestions**.
2. **Two-Step Environment Setup Loop (`env-setup-agent`)**:
   - **Step 1 (Probe)**: Invoke `env-setup-agent` to receive its initial probe cell (`STEP: "PROBE"`). Execute this cell in Colab and send its stdout (Python version, GPU status, `pip list` grep for core libraries) back via `send_message`.
   - **Step 2 (Plan & Validation)**: Receive the final structured payload. If `INCOMPATIBLE` is reported (e.g. Python version mismatch or conda-only packages), halt and report to the user. If `HARDWARE_MISMATCH` is reported (e.g., `wants GPU, found CPU`), prompt the user via `ask_question` to switch Colab runtime (*Runtime* $\rightarrow$ *Change runtime type*) before continuing.
   - **Restart & Resume Protocol**: If `REQUIRES_RESTART: true`, place the install cell first in the notebook (`index: 0`) and run it. Immediately restart the Colab runtime (via `open_colab_browser_connection` / runtime restart), warn the user that variables are reset, and run only the subsequent verification/smoke-test cell before proceeding.
3. **Two-Phase Adaptive EDA Loop (`eda-agent`)**:
   - Pass required inputs (`task_modality`, `dataset_source`, `dataframe_engine`, `artifact_dir`) and optional inputs (`target_column`, `workflow_mode`, `split_info`). If it returns `STATUS: "NEEDS_INPUT"`, prompt user via `ask_question`.
   - Run Phase 1 inspection cells (`PHASE: "INSPECT"`), capture outputs, and loop back via `send_message` so `eda-agent` drafts Phase 2 diagnostics (`PHASE: "VISUALIZE_AND_DIAGNOSE"`) and updates `eda-agent_tasks.md` with verified `## Findings`.
4. **Ordered Cleaning & Leak-Free Splitting (`data-cleaning-agent`)**:
   - Pass inputs (`task_modality`, `dataframe_engine`, `artifact_dir`, `target_column`, `id_columns`, `group_column`, `intended_model_family`, `workflow_mode`).
   - If `STATUS: "NEEDS_USER_CONFIRMATION"` is returned (e.g., dropping $> 5\%$ rows or non-trivial columns), prompt user via `ask_question` before executing.
   - Execute cleaning, pre-split deduplication, and Parquet checkpointing. Invoke `critic-agent` with `scope: "post_cleaning"`. If data leakage or split overlap is detected, resolve before training. Pass unfitted pipeline and clean split frames to `model-training-agent`.
5. **Phased Model Training & Leakage Tripwire (`model-training-agent`)**:
   - Anticipate modeling dependencies during setup (or install `ADDITIONAL_PACKAGES` at the top setup cell). Note that library recommendations are strictly advisory and must be checked against what is installed.
   - Execute Phase 1 (`PHASE: "BASELINE"`), capture validation score and `%%time` duration. Invoke `critic-agent` with `scope: "post_baseline"`. If metric triggers the "too good" tripwire (`STATUS: "SUSPECTED_LEAKAGE"`), halt and route back to EDA/cleaning. Otherwise, loop back via `send_message`.
   - Execute Phase 2 (`PHASE: "CANDIDATES_AND_TUNING"`) within the compute budget.
   - Pass the verified `SELECTED_MODEL` payload (full fitted pipeline, feature order, label mapping, decision threshold, load snippet) directly to `inference-agent`.
6. **Inference Execution & Prediction Persistence (`inference-agent`)**:
   - Each subagent writes its plan/rationale/invariants to `<subagent_name>_tasks.md` and returns executable cells separately in `CELLS_TO_EXECUTE`.
   - Pass `selected_model` and `inference_data`. If missing or ambiguous, prompt user via `ask_question` **before** invoking `inference-agent`.
   - Execute memory cleanup, resilient loading (in-memory with checkpoint fallback), schema/skew assertions, modality inference pipeline, and persist predictions to disk (`PREDICTIONS_PATH`).
   - If a cell fails, hits OOM, or returns an unexpected schema, send live traceback/stdout back via `send_message` (up to 2 retries) for a patched cell before asking the user.

### 3. Multi-Gate Critic Audits (`critic-agent`)
- Reused across 3 gates (`post_cleaning`, `post_baseline`, `final`). For each gate, pass `scope`, cells, truncated outputs (head/tail/signals, with images stripped to `[PLOT_RENDERED: ...]`), active `workflow_mode`, and relevant task files.
- **Apply `SAFE_AUTO_FIXES` Automatically**: Fast (`< 5s`), non-destructive leaf/display fixes (e.g., plot labels, formatting, printing already-computed metrics, summary markdown).
- **Confirm `ESCALATED_FIXES` With User First**: Never auto-apply changes to **model training cells** or **upstream data-mutating cells** (data leakage, split contamination, metric tripwires). Present `ESCALATED_FIXES` via `ask_question` first.

### 4. Notebook Export & Google Drive Sync
- When ready to export and sync, read and follow [`references/drive-sync-and-export.md`](references/drive-sync-and-export.md) to assemble the `.ipynb` via `scripts/export_to_ipynb.py` and sync according to the user's session-level `drive_sync_policy`.
