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
| **Support** | `dataset-discovery-agent` | [`agents/support/dataset-discovery-agent.md`](agents/support/dataset-discovery-agent.md) | Searches Kaggle, HuggingFace, OpenML, UCI, and web hubs (API-first with `search_web`/`exa` fallback); reports license, size, recency, and auth requirements; asks user via Orchestrator before filtering ambiguous results. |
| **Notebook** | `env-setup-agent` | [`agents/notebook/env-setup-agent.md`](agents/notebook/env-setup-agent.md) | Inspects repos, pins/installs packages (`!{sys.executable} -m pip install`), retrieves Colab `userdata` secrets, sets global seeds, and logs hardware/library versions. |
| **Notebook** | `eda-agent` | [`agents/notebook/eda-agent.md`](agents/notebook/eda-agent.md) | Runs task-adaptive EDA (Tabular, NLP/LLM, CV, Time Series, RL), logs dataset provenance/checksums, and prints schemas, anomalies, and exploratory plots. |
| **Notebook** | `data-cleaning-agent` | [`agents/notebook/data-cleaning-agent.md`](agents/notebook/data-cleaning-agent.md) | Executes task-adaptive preprocessing, tokenization, augmentations, or leak-free splits (`Quick Exploration` inline cells by default, or `Polished Deliverable` `.py` scripts + checkpoints). |
| **Notebook** | `model-training-agent` | [`agents/notebook/model-training-agent.md`](agents/notebook/model-training-agent.md) | Establishes a task-appropriate baseline first, trains candidate models, logs experiment comparisons, and creates task-conditional diagnostics and explainability plots. |
| **Notebook** | `inference-agent` | [`agents/notebook/inference-agent.md`](agents/notebook/inference-agent.md) | Runs single/batched inference and latency benchmarks using in-notebook trained models or user-selected pretrained checkpoints. |
| **Notebook** | `critic-agent` | [`agents/notebook/critic-agent.md`](agents/notebook/critic-agent.md) | Audits `<subagent_name>_tasks.md` against live Colab outputs; separates safe leaf/display fixes (`SAFE_AUTO_FIXES`) from training/data-mutating changes (`ESCALATED_FIXES`). |

> **Subagent Definition Rules**: Define (`define_subagent`) and invoke (`invoke_subagent` with `Model: "inherit"`) only the subagents needed for the user's task. Set `enable_mcp_tools: false` for all `agents/notebook/*` subagents (`true` only for `dataset-discovery-agent` web search), `enable_write_tools: true` (strictly for writing `<subagent_name>_tasks.md`), and `enable_subagent_tools: false`.

---

## Main Orchestrator Workflow Contract

### 0. Standalone Discovery vs. Existing Notebook vs. New Notebook
- **Standalone Support Mode (No Notebook)**: If the user only wants to explore datasets or survey a domain, invoke `dataset-discovery-agent` (`agents/support/dataset-discovery-agent.md`) directly—do **not** open Colab or spawn notebook subagents. If it returns `NEEDS_USER_CLARIFICATION`, pass its `CLARIFICATION_QUESTIONS` to `ask_question` and send the user's answers back via `send_message`.
- **Out-of-Pipeline Scope Check**: If the user's task centers on an `[Out-of-Pipeline]` domain in [`best-practices/data-science-libraries.md`](best-practices/data-science-libraries.md) (§8 Optimization/SAT/SMT/Planning, §17 Simulation/Physics, §27 Federated Learning, §28 Model Serving/APIs), do **not** force standard tabular EDA/Cleaning; clarify workflow expectations via `ask_question`.
- **Existing Notebook Entry Flow**: If the user is editing or debugging an existing notebook (in an open Colab tab or Google Drive), read [`references/colab-mcp-and-recovery.md`](references/colab-mcp-and-recovery.md) first: call `get_cells(includeOutputs=False)` to inventory existing imports, engine, and completed phases, and run **only** the phases needed for the requested change.

### 1. Upfront Session Configuration & Task Checklist (`task.md`)
Before starting a new notebook pipeline (or after dataset selection via `dataset-discovery-agent`), consolidate any unspecified preferences into a **single** `ask_question` call and record them in `task.md`:
1. **Workflow Mode**:
   - **`Quick Exploration` (Default)**: Concise inline code cells, leak-free in-memory transformations, essential Markdown headers, and minimal ceremony (no external `.py` script extraction or intermediate disk checkpoints unless requested).
   - **`Polished Deliverable`**: Full engineering rigor per [`best-practices/notebook-best-practices.md`](best-practices/notebook-best-practices.md) (Markdown before every code cell, modular `scripts/clean_XX_*.py` extraction, `DATA_CLEANING_LOG.md`, `< 100 MB` CSV / `>= 100 MB` Parquet checkpoints, `%%time`, and `printmd()`).
2. **Dataframe Engine** (for tabular tasks when not specified by user or existing notebook): `Polars`, `Pandas`, `DuckDB`, or `PySpark`.
3. **Once-Per-Session Google Drive Sync Policy**: Ask **once** whether to sync the `.ipynb` notebook (and any optional scripts/checkpoints) to Google Drive at the end of the workflow or keep exports local only. Remember and honor this answer for the entire session.

Write the tailored phase checklist to `task.md` (e.g., *EDA-only*: `env-setup` $\rightarrow$ `eda` $\rightarrow$ `critic`; *Full Pipeline*: `env-setup` $\rightarrow$ `eda` $\rightarrow$ `data-cleaning` $\rightarrow$ `model-training` $\rightarrow$ `inference` $\rightarrow$ `critic`) and update `[ ]` $\rightarrow$ `[/]` $\rightarrow$ `[x]` as phases complete.

### 2. Colab Connection, Iterative Subagent Execution & Error Recovery
Read [`references/colab-mcp-and-recovery.md`](references/colab-mcp-and-recovery.md) when connecting to Colab or handling errors. For each required phase in `task.md`:
1. **Line-Range Library Lookup (Optional Advisory Tips)**:
   - Never read all of [`best-practices/data-science-libraries.md`](best-practices/data-science-libraries.md). Read only its **Section-to-Line Index Map** (lines 35–68), then call `view_file` with `StartLine` and `EndLine` for the relevant domain section.
   - Inject **at most 5 advisory library tips** (name + `pip` package only), filtered by `Engine Compatibility` (`polars` / `pandas` / `both`). Priority hierarchy: **User preference > Colab Runtime constraints > Orchestrator suggestions**.
2. **Invoke Subagent & Parse Separated Output Contract**:
   - Pass the artifact directory path, active `workflow_mode` (`quick` default vs. `polished`), task modality (`tabular`, `nlp_llm`, `cv`, `time_series`, `rl`, `unsupervised`), `dataframe_engine`, and live stdout/schema from the preceding phase.
   - For `inference-agent`: automatically use models trained earlier in the notebook; if none exist (or choice is ambiguous), ask the user (`ask_question`) **before** invoking `inference-agent`.
   - Each subagent writes its plan/rationale/invariants to `<subagent_name>_tasks.md` and returns executable cells separately in a structured `CELLS_TO_EXECUTE` block.
3. **Execute Cells & Loop Back on Errors/Surprises (`send_message`)**:
   - Add (`add_text_cell`, `add_code_cell`) and run (`run_code_cell`) each cell in Colab, recording `cellId` and step output file paths.
   - If a cell fails, hits OOM, requires a C-extension kernel restart, or reveals a surprising schema, follow [`references/colab-mcp-and-recovery.md`](references/colab-mcp-and-recovery.md) and send the live traceback/stdout back to the subagent via `send_message` (up to 2 retries) for a patched cell before asking the user.

### 3. Scoped End-of-Notebook Critic Gate (`critic-agent`)
- Once all planned phases finish executing, invoke `critic-agent` with the current notebook cell list, execution outputs, active `workflow_mode`, task modality, and paths to all `<subagent_name>_tasks.md` files.
- **Apply `SAFE_AUTO_FIXES` Automatically**: Only apply fixes to fast (`< 5s`), non-destructive leaf/display cells (e.g., plot labels, formatting, printing already-computed metrics, summary markdown).
- **Confirm `ESCALATED_FIXES` With the User First**: Never auto-apply changes to **model training cells** or **upstream data-mutating cells** — *because* re-running them can invalidate downstream state or consume hours of GPU compute. Present `ESCALATED_FIXES` via `ask_question` first.

### 4. Notebook Export & Google Drive Sync
- When ready to export and sync, read and follow [`references/drive-sync-and-export.md`](references/drive-sync-and-export.md) to assemble the `.ipynb` via `scripts/export_to_ipynb.py` and sync according to the user's session-level `drive_sync_policy`.
