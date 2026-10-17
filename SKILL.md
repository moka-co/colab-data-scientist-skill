---
name: colab-data-scientist-skill
description: Use when exploring data science domains or discovering datasets across Kaggle, HuggingFace, and Google Dataset Search, OR when building, editing, and running data science, machine learning, EDA, or model inference notebooks in Google Colab and syncing with Google Drive.
---

# Colab Data Scientist Skill (Multi-Subagent Orchestration)

## Overview

Multi-agent data science workflow in which the **Main Orchestrator** coordinates two completely separate tiers of subagents:
1. **Orchestrator Support Subagents (`agents/support/`)**: Completely independent of notebooks and Google Colab. Can be run standalone when the user just wants to explore a domain or discover available datasets, or invoked by the Orchestrator to gather dataset metadata before building a notebook.
2. **Notebook Execution Subagents (`agents/notebook/`)**: Operate when creating or editing a Google Colab notebook, where the Orchestrator manages the task checklist, Colab execution (`colab-mcp`), user questions (`ask_question`), and Google Drive persistence (`gdrive` skill).

**REQUIRED SUB-SKILL:** Use `gdrive` (from your plugins directory, e.g. `~/.gemini/config/plugins/gdrive/skills/gdrive/SKILL.md`) for all Google Drive operations. Never configure a third-party Google Drive MCP server in your `mcp_config.json`.

## Subagent Roster (`agents/`)

### 1. Orchestrator Support Subagents (`agents/support/`) — Notebook-Independent
Completely decoupled from Google Colab and notebooks. Can be invoked standalone for domain/dataset research without opening Colab, or used by the Main Orchestrator when preparing a notebook workflow. Define with `enable_mcp_tools: true` (for `exa` web search/fetch; **never** `colab-mcp`), `enable_write_tools: true` (strictly for writing `<subagent_name>_tasks.md`), `enable_subagent_tools: false`:

| Subagent Name | Spec File | Role & Mandatory Rules |
|---|---|---|
| `dataset-discovery-agent` | [`agents/support/dataset-discovery-agent.md`](agents/support/dataset-discovery-agent.md) | **Dataset & Database Discovery Expert**: Searches Kaggle, HuggingFace Datasets, Google Dataset Search (`https://datasetsearch.research.google.com/`), UCI ML Repository, OpenML, Zenodo, and PapersWithCode. Writes its granular tasks and insights to `dataset-discovery-agent_tasks.md`. Never silently filters large/ambiguous search results on its own—returns `NEEDS_USER_CLARIFICATION` questions for the Orchestrator to ask the user (`ask_question`) when narrowing down datasets, then outputs a ranked report of matching datasets. |

### 2. Notebook Execution Subagents (`agents/notebook/`) — Colab Notebook Specialists
Design and review cells when a Google Colab notebook is being created or edited. **All notebook execution subagents MUST adhere to [`best-practices/notebook-best-practices.md`](best-practices/notebook-best-practices.md)** (Markdown cells preceding code cells, small/focused cells, extracting logic into `.py` functions, `%%time` profiling, `Pathlib`, rich `printmd()` outputs, early sanity assertions, memory management, and Drive checkpointing).

Define (`define_subagent` with `enable_mcp_tools: false`, `enable_write_tools: true` strictly for writing `<subagent_name>_tasks.md`, `enable_subagent_tools: false`) and invoke (`invoke_subagent` with `Model: "inherit"`) only the subagents required by the user's task:

| Subagent Name | Spec File | Role & Mandatory Rules |
|---|---|---|
| `env-setup-agent` | [`agents/notebook/env-setup-agent.md`](agents/notebook/env-setup-agent.md) | **Environment, Dependencies & Hardware Expert (Always First in Notebook)**: Writes `env-setup-agent_tasks.md`; adheres to [`best-practices/notebook-best-practices.md`](best-practices/notebook-best-practices.md); installs/pins packages (including the chosen dataframe engine via `!{sys.executable} -m pip install`), resolves Colab dependency conflicts, logs CPU/GPU/TPU hardware. **Mandatory**: If importing a repository or project, it MUST first read and understand the repo's code (`requirements.txt`, `pyproject.toml`, entrypoints, imports, core modules). |
| `eda-agent` | [`agents/notebook/eda-agent.md`](agents/notebook/eda-agent.md) | **Exploratory Data Analysis Specialist**: Writes `eda-agent_tasks.md`; adheres to [`best-practices/notebook-best-practices.md`](best-practices/notebook-best-practices.md); loads raw data using the chosen dataframe engine, prints shape, dtypes/schema, missingness, duplicates, target distribution, and the **complete numbered list of features (`df.columns` / `X.columns`)**, validates early assertions, and creates exploratory plots. |
| `data-cleaning-agent` | [`agents/notebook/data-cleaning-agent.md`](agents/notebook/data-cleaning-agent.md) | **Data Cleaning & Preprocessing Specialist**: Writes `data-cleaning-agent_tasks.md`; adheres to [`best-practices/notebook-best-practices.md`](best-practices/notebook-best-practices.md) and [`best-practices/pandas-best-practices.md`](best-practices/pandas-best-practices.md) (when Pandas is selected); never assumes clean data; writes modular, idempotent Python scripts (`scripts/clean_XX_<clarifying_name>.py`), saves intermediate datasets after each major change (`< 100 MB` as `.csv`, `>= 100 MB` as compressed `.parquet`), requests user confirmation via Orchestrator (`ask_question`) before bulk zipping or background offloading, documents every change in `DATA_CLEANING_LOG.md` + transparent notebook Before/After summaries, and performs leakage-free train/val/test splits. |
| `model-training-agent` | [`agents/notebook/model-training-agent.md`](agents/notebook/model-training-agent.md) | **Model Training & Evaluation Specialist**: Writes `model-training-agent_tasks.md`; adheres to [`best-practices/notebook-best-practices.md`](best-practices/notebook-best-practices.md); **always establishes a baseline model first** (the most common or simplest model/architecture for the task, e.g. Logistic Regression, OLS, naive baseline) before fitting complex models; tracks experiments in a structured comparative table; creates dedicated **Training cell(s)** (`.fit()` or training loop with `tqdm`, saving trained model variables/checkpoints to Drive for session resumability), task-adaptive evaluation/error metrics, Dual Visualizations (Diagnostic Panel + Data-Space / learned boundary plot), and feature attribution / explainability analysis (SHAP / permutation importance). |
| `inference-agent` | [`agents/notebook/inference-agent.md`](agents/notebook/inference-agent.md) | **Model Inference Specialist**: Writes `inference-agent_tasks.md`; adheres to [`best-practices/notebook-best-practices.md`](best-practices/notebook-best-practices.md); automatically uses models already trained in the notebook. If no model was trained in the notebook or the choice is unclear, the **Main Orchestrator asks the user first** (`ask_question`), then spawns `inference-agent` later with **(1) which exact model to use** and **(2) where to get it**. |
| `critic-agent` | [`agents/notebook/critic-agent.md`](agents/notebook/critic-agent.md) | **End-of-Notebook Quality Critic**: Runs after all specialist phases finish executing in Colab. Reads all `<subagent_name>_tasks.md` files alongside notebook cells + live outputs, audits full compliance with [`best-practices/notebook-best-practices.md`](best-practices/notebook-best-practices.md) and [`best-practices/pandas-best-practices.md`](best-practices/pandas-best-practices.md), audits experiment tracking and explainability plots, writes `critic-agent_tasks.md`, and returns `SAFE_AUTO_FIXES` (auto-applied by Orchestrator) and `MAJOR_ARCHITECTURAL_SUGGESTIONS` (Orchestrator asks user). |

## Main Orchestrator Workflow Contract

0. **Standalone Support Mode vs. Notebook Mode & User-Driven Dataset Filtering**:
   - **Standalone Support Mode (No Notebook)**: If the user only wants to explore a domain, survey available datasets, or compare data sources (without creating or editing a Colab notebook), invoke `dataset-discovery-agent` (`agents/support/dataset-discovery-agent.md`) directly—do **not** open a Colab connection, ask for a dataframe engine, or spawn any `agents/notebook/` subagents.
   - **Interactive Filtering (`NEEDS_USER_CLARIFICATION`)**: Whenever `dataset-discovery-agent` finds many datasets across different modalities/sizes/tasks or the user's request is underspecified, it returns `NEEDS_USER_CLARIFICATION` with `CLARIFICATION_QUESTIONS`. The Main Orchestrator must present these questions to the user via `ask_question` and pass the user's answers back to `dataset-discovery-agent` via `send_message` so the user—not the agent—decides how to filter the datasets.
   - **Notebook Mode (With Support Agent When Needed)**: If a notebook is being created or edited and a dataset needs to be discovered or located, the **Main Orchestrator** invokes `dataset-discovery-agent` first (completing any interactive filtering above), asks the user (`ask_question`) to select from the final `CANDIDATE_DATASETS`, and then passes the chosen dataset's details into the notebook pipeline below.

1. **Clarify Dataframe Engine Upfront & Write Task/Todo List (Notebook Mode)**:
   - **Dataframe Engine Selection**: If the user did not explicitly specify a dataframe library (e.g., **Pandas** vs. **Polars** vs. **DuckDB** / **PySpark**) in their request, **always ask the user upfront (`ask_question`)** which dataframe engine they prefer before starting `env-setup-agent` / `eda-agent` / `data-cleaning-agent`. Pass the chosen engine (`dataframe_engine`) to every notebook subagent so native idioms are used consistently across the entire notebook.
   - Write a task-tailored checklist to `task.md` (in the session artifact directory) and/or `<project_dir>/todo.md` in the workspace listing the exact subagent phases needed (e.g., *EDA-only*: `env-setup` $\rightarrow$ `eda` $\rightarrow$ `critic`; *Pretrained Inference*: `env-setup` $\rightarrow$ `eda` $\rightarrow$ `inference` $\rightarrow$ `critic`; *Full Training + Inference*: `env-setup` $\rightarrow$ `eda` $\rightarrow$ `data-cleaning` $\rightarrow$ `model-training` $\rightarrow$ `inference` $\rightarrow$ `critic`).
   - Update checkboxes (`[ ]` $\rightarrow$ `[/]` $\rightarrow$ `[x]`) after each phase completes.

2. **Manage Colab Connection (`colab-mcp`) & Pre-flight Liveness Check**:
   - Only the Main Orchestrator calls `colab-mcp` tools (prevents WebSocket race conditions).
   - **Pre-flight & Liveness Check Protocol**:
     - At the start of the session or before each execution batch, run a lightweight ping (e.g., `get_cells(includeOutputs=False)`).
     - If the connection times out, returns an error, or indicates `Unknown tool: <name>` (due to idle WebSocket disconnects), immediately call `open_colab_browser_connection` to restore the active tab bridge before proceeding.
   - Valid `colab-mcp` tools: `get_cells`, `add_text_cell`, `add_code_cell`, `update_cell`, `run_code_cell`, `delete_cell`, `move_cell` (never `get_cell`, `edit_cell`, `run_cell`).
   - If `colab-mcp` is missing from `~/.gemini/config/mcp_config.json`, configure `"command": "uvx"` (or the path to `uvx` on Windows/macOS/Linux) and `"args": ["git+https://github.com/googlecolab/colab-mcp"]`.

3. **Orchestrate Specialist Subagents, Track `<subagent_name>_tasks.md`, & Execute Cells Sequentially**:
   - For each required phase in `task.md`:
     1. Read the subagent's spec file in `agents/support/<name>.md` or `agents/notebook/<name>.md`, call `define_subagent` (if not yet defined in the session), and call `invoke_subagent` passing the task/artifact directory path (where the subagent MUST write `<subagent_name>_tasks.md`), the chosen `dataframe_engine` (`polars`, `pandas`, `duckdb`, `pyspark`), and live context/outputs from the preceding phase.
     2. **Mandatory User Prompt for Checkpoint Offloading / Compression**:
        - If `data-cleaning-agent` returns `OFFLOAD_PROMPT_NEEDED` (due to large or numerous checkpoints), **always ask the user first (`ask_question`)** whether they want checkpoints zipped or uploaded asynchronously to Google Drive before initiating the operation.
     3. **Special Rule for `inference-agent`**:
        - If a model was trained earlier in the notebook by `model-training-agent`, instruct `inference-agent` to automatically use that trained model variable/checkpoint.
        - If no model was trained in the notebook (or multiple models exist and the user did not specify which one), the **Orchestrator must ask the user first** (`ask_question`) before spawning `inference-agent`, and then prompt `inference-agent` with **what exact model to use + where to get it**.
     4. Inspect the subagent's generated `<subagent_name>_tasks.md` file to review its granular sub-tasks, rationale, and invariants, then execute the paired Markdown (`add_text_cell`) and Code (`add_code_cell` + `run_code_cell`) cells in Colab, recording each `cellId` and step output path.

4. **Run End-of-Notebook Critic Gate (`critic-agent`) with Full `<subagent_name>_tasks.md` Insight**:
   - Once all specialist phases have executed in Colab, spawn `critic-agent` with the full notebook cells, execution outputs, and paths to all `<subagent_name>_tasks.md` files (`dataset-discovery-agent_tasks.md`, `env-setup-agent_tasks.md`, `eda-agent_tasks.md`, `data-cleaning-agent_tasks.md`, `model-training-agent_tasks.md`, `inference-agent_tasks.md`).
   - Review `critic-agent_tasks.md` and automatically apply all `SAFE_AUTO_FIXES` in Colab (`update_cell` + `run_code_cell`) before syncing to Drive; ask the user only if `critic-agent` flags `MAJOR_ARCHITECTURAL_SUGGESTIONS`.

5. **Targeting Cells by URL or `<cellId>`**:
   - Never call `read_url_content` on `https://colab.research.google.com/notebooks/empty.ipynb#scrollTo=<cellId>...`. Extract `<cellId>` and use `get_cells(includeOutputs=False)`, `update_cell(cellId, content)`, and `run_code_cell(cellId)`.

6. **Automatic Google Drive Persistence**:
   - Automatically export and upload/update the notebook in Google Drive (`Colab Notebooks`) at the end of the workflow (after Critic fixes) and after subsequent edits—do not wait for the user to ask.
   - When `data-cleaning-agent` runs, also sync its generated cleaning scripts (`scripts/clean_*.py`), intermediate dataset checkpoints (`data/intermediate/*` saved as `.csv` when `< 100 MB` or compressed `.parquet` when `>= 100 MB`), and `DATA_CLEANING_LOG.md` to Google Drive alongside the notebook.
   - **Never** call `get_cells(includeOutputs=True)` when plot outputs exist. Call `get_cells(includeOutputs=False)` and merge `run_code_cell` step output files via `scripts/export_to_ipynb.py`:
     ```bash
     python scripts/export_to_ipynb.py --cells-json cells.json --out "$SCRATCH_DIR/<Name>.ipynb" --name "<Name>.ipynb" --cell-output <cellId>=<step_output_path>
     ```
   - **First save**: `gdrive mutate upload "<local_ipynb_path>" --mime-type "application/vnd.google.colaboratory" --parent <FOLDER_ID>`
   - **Subsequent edits**: `gdrive mutate update <FILE_ID> "<local_ipynb_path>" --mime-type "application/vnd.google.colaboratory"`
