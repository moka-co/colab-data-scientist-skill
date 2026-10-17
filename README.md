# Colab Data Scientist Skill

A complete data science and machine learning workflow methodology for coding agents in Google Colab, built on specialized subagents, interactive browser runtime execution with iterative error feedback, task-adaptive modeling across modalities, and consent-driven Google Drive synchronization.

## Table of Contents

- [How it works](#how-it-works)
- [Installation](#installation)
  - [Antigravity](#antigravity)
- [The Basic Workflow](#the-basic-workflow)
- [What's Inside](#whats-inside)
  - [Specialist Subagents](#specialist-subagents)
  - [References & Operational Guides](#references--operational-guides)
  - [Scripts & Utilities](#scripts--utilities)
- [Tooling & Configuration](#tooling--configuration)
  - [Google Colab MCP](#1-google-colab-mcp-colab-mcp)
  - [Google Drive Sync](#2-google-drive-sync-gdrive)
- [When Something Goes Wrong](#when-something-goes-wrong)
- [Philosophy](#philosophy)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## How it works

When you ask your agent to discover datasets or build, run, or debug a notebook in Google Colab, the **Main Orchestrator** sets up a structured workflow tailored to your task:

1. **Upfront Alignment (`ask_question`)**: The Orchestrator confirms your **Workflow Mode** (**`Quick Exploration`** by default for fast, low-ceremony inline code, or **`Polished Deliverable`** for full Markdown narratives, modular `scripts/clean_*.py` scripts, and intermediate checkpoints), your preferred dataframe engine (**Pandas**, **Polars**, **DuckDB**, or **PySpark**), and asks **once per session** whether to enable Google Drive synchronization.
2. **Existing Notebook Inventory**: If attaching to an existing notebook, the Orchestrator inspects existing cells via `get_cells(includeOutputs=False)`, summarizes what already exists, and skips completed phases.
3. **Sequential Specialist Dispatch & Iterative Feedback**: Subagents write their plan and invariants to `<subagent_name>_tasks.md` and return a structured `CELLS_TO_EXECUTE` JSON array. The Orchestrator inserts and executes cells via `colab-mcp`, and if a cell fails or returns an unexpected schema, messages the active subagent via `send_message` with the traceback/output for targeted self-correction.
4. **Task-Adaptive Pipeline**: EDA, preprocessing, baselines, metrics, and visualizations automatically adapt to the task modality (**Tabular**, **NLP/LLM**, **Computer Vision**, **Time Series**, **Reinforcement Learning**, or **Unsupervised**).
5. **Safe Quality Gate**: Before finishing, `critic-agent` audits the notebook. Only fast (`< 5s`), non-destructive leaf/display cell fixes qualify as `SAFE_AUTO_FIXES`; any change to training cells or upstream data-mutating cells is classified under `ESCALATED_FIXES` and requires user confirmation.
6. **Zero-Loss Export & Consent-Based Drive Sync**: The notebook is assembled into a valid `.ipynb` file (preserving outputs) via `scripts/export_to_ipynb.py` and synced to Google Drive if session Drive consent was granted.

## Installation

### Antigravity

Install this skill either globally across all your projects or scoped to a specific repository workspace.

#### Global Installation (Recommended)

To make the skill automatically available across all projects in Antigravity:

**macOS / Linux**:
```bash
git clone https://github.com/obra/colab-data-scientist-skill.git ~/.gemini/config/skills/colab-data-scientist-skill
```

**Windows (PowerShell)**:
```powershell
git clone https://github.com/obra/colab-data-scientist-skill.git "$HOME\.gemini\config\skills\colab-data-scientist-skill"
```

Verify that `SKILL.md`, `agents/`, `references/`, `best-practices/`, and `scripts/` exist inside `~/.gemini/config/skills/colab-data-scientist-skill/`.

#### Workspace-Level Installation

To scope the skill exclusively to a single project:

1. Create the project skills directory:
   ```bash
   mkdir -p .agents/skills
   ```
2. Clone or submodule this repository:
   ```bash
   git submodule add https://github.com/obra/colab-data-scientist-skill.git .agents/skills/colab-data-scientist-skill
   ```

Antigravity automatically discovers skills in `~/.gemini/config/skills/` and `.agents/skills/`, injecting their descriptions on demand.

## The Basic Workflow

```mermaid
flowchart TD
    User([User]) --> Orchestrator[Main Orchestrator]

    subgraph SupportGroup["Orchestrator Support Subagents (Notebook-Independent)"]
        Discovery["dataset-discovery-agent\n(Kaggle, HuggingFace, Google Dataset Search, UCI, OpenML)"]
    end

    Orchestrator <-->|"Standalone Domain Research\nOR Pre-Notebook Discovery"| Discovery

    subgraph NotebookGroup["Notebook Execution Pipeline (Task-Adaptive & Iterative)"]
        direction LR
        A["1. Mode, Engine & Consent\n(Quick Exploration default)"] --> B["2. Environment\n(env-setup-agent)"]
        B --> C["3. EDA\n(eda-agent)"]
        C --> D["4. Preprocessing\n(data-cleaning-agent)"]
        D --> E["5. Model Training\n(model-training-agent)"]
        E --> F["6. Inference\n(inference-agent)"]
        F --> G["7. Quality Gate\n(critic-agent)"]
        G --> H["8. Export & Optional Drive Sync\n(export_to_ipynb + gdrive)"]
    end

    Orchestrator --> A
```

- **Standalone Support Mode (No Notebook Required)**: Run `dataset-discovery-agent` standalone to search Kaggle, HuggingFace Datasets, Google Dataset Search, OpenML, and UCI without opening Colab. Uses official Python APIs and built-in web search (with optional `exa` MCP enhancement) and reports dataset **License**, **Size**, **Last Updated Date**, and **Access/Secret Requirements** (`HF_TOKEN`, `KAGGLE_KEY`).
- **Notebook Execution Mode**:
  1. **Mode, Engine & Session Consent**: Defaults to **`Quick Exploration`** mode (concise inline code, section headers only) unless **`Polished Deliverable`** mode is requested; asks once per session whether to sync artifacts to Google Drive.
  2. **`env-setup-agent`**: Resolves dependencies, handles Colab runtime conflicts, configures secrets via `google.colab.userdata`, sets global random seeds (`SEED = 42`), logs core library versions, and flags C-extension installs requiring kernel restarts (`requires_restart_after: true`).
  3. **`eda-agent`**: Logs dataset provenance (source URL, file size, SHA256/fingerprint) and runs task-adaptive exploration (tabular schemas/distributions, NLP token lengths, CV image grids/dimensions, time-series stationarity/ACF, or RL observation/action spaces).
  4. **`data-cleaning-agent`**: Executes inline leak-free preprocessing in `Quick Exploration` mode, or writes modular `scripts/clean_XX_<name>.py` scripts, intermediate checkpoints, and `DATA_CLEANING_LOG.md` in `Polished Deliverable` mode. Adapts splits and pipelines to tabular, NLP tokenization, CV augmentations, or chronological time-series windows.
  5. **`model-training-agent`**: Establishes a task-appropriate baseline first, logs experiments in a comparative tracking table, and generates task-matched diagnostics and explainability (SHAP/permutation importance for tabular, sample generation/confusion matrices for NLP, Grad-CAM/misclassified grids for CV, forecast overlays for time series, reward curves for RL, or silhouette/UMAP projections for unsupervised).
  6. **`inference-agent`**: Reuses in-memory models or loads checkpoints for batch evaluation and standalone prediction.
  7. **`critic-agent`**: Audits `<subagent_name>_tasks.md` files and live notebook outputs. Automatically applies only fast (`< 5s`) leaf/display `SAFE_AUTO_FIXES` and escalates any training or data-mutating `ESCALATED_FIXES` to the user.
  8. **Export & Optional Drive Sync**: Reconstructs a clean `.ipynb` via `scripts/export_to_ipynb.py` and syncs to Google Drive if session Drive consent was enabled (verifying Drive file IDs and `modifiedTime` to prevent overwriting concurrent edits).

## What's Inside

### Specialist Subagents

Organized into two tiers inside [`agents/`](agents/):

#### 1. Orchestrator Support Subagents (`agents/support/`) — Notebook-Independent
- **[`dataset-discovery-agent`](agents/support/dataset-discovery-agent.md)**: Searches Kaggle, HuggingFace Datasets, Google Dataset Search, UCI, OpenML, Zenodo, and PapersWithCode using resilient API/web search fallbacks. Always reports license, size, recency, and secret requirements.

#### 2. Notebook Execution Subagents (`agents/notebook/`)
Each notebook subagent writes its planning rationale and verification invariants to `<subagent_name>_tasks.md` and returns executable cells in a structured `CELLS_TO_EXECUTE` JSON array:
- **[`env-setup-agent`](agents/notebook/env-setup-agent.md)**: Hardware checks, dependency conflict resolution, `google.colab.userdata` secret retrieval, global seeds, and library version provenance.
- **[`eda-agent`](agents/notebook/eda-agent.md)**: Dataset provenance logging and task-adaptive exploratory analysis across Tabular, NLP, CV, Time Series, and RL.
- **[`data-cleaning-agent`](agents/notebook/data-cleaning-agent.md)**: Leak-free preprocessing and splitting in `Quick Exploration` (inline) or `Polished Deliverable` (modular `scripts/clean_*.py` + checkpoints) mode.
- **[`model-training-agent`](agents/notebook/model-training-agent.md)**: Task-conditional baselines, comparative experiment tracking, diagnostic visualizations, and explainability.
- **[`inference-agent`](agents/notebook/inference-agent.md)**: Batch inference, sanity checks, and formatted prediction outputs.
- **[`critic-agent`](agents/notebook/critic-agent.md)**: Mode- and task-aware quality gate separating fast leaf/display `SAFE_AUTO_FIXES` from `ESCALATED_FIXES`.

### References & Operational Guides

- **[`references/colab-mcp-and-recovery.md`](references/colab-mcp-and-recovery.md)**: Operational reference for `colab-mcp` tools (with rationales), existing notebook entry flow, iterative subagent execution/retry loop via `send_message`, and failure recovery protocols (kernel restarts, variable wipes, OOM/quota limits).
- **[`references/drive-sync-and-export.md`](references/drive-sync-and-export.md)**: Once-per-session Google Drive consent protocol, `.ipynb` export pipeline via `scripts/export_to_ipynb.py`, and Drive file ID collision/concurrent edit safety checks.
- **[`best-practices/notebook-best-practices.md`](best-practices/notebook-best-practices.md)**: Universal notebook standards (seeds, zero leakage, `userdata` secrets, labeled plots) alongside `Quick Exploration` vs. `Polished Deliverable` mode guidelines.
- **[`best-practices/pandas-best-practices.md`](best-practices/pandas-best-practices.md)**: Core pandas patterns (schema contracts, avoiding chained indexing/`iterrows()`, vectorized operations, and explicit merge validations).
- **[`best-practices/data-science-libraries.md`](best-practices/data-science-libraries.md)**: Curated reference catalog of Python libraries across 28 domains with exact `pip package` names, engine compatibility tags (`pandas`/`polars`/`both`), Colab runtime notes, and a line-index map for targeted line-range inspection.

### Scripts & Utilities

- **[`scripts/export_to_ipynb.py`](scripts/export_to_ipynb.py)**: Assembles a standard Jupyter/Colab `nbformat` v4 notebook by combining cell structures from `get_cells(includeOutputs=False)` with step output JSON files, including ANSI escape/progress bar cleaning and fuzzy cell ID matching.

## Tooling & Configuration

### 1. Google Colab MCP (`colab-mcp`)

The Main Orchestrator connects to active Google Colab tabs via [`colab-mcp`](https://github.com/googlecolab/colab-mcp). Add it to `~/.gemini/config/mcp_config.json`:

```json
{
  "mcpServers": {
    "colab-mcp": {
      "command": "uvx",
      "args": [
        "git+https://github.com/googlecolab/colab-mcp"
      ]
    }
  }
}
```

> **Windows Note**: If `uvx` is not in your global `PATH`, specify the full path to `uvx.exe` (e.g. `"C:\\Users\\<YourUsername>\\AppData\\Roaming\\Python\\Python314\\Scripts\\uvx.exe"`).

### 2. Google Drive Sync (`gdrive`)

For optional synchronization into Google Drive:
- Ensure the `gdrive` skill is present in `~/.gemini/config/plugins/gdrive/` (or your local plugins directory).
- The Orchestrator asks once per session before uploading or syncing any files to Google Drive.

## When Something Goes Wrong

- **WebSocket Disconnects / Unknown Tool**: If Colab's idle WebSocket disconnects and tools return `Unknown tool`, call `open_colab_browser_connection` to reconnect.
- **Runtime Disconnects / Variable Wipes**: If a cell fails with `NameError` on previously defined variables after a disconnect, re-run from the setup cell through the latest checkpoint cell rather than re-running multi-hour training from scratch.
- **Package Installs Requiring Kernel Restart**: When a C-extension or core library (`numpy`, `scipy`, `torch`, `pyarrow`) is upgraded (`requires_restart_after: true`), restart the runtime before running downstream import cells.
- **Cell Output Truncation**: Never call `get_cells(includeOutputs=True)` on notebooks with plots or heavy outputs because base64 image payloads exceed WebSocket buffer limits. Use `get_cells(includeOutputs=False)` and merge outputs via `scripts/export_to_ipynb.py`.

## Philosophy

- **Pragmatic Ceremony (`Quick Exploration` by Default)**: Fast, clean inline code for everyday exploration, with full modular scripts and narrative documentation available on demand in `Polished Deliverable` mode.
- **Task-Adaptive Intelligence**: NLP, CV, Time Series, RL, and Tabular tasks each get domain-appropriate baselines, splits, metrics, and diagnostics.
- **Strict Orchestrator Control with Iterative Feedback**: Only the Orchestrator mutates live notebook state, while subagents receive real execution outputs via `send_message` to self-correct errors.
- **Safe Quality Auditing**: Auto-fixes never silently re-run expensive training loops or invalidate downstream data state.

## Acknowledgements

Special thanks to [**Superpowers**](https://github.com/obra/superpowers) by [@obra](https://github.com/obra) for pioneering composable agentic skills, subagent-driven workflows, and disciplined development methodology that served as the key inspiration for this project's architecture and design.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
