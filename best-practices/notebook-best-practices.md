# Notebook Best Practices Guide

This reference guide establishes engineering and workflow standards for building, executing, and maintaining clean, reproducible, and fault-tolerant notebooks in **Google Colab**.

All **Notebook Execution Subagents** — **Group 1: Data Pipeline** (`env-setup-agent`, `data-cleaning-agent`, `eda-agent`) and **Group 2: Model Selection & Evaluation** (`model-training-agent`, `inference-agent`, `critic-agent`, `report-writer-agent`) — must adhere to the **Universal Standards** below and adjust ceremony based on the active **Workflow Mode**.

---

## 0. Workflow Modes: `Quick Exploration` (Default) vs. `Polished Deliverable`

### Universal Standards (Mandatory in Both Modes)
- **Global Reproducibility Seeds**: Set `random`, `numpy`, and `torch` seeds at the top of the notebook.
- **Composable Data Pipeline & Zero Data Leakage**: Structure Group 1 (`extract` $\rightarrow$ `clean` $\rightarrow$ `transform` $\rightarrow$ `analyze/plot`) using composable functions (`extract_data`, `clean_data`, `transform_data`) and `Pipeline`/`ColumnTransformer` objects. Deduplicate before splitting; fit all scalers, imputers, tokenizers, and encoders — and compute target-aware EDA statistics — strictly on the `train` split.
- **Secret Hygiene**: Never hardcode API keys or tokens; always load credentials via `from google.colab import userdata` or `os.environ`.
- **Plot Quality & Subplot Discipline**: Every chart must have an explicit title, labeled axes (with units where applicable), and readable legends. Avoid subplots: default to one figure per plot (figsize 8x4); allow subplots only for direct comparison of the same variable (max 2 panels, sharey=True, figsize at least 12x4); call `plt.show()` followed by `plt.close()` after each figure.
- **Library-First Modeling (Never Reinvent the Wheel)**: Always use established, battle-tested open-source libraries (e.g. `statsforecast`, `neuralforecast`, `darts`, `prophet` for time series; `scikit-learn`, `lightgbm`, `xgboost` for tabular; `transformers`, `timm` for deep learning). Never hand-roll model algorithms, manual forecasting recurrence loops, or DIY neural architectures from scratch.
- **Reliable Package Installation**: Use `!{sys.executable} -m pip install ...` in a consolidated setup cell so packages install into the active kernel.
- **Clean Colab Markdown & Formatting Restraints**: Never use unsupported GitHub alert syntax (`> [!WARNING]`, `> [!NOTE]`). Do not use emojis in notebook text, headers, or comments. Avoid numbered lists; use unordered bullet lists (`-`) or sub-headers instead.

### Mode-Specific Ceremony
| Practice | `Quick Exploration` Mode (Default) | `Polished Deliverable` Mode |
| :--- | :--- | :--- |
| **Markdown Narrative** | Major section headers (`## Section`) only; concise inline comments for individual cells | Precede every code cell with a Markdown cell explaining the *why* and hypothesis |
| **Data Pipeline Architecture** | Clean inline `extract_data` / `clean_data` / `transform_data` functions and `Pipeline` objects inside notebook cells | Extract pipeline stages into modular `scripts/pipeline_XX_<stage>.py` scripts, write intermediate checkpoints (`data/intermediate/`), and log to `DATA_CLEANING_LOG.md` |
| **Runtime Profiling & Styling** | Use `%%time` and `tqdm` only on genuinely slow cells (`> 5s`); standard `print()` or DataFrame display | Use `%%time` on all non-trivial cells, `tqdm` on loops, and `printmd()` for rich styled metric alerts |

---

## 1. Structure & Narrative Flow
- **Hierarchical Markdown Headers**: Use structured Markdown headers (`#`, `##`, `###`) so the notebook is navigable via the Colab table of contents.
- **Explain the "Why", Not Just the "What"**: Use Markdown headers (or preceding cells in `Polished Deliverable` mode) to explain hypotheses, rationale, and interpretation of findings rather than restating code mechanics.
- **Colab Markdown Standards (No Unsupported GitHub Alerts, No Emojis, No Numbered Lists)**:
  - **No GitHub Alert Syntax**: Google Colab does NOT support GitHub Flavored Markdown alerts (`> [!NOTE]`, `> [!WARNING]`, `> [!IMPORTANT]`, `> [!TIP]`, `> [!CAUTION]`). Never write `> [!ALERT]` blocks in notebook Markdown cells—Colab renders them as raw, unstyled text. Use standard Markdown callouts instead: `> **Warning:** ...` or `> **Note:** ...`.
  - **No Emojis**: Do not use emojis anywhere in notebook Markdown cells, code comments, or printed output. Keep all text plain, professional, and clean.
  - **No Numbered Lists**: Do not use numbered lists (e.g., `1.`, `2.`, `3.`) in notebook narrative or explanation cells. Use clean unordered bullet lists (`-`) or distinct subheadings (`### Subtopic`) instead.
- **Feature Glossary Table Formatting Rules**:
  - **Max 4 columns**: `Column | Type | Domain Context | Meaning`. Never add a `Role` column; group rows under short subheaders instead (e.g. `#### Target Variable`, `#### Numerical Predictors`, `#### Categorical Features`).
  - **Subheader Grouping**: If a column attribute is identical across all rows in a group, state it once in the subheader (e.g. `#### Financial Indicators (Type: float64)`).
  - **Cell Conciseness**: Each cell must be single-line and at most ~60 characters. No bullets (`-`, `*`) or `<br>` line break tags inside table cells.
  - **No LaTeX in Tables**: Do not put LaTeX math formulas, category mappings, or ranges inside table cells. Put formulas, category mappings, and valid ranges in a `"Features notes"` bullet list below the table, written once.
  - **Inline Code Formatting**: Use backticks only for column names (e.g. `` `col_name` ``).
  - **Table Width**: Total rendered table width must fit ~100 characters; if it doesn't, split into more groups with separate subheaders.
- **Results & Debrief (`report-writer-agent`)**: Full workflows conclude with the `## Results & Debrief` section authored by `report-writer-agent` (model comparison table, calibrated model selection justification, single-figure evidence plots with one-line captions, error analysis, and a retrospective debrief with `-` bullet next steps). Standalone `data_pipeline_only` runs conclude with a concise pipeline summary cell.

---

## 2. Cell Granularity, Modularity & Functions
- **Small, Focused Cells**: Keep cells small and dedicated to a single logical step. One clear responsibility per cell makes re-running, debugging, and tracing state changes straightforward.
- **Reusable Functions & Modular Scripts**:
  - Encapsulate pipeline stages (`extract_data`, `clean_data`, `transform_data`) and repeated transformations into clean, reusable functions.
  - In **`Polished Deliverable` mode**, move reusable pipeline functions into separate Python scripts (`scripts/pipeline_XX_<stage>.py`, `scripts/model_utils.py`) so notebook cells remain concise and import-driven.
- **Runtime Profiling with `%%time`**:
  - On cells taking non-trivial time to run (`> 5s`, such as data ingestion, heavy transformations, or model fitting), place `%%time` at line 1 of the cell.
- **Path Handling with `pathlib.Path`**:
  - Prefer `pathlib.Path` over `os.path` and anchor paths from a single `BASE_DIR = Path(".")` configured at the top of the notebook.

---

## 3. Visual Logging & Output Formatting
- **Rich Markdown Output (`Polished Deliverable` mode or key summaries)**:
  ```python
  from IPython.display import Markdown, display

  def printmd(string, color=None):
      colorstr = f"<span style='color:{color}'>{string}</span>" if color else string
      display(Markdown(colorstr))
  ```
- **Visual Progress with `tqdm`**:
  - Wrap long-running loops, downloads, or batch iterations with `tqdm.auto.tqdm`.
- **Plot Quality, Labeling & Subplot Avoidance**:
  - **Default: One Figure Per Plot**: One idea per figure, with a descriptive title, labeled axes with units, and readable legends. Use a consistent figsize (e.g. `8x4`).
  - **No Overview Subplot Grids**: Do not use subplot grids for overviews of many columns. Produce one plot per column, or only the most informative ones, and explicitly state which were skipped.
  - **Strict Subplot Exception**: Subplots are allowed only for direct comparison on the same variable (e.g. train vs. val, before vs. after), max 2 panels, shared axis scales (`sharey=True`), and figsize at least `12x4`.
  - **Explicit Figure Display & Cleanup**: Call `plt.show()` after each figure. Call `plt.close()` to prevent memory bloat.

---

## 4. Kernel, Environment, Secrets & Provenance
- **Reliable Package Installation**:
  - Always target the active kernel's Python interpreter because `!pip` in a subshell can mismatch the kernel environment:
    ```python
    !{sys.executable} -m pip install -q package_name
    ```
- **Consolidated Setup Cell & Version Provenance**:
  - Consolidate all installs at the top of the notebook because Colab VMs reset on disconnection.
  - Log `sys.version` and `__version__` for core libraries (`numpy`, `pandas`/`polars`, `sklearn`, `torch`, etc.) immediately after imports.
- **Secret Management (`google.colab.userdata`)**:
  - Never hardcode credentials in cells. Retrieve keys via Colab Secrets with an environment fallback:
    ```python
    from google.colab import userdata
    import os

    token = userdata.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
    ```

---

## 5. Debugging & Error Handling
- **IPython Debugger (`%debug`)**:
  - When investigating an uncaught exception interactively, `%debug` in a scratch cell opens post-mortem inspection.
- **Programmatic Breakpoints (`set_trace`)**:
  - Use `from IPython.core.debugger import set_trace; set_trace()` only during interactive human debugging—never leave active `set_trace()` calls in automated cells executed via `colab-mcp` because `stdin` blocks indefinitely.

---

## 6. Reproducibility, Data Provenance & Early Validation
- **Seed Setting**: Set random seeds across all active libraries at the top of the notebook:
  ```python
  import os, random, numpy as np

  SEED = 42
  random.seed(SEED)
  os.environ["PYTHONHASHSEED"] = str(SEED)
  np.random.seed(SEED)
  try:
      import torch
      torch.manual_seed(SEED)
      if torch.cuda.is_available():
          torch.cuda.manual_seed_all(SEED)
  except ImportError:
      pass
  ```
- **Data Provenance & Sanity Checks**:
  - Log the dataset source URL/identifier, file byte size, row/column count, and a quick SHA256 hash or sample fingerprint right after extraction in `data-cleaning-agent`.
  - Assert basic invariants immediately (`assert len(raw_df) > 0`, expected columns, zero split overlap).
- **Sample-First Development**:
  - Test complex pipelines on a small representative subset or 1–2 mini-batches before running full multi-epoch training.

---

## 7. Compute, Memory & Session Persistence
- **Deliberate Hardware Runtime**:
  - Verify GPU availability via `!nvidia-smi` and `torch.cuda.is_available()` when deep learning or GPU libraries (`cuDF`, `xgboost` GPU) are used.
- **Explicit Memory Reclamation**:
  - Free large intermediate objects when no longer needed (`del temp_df; gc.collect(); torch.cuda.empty_cache()`).
- **Persistence & Checkpointing**:
  - Save Group 1 Parquet split checkpoints (`train.parquet`, `val.parquet`, `test.parquet`), and when training takes `> 2 minutes` or `Polished Deliverable` mode is active, persist model checkpoints to disk (and sync to Google Drive if session Drive consent was granted).

---

## 8. Subagent Review Checklist

### Universal Checks (Both Modes)
- [ ] Setup cell uses `!{sys.executable} -m pip install` and logs core library versions.
- [ ] Global random seeds (`SEED = 42`) are set at the top of the notebook.
- [ ] Secrets use `google.colab.userdata` or environment variables—zero plaintext keys.
- [ ] Group 1 Data Pipeline (`extract` $\rightarrow$ `clean` $\rightarrow$ `transform` $\rightarrow$ `analyze/plot`) logs data provenance, uses composable pipeline functions/objects, deduplicates pre-split, fits transformers strictly on `train`, and computes target-aware EDA strictly on `train`.
- [ ] Established domain libraries used for models (e.g. StatsForecast, NeuralForecast, LightGBM, Scikit-Learn)—zero hand-rolled / DIY model implementations.
- [ ] Task-appropriate baseline model, evaluation metrics, and labeled visualizations are present (avoiding subplot grids; 8x4 single plots; max 2 comparison panels, sharey=True, 12x4; `plt.show()` + `plt.close()` after each).
- [ ] Markdown cells use clean Colab-supported formatting (no raw `> [!ALERT]` blocks, no emojis, no numbered lists; bullet points `-` or subheaders only).
- [ ] Full workflows conclude with `## Results & Debrief` (`report-writer-agent`) grounded strictly in executed notebook outputs (or a pipeline summary for `data_pipeline_only` runs).

### Additional Checks for `Polished Deliverable` Mode
- [ ] Every code cell is preceded by a Markdown cell explaining intent and rationale.
- [ ] Data Pipeline stages are modularized in `scripts/pipeline_XX_<stage>.py` with intermediate checkpoints and `DATA_CLEANING_LOG.md`.
- [ ] Key outputs use `printmd()` and non-trivial cells include `%%time`.
