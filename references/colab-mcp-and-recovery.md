# Colab MCP Execution, Existing Notebooks & Failure Recovery Reference

This reference document defines the operational mechanics for managing Google Colab sessions via `colab-mcp`, handling existing notebooks, and recovering from runtime errors, disconnects, kernel restarts, and out-of-memory (OOM) failures.

The Main Orchestrator reads this file when connecting to Colab, opening an existing notebook, or handling runtime execution issues.

---

## 1. Valid `colab-mcp` Tools & Constraints (With Rationales)

- **Exclusive Orchestrator Control**: Only the Main Orchestrator invokes `colab-mcp` tools (`enable_mcp_tools: false` for all notebook subagents) — *because* concurrent tool calls over a single browser WebSocket cause race conditions and out-of-order cell execution.
- **Valid Tool Names**: Use strictly `open_colab_browser_connection`, `get_cells`, `add_text_cell`, `add_code_cell`, `update_cell`, `run_code_cell`, `delete_cell`, and `move_cell`.
  - **Never call `get_cell`, `edit_cell`, or `run_cell`** — *because* those tool names do not exist in the `colab-mcp` server schema and will fail immediately.
- **Never Call `read_url_content` on Colab URLs** (`https://colab.research.google.com/notebooks/...#scrollTo=<cellId>`):
  - *Because* Colab notebook URLs require interactive Google browser authentication and return an unauthenticated sign-in page over raw HTTP. Always extract `<cellId>` from the URL fragment and inspect or mutate it via `get_cells(includeOutputs=False)`, `update_cell(cellId, content)`, and `run_code_cell(cellId)`.
- **Never Call `get_cells(includeOutputs=True)` When Plots or Large Outputs Exist**:
  - *Because* base64-encoded PNG figures and long stdout streams exceed MCP WebSocket payload limits and truncate the response. Always call `get_cells(includeOutputs=False)` and rely on the individual `run_code_cell` step output files.
- **Pre-Flight Liveness & Idle Disconnect Recovery**:
  - At the start of a Colab session or before running a batch of cells, verify connection liveness with `get_cells(includeOutputs=False)`.
  - If any `colab-mcp` call times out or returns `Unknown tool: <name>` — *because* Colab's browser WebSocket disconnects after idle periods — immediately call `open_colab_browser_connection` to re-establish the bridge before retrying.
- **MCP Configuration (If Missing)**:
  - If `colab-mcp` is not yet configured in `~/.gemini/config/mcp_config.json`, register `"command": "uvx"` (or the full path to `uvx` on Windows/macOS/Linux) with `"args": ["git+https://github.com/googlecolab/colab-mcp"]`.

---

## 2. Entry Flow for Existing Notebooks (Open Colab Tab or Google Drive)

When the user asks to edit, debug, extend, or run an **existing** notebook (rather than creating one from scratch):

1. **Inventory Current State First**:
   - Call `get_cells(includeOutputs=False)` to inspect all existing Markdown and Code cells, their `cellId`s, and their `execution_count`s.
   - Identify:
     - Which packages and imports are already present.
     - Which dataframe engine (`pandas`, `polars`, `duckdb`, `pyspark`) or deep learning framework (`torch`, `tensorflow`, `jax`) the notebook already uses — adopt the existing notebook's engine/style automatically unless the user asks to change it.
     - Which phases are already complete (e.g., environment setup, data loading, EDA, cleaning, or model training).
2. **Do Not Blindly Prepend All Phases**:
   - Skip completed phases. Do **not** run `env-setup-agent` at the top if imports and installs are already present and working.
   - Only invoke the specialist subagent(s) needed for the user's specific request (e.g., adding a new model via `model-training-agent`, fixing a bug in a specific cell, or running `inference-agent` on an already-trained model).
3. **Verify In-Memory Kernel State Before Appending Downstream Cells**:
   - If the existing notebook has code cells with `execution_count: null` (or if the kernel was recently restarted), run the necessary upstream cells sequentially via `run_code_cell` so required variables (`df`, `X_train`, `trained_model`) exist in memory before running new downstream cells.

---

## 3. Iterative Subagent Execution & Feedback Loop

Subagents do not have direct access to `colab-mcp`. To ensure subagents can react to real execution outputs, surprising schemas, or runtime errors:

1. **Invoke & Parse**:
   - The Orchestrator invokes the phase subagent (`invoke_subagent`), which updates `<subagent_name>_tasks.md` (plan/rationale) and returns a structured `CELLS_TO_EXECUTE` block in its response.
2. **Execute & Observe**:
   - The Orchestrator adds (`add_text_cell` / `add_code_cell`) and executes (`run_code_cell`) the cells in Colab, capturing stdout, stderr, warnings, and return values.
3. **Loop Back When Needed (`send_message`)**:
   - **On Execution Error or Surprising Output**: If a cell raises an exception, reveals unexpected data anomalies (e.g., corrupted columns, unexpected date formats, class collapse), or produces broken plots, the Orchestrator sends the exact traceback/output back to the same subagent conversation ID via `send_message` (up to **2 retries**).
   - The subagent diagnoses the live output and returns replacement code for the affected cell(s), which the Orchestrator applies via `update_cell` + `run_code_cell`.
   - **Pass Live Outputs Forward**: Once a phase succeeds, the Orchestrator passes the actual execution outputs (e.g., printed schema, missingness counts, baseline metrics, trained variable names) into the prompt of the *next* phase's subagent.

---

## 4. Runtime Failure Recovery Protocols

### A. Cell Execution Exceptions (`SyntaxError`, `KeyError`, `ValueError`, `ImportError`, etc.)
1. **Trivial Typo / Missing Import**: The Orchestrator may patch a one-line import or variable typo directly via `update_cell` + `run_code_cell`.
2. **Domain / Schema / API Error**: Send the full traceback and preceding schema summary to the active subagent via `send_message` to produce a corrected cell (max 2 retries).
3. **Unresolved After 2 Retries**: Stop looping and ask the user via `ask_question` with a concise explanation of the failure and options to proceed.

### B. Package Installations Requiring a Kernel Restart
When `env-setup-agent` installs or upgrades packages that replace pre-installed C-extensions or binary wheels (e.g., upgrading `numpy`, `scipy`, `pillow`, or enabling `cudf.pandas`):
1. Run the `!{sys.executable} -m pip install ...` cell via `run_code_cell`.
2. If pip output warns that a restart is required (or if importing the upgraded library raises a C-API / binary incompatibility error):
   - Execute a one-off restart snippet (`import os; os.kill(os.getpid(), 9)`) or trigger kernel restart.
   - Wait 3–5 seconds, call `open_colab_browser_connection` if needed, and re-run the import/hardware verification cell (skipping the already-completed `pip install` cell).

### C. Runtime Disconnects, Timeouts & Variable Wipe (`NameError` Mid-Workflow)
If Colab's VM resets or disconnects mid-workflow, wiping in-memory variables:
1. Reconnect via `open_colab_browser_connection`.
2. Re-run the environment setup / import cell(s).
3. **Resume from Checkpoints When Available**:
   - Check if intermediate dataset checkpoints (`data/intermediate/*`) or saved model checkpoints exist on local disk or mounted Google Drive.
   - If checkpoints exist, load the latest checkpoint directly instead of re-running expensive upstream data cleaning or training cells.
   - If no checkpoints exist (e.g., in `Quick Exploration` mode), re-run the upstream cells in order (`run_code_cell`).

### D. Out-of-Memory (CPU RAM or GPU VRAM) & Quota Exhaustion
If a cell fails with `torch.cuda.OutOfMemoryError`, `MemoryError`, or a kernel crash (exit code `137` / `SIGKILL`):
1. **Immediate Cleanup**: Execute a cleanup snippet:
   ```python
   import gc
   gc.collect()
   try:
       import torch
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
   except ImportError:
       pass
   ```
2. **Mitigation via Subagent (`send_message`)**:
   - For **Deep Learning / LLM / CV**: Instruct the subagent to reduce `batch_size`, enable gradient accumulation, switch to mixed precision (`fp16` / `bf16`), or use 4-bit/8-bit quantization (`bitsandbytes` / `unsloth`).
   - For **Tabular Data**: Instruct the subagent to downcast numeric dtypes, use lazy evaluation (`pl.scan_parquet` / `pl.scan_csv` or DuckDB out-of-core queries), or sample the dataset after asking the user.
   - If GPU quota is exhausted on Colab, ask the user (`ask_question`) whether to fall back to a lightweight CPU-compatible baseline or pause and export the notebook.
