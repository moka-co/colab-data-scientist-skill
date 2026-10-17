# Notebook Best Practices Guide

This reference guide establishes core engineering and workflow standards for building, executing, and maintaining clean, reproducible, and fault-tolerant notebooks in **Google Colab**.

All **Notebook Execution Subagents** (`env-setup-agent`, `eda-agent`, `data-cleaning-agent`, `model-training-agent`, `inference-agent`, `critic-agent`) must reference and adhere to these guidelines.

---

## 1. Structure & Narrative Flow
- **Markdown Before Code**: Always precede every code cell with a Markdown title/description cell explaining what the step intends to achieve and *why*—only then add the code cell.
- **Hierarchical Markdown Headers**: Use structured Markdown headers (`#`, `##`, `###`) to ensure the notebook is automatically navigable via the Colab table of contents.
- **Explain the "Why", Not Just the "What"**: Use Markdown to explain hypotheses, rationale, and interpretation of findings, rather than merely restating code mechanics in comments.
- **Conclusions & Next Steps**: End the notebook with a summary cell highlighting final results, metrics, caveats/limitations, and suggested next steps.

---

## 2. Cell Granularity, Modularity & Functions
- **Small, Focused Cells**: Keep cells small and dedicated to a single logical step. One clear responsibility per cell makes re-running, debugging, and tracing state changes straightforward.
- **Extract Logic into Functions & `.py` Modules**:
  - Frequently rewrite cell logic into reusable, tested functions.
  - Move these functions at regular intervals into separate Python scripts (`scripts/clean_*.py`, `scripts/model_utils.py`, etc.) to keep notebook cells concise and import-driven.
- **Runtime Profiling with `%%time`**:
  - If a particular cell takes non-trivial time to run (e.g. data ingestion, heavy transformations, model fitting), add the `%%time` cell magic at the very top as both a runtime logger and a visual indicator.
- **Path Handling with `Pathlib`**:
  - Avoid `os.path`. Use `pathlib.Path` across all file manipulations for readable, platform-independent path operations.
- **Avoid Hardcoded Absolute Paths**:
  - Use relative paths or anchor paths from a single `BASE_DIR = Path(...)` configured at the top of the notebook.

---

## 3. Rich Markdown Output & Visual Logging
- **Markdown Printing for Rich Output**:
  - Include rich Markdown formatting in code cell outputs to highlight key parameters, metrics, performance alerts, and warnings with colors and bold styling.
  - Recommended utility:
    ```python
    from IPython.display import Markdown, display

    def printmd(string, color=None):
        colorstr = f"<span style='color:{color}'>{string}</span>" if color else string
        display(Markdown(colorstr))

    # Example:
    printmd("**Training complete — Final Val Loss: 0.142**", color="green")
    ```
- **Visual Progress with `tqdm`**:
  - Always wrap long loops, downloads, or batch iterations with `tqdm.auto.tqdm` to display live visual progress bars.
- **Plot Quality & Labeling**:
  - Every chart must have explicit titles, axis labels with units, clear legends, and readable font sizes.

---

## 4. Kernel, Environment & Package Management
- **Kernel vs. Shell Environments**:
  - Remember: running Jupyter from an environment does **NOT** mean that shell execution via `!` shares the exact same environment variables.
- **Reliable Package Installation**:
  - If `!pip install foo` fails to install cleanly into the active kernel, use:
    ```bash
    !{sys.executable} -m pip install -q foo
    ```
    (`sys.executable` guarantees targeting the Python interpreter running the current kernel).
- **Consolidated Setup Cell**:
  - Consolidate all installation commands in a single setup cell at the top of the notebook, because Colab virtual machines reset on session disconnection.
- **Pin Dependencies & Record Versions**:
  - Pin package versions (`!pip install package==x.y.z`) or install from a verified `requirements.txt`.
  - Log Python version and installed package versions with `!pip freeze` or explicit version prints for core libraries.
- **Format & Lint with Ruff**:
  - Enforce clean Python formatting and linting using `ruff` (`!ruff check .` / `!ruff format .`).
- **Secret Management**:
  - Never commit API keys, tokens, or credentials in plain text. Use Colab Secrets (`from google.colab import userdata; key = userdata.get('KEY')`) or environment variables.

---

## 5. Debugging & Error Handling
- **IPython Debugger (`%debug`)**:
  - When a cell encounters an uncaught exception, suggest the user run `%debug` in a new cell to launch the interactive IPython debugger at the point of failure ([IPython Debugger Documentation](http://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-debug)).
- **Programmatic Debugger Checkpoints (`set_trace`)**:
  - For step-by-step debugging inside functions or training loops, insert:
    ```python
    from IPython.core.debugger import set_trace
    set_trace()
    ```
    This provides an interactive breakpoint directly within the notebook, similar to `pdb` in an IDE.

---

## 6. Reproducibility & Early Validation
- **Seed Setting**: Set random seeds across all libraries at the top of the notebook:
  ```python
  import random, os, numpy as np, torch
  seed = 42
  random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  np.random.seed(seed)
  if torch.cuda.is_available():
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
  ```
- **Early Sanity Checks & Contract Validation**:
  - Validate data assumptions immediately after loading.
  - Add quick assertion checks (e.g. `assert len(df) > 0`, shape checks, dtypes verification, and zero unexpected NaNs).
- **Sample-First Development**:
  - Develop and test pipelines on a small representative sample first (e.g. `df.sample(1000, random_state=42)`), then run against the full dataset once logic is verified.

---

## 7. Compute, Memory & Session Management
- **Deliberate Hardware Runtime**:
  - Choose the hardware accelerator deliberately (*Runtime* $\rightarrow$ *Change runtime type*). Switch back to CPU for tabular/lightweight tasks when a GPU is unnecessary.
  - Verify GPU status with `!nvidia-smi` and PyTorch checks.
- **Explicit Memory Reclamation**:
  - Proactively free large memory objects when done:
    ```python
    import gc
    del large_intermediate_df
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ```
- **Disk Caching for Expensive Operations**:
  - Cache heavy computations or intermediate datasets to disk (e.g. Parquet for dataframes, `.npy` for matrices, `.pt` for tensors) rather than re-computing.

---

## 8. Persistence & Resumability (Colab VM Ephemerality)
- **Mount Google Drive for Persistent Storage**:
  - Because Colab VM disk storage is wiped when a session terminates, mount Google Drive for all persistent assets:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
- **Regular Checkpointing**:
  - Save checkpoints, intermediate data, and model weights regularly to Google Drive.
- **Session Limits & Resumable Pipelines**:
  - Never depend on uninterrupted long-running sessions (free tiers have idle timeouts and max duration caps).
  - Ensure training loops and data processing pipelines can cleanly resume from the latest saved checkpoint on Drive.

---

## 9. Subagent Review Checklist

When writing or reviewing notebook cells, verify:
- [ ] Every code cell is preceded by a Markdown cell explaining intent and rationale.
- [ ] Markdown headers build a clear, hierarchical table of contents.
- [ ] Cells are small and focused on a single logical responsibility.
- [ ] Heavy logic is encapsulated in functions and modular `.py` files.
- [ ] Long-running cells have `%%time` and loops use `tqdm`.
- [ ] Paths use `pathlib.Path` from a relative or `BASE_DIR` root.
- [ ] Key parameters and metrics are highlighted using `printmd()`.
- [ ] Setup cell consolidates pinned `pip` installs via `!{sys.executable} -m pip install`.
- [ ] Seeds are set at the top for reproducibility.
- [ ] Early sanity assertions validate shapes, dtypes, and null counts.
- [ ] Memory is managed via `del`, `gc.collect()`, and `torch.cuda.empty_cache()`.
- [ ] Drive is mounted for persistence, and training is checkpointed/resumable.
- [ ] No hardcoded secrets; credentials use environment variables or Colab Secrets.
- [ ] Notebook finishes with a conclusions, caveats, and next steps summary.
