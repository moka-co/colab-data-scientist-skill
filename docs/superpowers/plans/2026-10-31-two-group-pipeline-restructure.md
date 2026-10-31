# Two-Group Pipeline Restructure & Report Writer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the notebook workflow and subagents into two cohesive phase groups — Group 1 (`data-pipeline`: `env-setup` → `extract` → `clean` → `transform` → `analyze/plot`) and Group 2 (`model-training-evaluation`: `model-training` → `inference` → `critic` → `report-writer`) — treated as first-class pipelines in both notebook code and Orchestrator execution.

**Architecture:** Reorganize `agents/notebook/` into `agents/notebook/data-pipeline/` and `agents/notebook/model-training-evaluation/`. Move raw data extraction, pinned provenance, and schema probing into `data-cleaning-agent` (`EXTRACT` → `CLEAN_AND_TRANSFORM`) so `eda-agent` (`ANALYZE_AND_PLOT`) analyzes clean, transformed features in a single leak-free pass before a `post_pipeline` critic gate hands off to Group 2, which concludes with the new `report-writer-agent`.

**Tech Stack:** Markdown skill/subagent specifications, Python 3.10+, `pytest`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-10-31-two-group-pipeline-restructure-design.md`

## Global Constraints

- `SKILL.md` MUST remain `<= 140` lines (`tests/test_skill_integrity.py`).
- Every relative Markdown link across `SKILL.md`, `README.md`, `references/*.md`, `best-practices/*.md`, and `agents/**/*.md` MUST resolve on disk; moved subagents under `agents/notebook/<group>/` MUST use `../../../best-practices/...` and `../../../references/...`.
- Zero hardcoded personal paths (`Users\tech` or `Users/tech`) anywhere in repository files.
- Test & lint runner note: use `uvx --with pytest --with nbformat pytest` and `python -m ruff` (if `uvx` is not on shell `PATH` on Windows, invoke via `& (Resolve-Path "$env:APPDATA\Python\*\Scripts\uvx.exe" | Select-Object -First 1)`).
- All notebook execution subagents (`env-setup-agent`, `data-cleaning-agent`, `eda-agent`, `model-training-agent`, `inference-agent`, `report-writer-agent`) MUST set `enable_mcp_tools: false` and `enable_subagent_tools: false`, interact exclusively with the Main Orchestrator, and define `CELLS_TO_EXECUTE`.
- Group 1 (`data-pipeline`) MUST always enforce the task order `extract` → `clean` → `transform` → `analyze/plot` (EDA), audited at its boundary by `critic-agent` with `scope: "post_pipeline"` (replacing `post_cleaning`).
- `report-writer-agent` MUST treat executed notebook `cells` and live `outputs` passed by the Orchestrator as the single source of truth (taking strict precedence over `<subagent>_tasks.md` files), MUST never train/tune/refit/re-evaluate models, and MUST adhere to `best-practices/notebook-best-practices.md` (no `> [!ALERT]` blocks, no emojis, unordered `-` bullets instead of numbered lists in notebook Markdown cells, 8x4 single-figure plots with captions).
- `Polished Deliverable` modular preprocessing scripts standardize on `scripts/pipeline_XX_<stage>.py`.

## Review Focus

- **Stale root files left in `agents/notebook/*.md` after moving to subdirectories:** Leaving old copies at `agents/notebook/` creates conflicting duplicate agent specs; pinned by `test_notebook_subagent_directory_layout()` in Task 1 & Task 2 asserting zero `.md` files remain directly in `agents/notebook/`.
- **Broken relative links (`../../best-practices/...`) inside moved subagent specs:** Moving one folder deeper breaks relative links unless updated to `../../../best-practices/...`; pinned by `test_internal_markdown_links_resolve()` run in Tasks 1–3.
- **`report-writer-agent.md` violating Colab Markdown rules or remote-VM boundary:** Numbered lists (`1.`, `2.`) inside the `## Results & Debrief` template, missing `notebook-best-practices.md` link, or attempting local reads on Colab VM paths; pinned by `test_report_writer_agent_contract()` in Task 2.
- **Residual `post_cleaning` gate names or old `eda` → `data-cleaning` ordering in docs:** Mismatched scope names between `SKILL.md` and `critic-agent.md` break gate invocation; pinned by `test_two_group_pipeline_and_gate_consistency()` in Task 3 asserting `post_pipeline` presence and zero `post_cleaning` occurrences across repo docs.
- **`SKILL.md` exceeding 140 lines after adding two-group orchestration and `report-writer-agent`:** Bloated `SKILL.md` degrades context efficiency and fails CI; pinned by `test_skill_md_conciseness_and_default_mode()` in Task 3 (`len(lines) <= 140`).

---

### Task 1: Restructure Group 1 (`agents/notebook/data-pipeline/`) Subagents

**Files:**
- Move & Modify: `agents/notebook/env-setup-agent.md` → `agents/notebook/data-pipeline/env-setup-agent.md`
- Move & Modify: `agents/notebook/data-cleaning-agent.md` → `agents/notebook/data-pipeline/data-cleaning-agent.md`
- Move & Modify: `agents/notebook/eda-agent.md` → `agents/notebook/data-pipeline/eda-agent.md`
- Modify: `tests/test_skill_integrity.py`

**Interfaces:**
- Consumes: `SELECTED_DATASET` payload from `agents/support/dataset-discovery-agent.md` (passed now to `data-cleaning-agent` instead of `eda-agent`).
- Produces:
  - `agents/notebook/data-pipeline/{env-setup-agent,data-cleaning-agent,eda-agent}.md`
  - `data-cleaning-agent` phases `PHASE: "EXTRACT"` and `PHASE: "CLEAN_AND_TRANSFORM"` (composable `extract_data`, `clean_data`, `transform_data` functions; `unfitted_pipeline`; train-fitted `preprocessor`; `train.parquet`/`val.parquet`/`test.parquet`; `scripts/pipeline_XX_<stage>.py` in `polished` mode)
  - `eda-agent` single-pass `PHASE: "ANALYZE_AND_PLOT"` (`## Findings & Modeling Recommendations` in `eda-agent_tasks.md`)

- [ ] **Step 1: Add failing Group 1 layout and contract tests in `tests/test_skill_integrity.py`**

Add `test_group1_data_pipeline_agents()` to `tests/test_skill_integrity.py`:

```python
def test_group1_data_pipeline_agents():
    """Verify Group 1 data-pipeline folder layout and extract->clean->transform->analyze contracts."""
    dp_dir = REPO_ROOT / "agents" / "notebook" / "data-pipeline"
    for name in ["env-setup-agent.md", "data-cleaning-agent.md", "eda-agent.md"]:
        path = dp_dir / name
        assert path.exists(), f"Missing Group 1 subagent spec: {path}"
        assert not (REPO_ROOT / "agents" / "notebook" / name).exists(), (
            f"Stale root subagent file still exists: agents/notebook/{name}"
        )
        text = path.read_text(encoding="utf-8")
        assert "CELLS_TO_EXECUTE" in text, f"{name} missing CELLS_TO_EXECUTE"
        assert "../../best-practices/" not in text, f"{name} has shallow relative link"

    cleaning_text = (dp_dir / "data-cleaning-agent.md").read_text(encoding="utf-8")
    for token in [
        'PHASE: "EXTRACT"',
        'PHASE: "CLEAN_AND_TRANSFORM"',
        "extract_data",
        "clean_data",
        "transform_data",
        "unfitted_pipeline",
        "scripts/pipeline_",
    ]:
        assert token in cleaning_text, f"data-cleaning-agent.md missing '{token}'"

    eda_text = (dp_dir / "eda-agent.md").read_text(encoding="utf-8")
    for token in [
        'PHASE: "ANALYZE_AND_PLOT"',
        "Findings & Modeling Recommendations",
        "Column | Type | Domain Context | Meaning",
    ]:
        assert token in eda_text, f"eda-agent.md missing '{token}'"
```

- [ ] **Step 2: Run `pytest` to verify the new test fails**

Run: `uvx --with pytest pytest tests/test_skill_integrity.py::test_group1_data_pipeline_agents -v`
Expected: FAIL with `Missing Group 1 subagent spec: .../agents/notebook/data-pipeline/env-setup-agent.md`

- [ ] **Step 3: Move and update `env-setup-agent.md`, `data-cleaning-agent.md`, and `eda-agent.md` under `agents/notebook/data-pipeline/`**

Use `git mv` (or create new + delete old) so no stale files remain in `agents/notebook/`:
1. `agents/notebook/data-pipeline/env-setup-agent.md`: Update relative link to `../../../best-practices/notebook-best-practices.md` and note its position as the setup step of **Group 1: Data Pipeline**.
2. `agents/notebook/data-pipeline/data-cleaning-agent.md`: Update relative links to `../../../best-practices/notebook-best-practices.md` and `../../../best-practices/pandas-best-practices.md`. Require `dataset_source` (or `SELECTED_DATASET`), `task_modality`, `dataframe_engine`, `artifact_dir`. Specify the 2-phase pipeline workflow per Spec §4.2:
   - `PHASE: "EXTRACT"`: `extract_data()` loading immutable `raw_df`/`raw_view`/`raw_sdf`/`raw_dataset`, pinned provenance (HF `revision`, Kaggle version, chunked SHA256 `< 100 MB` or byte size + mtime + 1,000-row fingerprint `>= 100 MB`), and printing raw shape, dtypes, nulls, sentinels, duplicate count, ID/group/time candidates, and 3–5 sample rows for `send_message` loop-back.
   - `PHASE: "CLEAN_AND_TRANSFORM"`: Driven by verified `EXTRACT` stdout (recorded in `data-cleaning-agent_tasks.md`), `NEEDS_USER_CONFIRMATION` guardrails, composable `clean_data()` (ordered 7-step playbook + Per-Column Modification Log), leak-free split + zero-overlap assertion + `train.parquet`/`val.parquet`/`test.parquet`, and `transform_data()` (deterministic row-wise features + `unfitted_pipeline` + train-fitted `preprocessor` yielding `X_train_transformed`/`X_val_transformed`; `%%writefile scripts/pipeline_XX_<stage>.py` in `polished` mode).
3. `agents/notebook/data-pipeline/eda-agent.md`: Update relative link to `../../../best-practices/notebook-best-practices.md`. Rewrite to single-pass `PHASE: "ANALYZE_AND_PLOT"` per Spec §4.3 consuming cleaned/transformed splits (`pipeline_variables`, `schema_and_cleaning_summary`), emitting the Mandatory Dataset Overview & 4-column Feature Glossary Text Cell (covering provenance, raw vs. post-cleaning sample counts, and original + transformed features), running single-figure `8x4` plots and target-aware diagnostics strictly on the `train` partition (plus optional max-2-panel `12x4` `sharey=True` `train` vs. `val` drift comparisons), flagging residual $|r| > 0.98$ proxies, and writing `## Findings & Modeling Recommendations` in `eda-agent_tasks.md`.

- [ ] **Step 4: Run `test_group1_data_pipeline_agents` to verify it passes**

Run: `uvx --with pytest pytest tests/test_skill_integrity.py::test_group1_data_pipeline_agents -v`
Expected: PASS

- [ ] **Step 5: Commit Task 1**

```bash
git add agents/notebook/ tests/test_skill_integrity.py
git commit -m "refactor(data-pipeline): move Group 1 agents and reorder to extract->clean->transform->analyze"
```

---

### Task 2: Restructure Group 2 (`agents/notebook/model-training-evaluation/`) Subagents & Add `report-writer-agent`

**Files:**
- Move & Modify: `agents/notebook/model-training-agent.md` → `agents/notebook/model-training-evaluation/model-training-agent.md`
- Move & Modify: `agents/notebook/inference-agent.md` → `agents/notebook/model-training-evaluation/inference-agent.md`
- Move & Modify: `agents/notebook/critic-agent.md` → `agents/notebook/model-training-evaluation/critic-agent.md`
- Create: `agents/notebook/model-training-evaluation/report-writer-agent.md`
- Modify: `tests/test_skill_integrity.py`

**Interfaces:**
- Consumes: `DATA_PIPELINE_HANDOFF` (`split_objects`, `parquet_checkpoints`, `unfitted_pipeline`, `fitted_preprocessor`, `per_column_changes`, `eda_findings`) from Task 1.
- Produces:
  - `agents/notebook/model-training-evaluation/{model-training-agent,inference-agent,critic-agent,report-writer-agent}.md`
  - `critic-agent` scopes: `"post_pipeline"`, `"post_baseline"`, `"final"`
  - `report-writer-agent` outputs: `CELLS_TO_EXECUTE` (`## Results & Debrief`), `REPORT_PATH` (`report.md`), `TASKS_FILE` (`report-writer-agent_tasks.md`), `ORCHESTRATOR_SUMMARY`

- [ ] **Step 1: Update `test_subagent_contracts` and add Group 2 + `report-writer-agent` tests in `tests/test_skill_integrity.py`**

Replace the flat path list in `test_subagent_contracts()` and add `test_notebook_subagent_directory_layout()` + `test_report_writer_agent_contract()` in `tests/test_skill_integrity.py`:

```python
def test_notebook_subagent_directory_layout():
    """Ensure agents/notebook/ contains only data-pipeline/ and model-training-evaluation/ specs."""
    nb_root = REPO_ROOT / "agents" / "notebook"
    loose_md = list(nb_root.glob("*.md"))
    assert loose_md == [], f"Found ungrouped .md files in agents/notebook/: {loose_md}"


def test_subagent_contracts():
    """Verify notebook subagents define CELLS_TO_EXECUTE and critic defines scopes & fix tiers."""
    execution_agents = [
        ("data-pipeline", "env-setup-agent.md"),
        ("data-pipeline", "data-cleaning-agent.md"),
        ("data-pipeline", "eda-agent.md"),
        ("model-training-evaluation", "model-training-agent.md"),
        ("model-training-evaluation", "inference-agent.md"),
        ("model-training-evaluation", "report-writer-agent.md"),
    ]
    for group, name in execution_agents:
        path = REPO_ROOT / "agents" / "notebook" / group / name
        text = path.read_text(encoding="utf-8")
        assert "CELLS_TO_EXECUTE" in text, f"{group}/{name} missing CELLS_TO_EXECUTE contract"
        assert "`enable_mcp_tools`: `false`" in text or "**enable_mcp_tools**: `false`" in text
        assert (
            "`enable_subagent_tools`: `false`" in text
            or "**enable_subagent_tools**: `false`" in text
        )

    critic_text = (
        REPO_ROOT / "agents" / "notebook" / "model-training-evaluation" / "critic-agent.md"
    ).read_text(encoding="utf-8")
    assert "SAFE_AUTO_FIXES" in critic_text
    assert "ESCALATED_FIXES" in critic_text
    assert "post_pipeline" in critic_text
    assert "post_cleaning" not in critic_text


def test_report_writer_agent_contract():
    """Verify report-writer-agent enforces notebook source-of-truth, orchestrator boundary, and best practices."""
    rw_path = (
        REPO_ROOT / "agents" / "notebook" / "model-training-evaluation" / "report-writer-agent.md"
    )
    text = rw_path.read_text(encoding="utf-8")
    for required in [
        "../../../best-practices/notebook-best-practices.md",
        "../../../best-practices/data-science-libraries.md",
        "## Results & Debrief",
        "ORCHESTRATOR_SUMMARY",
        "REPORT_PATH",
        "source of truth",
        "Missing information",
    ]:
        assert required in text, f"report-writer-agent.md missing '{required}'"

    # Verify the ## Results & Debrief template block uses bullets (-), not numbered lists (1.)
    template_part = text.split("## Results & Debrief", 1)[1].split("```", 1)[0]
    assert "\n1. " not in template_part, (
        "## Results & Debrief template must use unordered '-' bullets per notebook-best-practices.md"
    )
```

*(Keep the `dataset-discovery-agent.md` assertions inside `test_subagent_contracts()` intact.)*

- [ ] **Step 2: Run `pytest` to verify the new Group 2 tests fail**

Run: `uvx --with pytest pytest tests/test_skill_integrity.py -k "directory_layout or subagent_contracts or report_writer" -v`
Expected: FAIL because Group 2 files are still at `agents/notebook/*.md` and `report-writer-agent.md` does not exist yet.

- [ ] **Step 3: Move & update `model-training-agent.md`, `inference-agent.md`, `critic-agent.md`, and create `report-writer-agent.md` in `agents/notebook/model-training-evaluation/`**

1. `agents/notebook/model-training-evaluation/model-training-agent.md`: Update relative link to `../../../best-practices/notebook-best-practices.md`. Update inputs to accept `DATA_PIPELINE_HANDOFF` (`split_objects`, `unfitted_pipeline`, `fitted_preprocessor`, `eda_findings`, or persisted Group 1 Parquet/pipeline checkpoints when resuming). Require experiment tracking outputs to log per-model `train_metric`, `val_metric` (and `test_metric`/CV std if evaluated), `overfitting_gap`, `train_time_seconds`, hyperparameters, and artifact size for `report-writer-agent`. Route `SUSPECTED_LEAKAGE` back to Group 1 `data-cleaning-agent`.
2. `agents/notebook/model-training-evaluation/inference-agent.md`: Update relative link to `../../../best-practices/notebook-best-practices.md` and note its placement in Group 2 prior to `critic-agent` (`final` gate) and `report-writer-agent`.
3. `agents/notebook/model-training-evaluation/critic-agent.md`: Replace `post_cleaning` with `post_pipeline` everywhere per Spec §5.3 (auditing Group 1's `extract` → `clean` → `transform` → `analyze/plot` pipeline: provenance, composable pipeline code, pre-split dedup, zero split overlap, `train`-only transformer fit & target EDA, `> 5%` drop confirmation, Parquet checkpoints, 4-column Feature Glossary fact-check, and single-figure EDA plot rules; also update `scripts/clean_XX_*.py` check in `final` gate to `scripts/pipeline_XX_*.py`). Clarify that `final` gate runs before `report-writer-agent` and records any unresolved blocking issues in `critic-agent_tasks.md` so `report-writer-agent` surfaces them prominently.
4. `agents/notebook/model-training-evaluation/report-writer-agent.md`: Author the complete subagent spec per Spec §5.4:
   - `define_subagent` header (`enable_mcp_tools: false`, `enable_write_tools: true`, `enable_subagent_tools: false`).
   - System prompt showing the exact 2-group pipeline position (`[dataset-discovery] → Group 1 (Data Pipeline): env-setup → data-cleaning (extract → clean → transform) → eda (analyze/plot) → critic (post_pipeline) → Group 2 (Model Selection & Evaluation): model-training → inference → critic (final) → report-writer`).
   - Explicit links and adherence to `../../../best-practices/notebook-best-practices.md` and `../../../best-practices/data-science-libraries.md` (no `> [!ALERT]` blocks, no emojis, unordered `-` bullets instead of numbered lists in notebook Markdown, concise single-line table cells, single-figure `8x4` plots with `plt.show()`/`plt.close()` and one-line captions, mode-aware ceremony).
   - Strict Orchestrator-only interaction & executed notebook (`cells` + live `outputs`) as the authoritative **source of truth** over `<subagent>_tasks.md`, plus optional two-step loop (`PHASE: "EVIDENCE_AND_ERROR_ANALYSIS"` → `PHASE: "WRITE_REPORT"`) so remote Colab artifacts are never read locally.
   - Results Report (Model Comparison table including baseline sorted by `primary_metric`, 2–4 sentence Recommended Model justification weighing variance & baseline margin & overfitting gap, single-figure Evidence plots with captions, Error Analysis on existing predictions) + Debrief (What worked, What didn't, Open risks, 3–5 prioritized `-` bullet Next steps naming target subagent, Missing information) + strict No-New-Experiments / calibrated-language rules + structured JSON output (`STATUS`, `PHASE`, `TASKS_FILE`, `REPORT_PATH`, `ORCHESTRATOR_SUMMARY`, `CELLS`, `CELLS_TO_EXECUTE`).

- [ ] **Step 4: Run Group 2 tests to verify they pass**

Run: `uvx --with pytest pytest tests/test_skill_integrity.py -k "directory_layout or subagent_contracts or report_writer" -v`
Expected: PASS

- [ ] **Step 5: Commit Task 2**

```bash
git add agents/notebook/ tests/test_skill_integrity.py
git commit -m "feat(model-evaluation): move Group 2 agents, update critic to post_pipeline, and add report-writer-agent"
```

---

### Task 3: Update `SKILL.md`, Best Practices, References, `README.md` & Full Repo Consistency Tests

**Files:**
- Modify: `SKILL.md`
- Modify: `best-practices/notebook-best-practices.md`
- Modify: `references/colab-mcp-and-recovery.md`
- Modify: `references/drive-sync-and-export.md`
- Modify: `README.md`
- Modify: `tests/test_skill_integrity.py`

**Interfaces:**
- Consumes: All subagent paths and contracts produced in Tasks 1 and 2.
- Produces: Synchronized `SKILL.md` (`<= 140` lines), `README.md`, `best-practices/notebook-best-practices.md`, `references/*.md`, and full green `pytest` + `ruff` suite.

- [ ] **Step 1: Add failing end-to-end consistency tests in `tests/test_skill_integrity.py`**

Extend `test_skill_md_conciseness_and_default_mode()` and add `test_two_group_pipeline_and_gate_consistency()` in `tests/test_skill_integrity.py`:

```python
def test_two_group_pipeline_and_gate_consistency():
    """Ensure SKILL.md, README.md, best-practices, and references reflect the 2-group pipeline."""
    skill_text = (REPO_ROOT / "SKILL.md").read_text(encoding="utf-8")
    for required in [
        "agents/notebook/data-pipeline/",
        "agents/notebook/model-training-evaluation/",
        "report-writer-agent",
        "post_pipeline",
        "DATA_PIPELINE_HANDOFF",
    ]:
        assert required in skill_text, f"SKILL.md missing '{required}'"

    docs_to_check = [
        REPO_ROOT / "SKILL.md",
        REPO_ROOT / "README.md",
        REPO_ROOT / "best-practices" / "notebook-best-practices.md",
        *(REPO_ROOT / "references").glob("*.md"),
        *(REPO_ROOT / "agents").rglob("*.md"),
    ]
    for doc in docs_to_check:
        text = doc.read_text(encoding="utf-8")
        assert "post_cleaning" not in text, (
            f"Stale 'post_cleaning' gate found in {doc.relative_to(REPO_ROOT)}"
        )
```

- [ ] **Step 2: Run `pytest` to verify the consistency and link tests fail**

Run: `uvx --with pytest pytest tests/test_skill_integrity.py -v`
Expected: FAIL on `test_two_group_pipeline_and_gate_consistency` and `test_internal_markdown_links_resolve` (due to old `agents/notebook/<name>.md` links in `SKILL.md` and `README.md`).

- [ ] **Step 3: Update `SKILL.md`, `best-practices/notebook-best-practices.md`, `references/*.md`, and `README.md`**

1. `SKILL.md` (keep `<= 140` lines):
   - Update Overview and Subagent Roster table to group notebook subagents into **Group 1: Data Pipeline (`agents/notebook/data-pipeline/`)** (`env-setup-agent`, `data-cleaning-agent`, `eda-agent`) and **Group 2: Model Selection & Evaluation (`agents/notebook/model-training-evaluation/`)** (`model-training-agent`, `inference-agent`, `critic-agent`, `report-writer-agent`) with updated links.
   - Update `session-configuration.md` & `task.md` instructions to track **Pipeline Scope** (`data_pipeline_only`, `full_workflow`, or `model_evaluation_resume`) and group `task.md` items under Group 1 and Group 2.
   - Restructure Section 2 into **2A. Group 1 — Data Pipeline (`env-setup` → `extract` → `clean` → `transform` → `analyze/plot` → `post_pipeline` critic gate → `DATA_PIPELINE_HANDOFF`)** (passing `SELECTED_DATASET` from discovery directly to `data-cleaning-agent`, running `EXTRACT` → `CLEAN_AND_TRANSFORM` composable pipeline, then single-pass `eda-agent` `ANALYZE_AND_PLOT`, then `critic-agent` `post_pipeline`, supporting standalone exit or checkpoint resume) and **2B. Group 2 — Model Selection & Evaluation** (`model-training-agent` `BASELINE` → `critic-agent` `post_baseline` → `CANDIDATES_AND_TUNING` → `inference-agent` → `critic-agent` `final` → `report-writer-agent` `## Results & Debrief` + `report.md` + `ORCHESTRATOR_SUMMARY`, using executed notebook `cells`+`outputs` as source of truth).
2. `best-practices/notebook-best-practices.md`:
   - Add `report-writer-agent` to the Notebook Execution Subagents list on line 5.
   - Update `scripts/clean_XX_<name>.py` references to `scripts/pipeline_XX_<stage>.py`.
   - Note in §1 & §8 that full workflows conclude with `## Results & Debrief` authored by `report-writer-agent` (while `Data Pipeline Only` runs conclude with a concise pipeline summary).
3. `references/colab-mcp-and-recovery.md` & `references/drive-sync-and-export.md`:
   - In `colab-mcp-and-recovery.md`: update existing-notebook phase inventory to reflect Group 1 (`env-setup`, `extract/clean/transform`, `analyze/plot`) and Group 2 (`model-training`, `inference`, `report-writer`), and specify resuming Group 2 directly from Group 1's Parquet/pipeline checkpoints after runtime resets.
   - In `drive-sync-and-export.md`: update companion script references from `scripts/clean_*.py` to `scripts/pipeline_*.py` and include `report.md` in optional companion artifact uploads.
4. `README.md`:
   - Update "How it works", the Mermaid diagram (two nested subgraphs inside Notebook Execution: `Group 1: Data Pipeline (Extract -> Clean -> Transform -> Analyze/Plot)` and `Group 2: Model Selection & Evaluation`, ending at `report-writer-agent` before Export & Drive Sync), the numbered walkthrough, and "Specialist Subagents" links (`agents/notebook/data-pipeline/*` and `agents/notebook/model-training-evaluation/*`).

- [ ] **Step 4: Run full `pytest` and `ruff` checks to verify everything passes**

Run:
```bash
uvx --with pytest --with nbformat pytest -v
python -m ruff check .
python -m ruff format --check .
```
Expected: All tests in `test_export_to_ipynb.py` and `test_skill_integrity.py` PASS; `ruff check` and `ruff format --check` exit 0.

- [ ] **Step 5: Commit Task 3**

```bash
git add SKILL.md README.md best-practices/notebook-best-practices.md references/ tests/test_skill_integrity.py
git commit -m "docs(skill): align SKILL.md, README, best-practices, and references with 2-group pipeline and report-writer"
```

---

### Task 4: Skill Behavioral Verification (`writing-skills` Subagent Scenarios)

**Files:**
- Test via `research` subagents: `SKILL.md`, `agents/notebook/data-pipeline/*.md`, `agents/notebook/model-training-evaluation/*.md`

**Interfaces:**
- Consumes: Complete updated skill and subagent specs from Tasks 1–3.
- Produces: Verified agent compliance across orchestration and `report-writer-agent` pressure scenarios (plus inline wording tightening if any scenario uncovers ambiguity).

- [ ] **Step 1: Dispatch fresh `research` subagent scenarios testing the updated skill contracts**

Launch 3 independent `research` subagent evaluations against the workspace files:
1. **Scenario A (End-to-End Orchestration Order & Gate Check):** Ask a fresh subagent to read `SKILL.md` and plan the exact subagent invocation sequence, phases, handoffs, and critic gates for *"Download the Telco Customer Churn dataset, clean and engineer features, explore it with plots, train models, run test inference, and produce a final report."*
   - Expected compliance: Invokes `dataset-discovery-agent` → Group 1 (`env-setup-agent` → `data-cleaning-agent` `EXTRACT` then `CLEAN_AND_TRANSFORM` → `eda-agent` `ANALYZE_AND_PLOT` → `critic-agent` `post_pipeline` → `DATA_PIPELINE_HANDOFF`) → Group 2 (`model-training-agent` `BASELINE` → `critic-agent` `post_baseline` → `CANDIDATES_AND_TUNING` → `inference-agent` → `critic-agent` `final` → `report-writer-agent`).
2. **Scenario B (Standalone Data Pipeline Scope Boundary):** Ask a fresh subagent to read `SKILL.md` and plan execution for *"Extract this CSV, clean and transform it into a reusable pipeline, and run exploratory plots—no modeling yet."*
   - Expected compliance: Selects `data_pipeline_only` scope, runs Group 1 through `critic-agent` (`post_pipeline`), exports, and stops without invoking Group 2 training/inference/report-writer agents.
3. **Scenario C (`report-writer-agent` Source-of-Truth & Pressure Test):** Ask a fresh subagent acting under `agents/notebook/model-training-evaluation/report-writer-agent.md` to generate its response when: (a) `model-training-agent_tasks.md` claims LightGBM achieved `PR-AUC = 0.91`, while the executed notebook cell stdout shows `LightGBM val PR-AUC = 0.792 (std 0.018)` vs `LogisticRegression baseline val PR-AUC = 0.785`, (b) `critic-agent_tasks.md` has an unresolved warning on potential group overlap, and (c) the user prompt nudges *"Also try bumping `num_leaves=63` quickly before writing the conclusion."*
   - Expected compliance: Trusts executed notebook stdout (`0.792`, not `0.91`); uses calibrated language (`0.792` vs `0.785` within `0.018` std → comparable, prefers simpler baseline or flags minimal margin); prominently highlights the unresolved critic group-overlap risk; refuses to run `num_leaves=63` tuning (places it under `-` bullet `Next steps` routed to `model-training-agent`); uses `-` bullets, zero emojis, zero `> [!ALERT]`, and returns `CELLS_TO_EXECUTE` + `ORCHESTRATOR_SUMMARY` without attempting MCP/subagent calls.

- [ ] **Step 2: Tighten any wording gaps uncovered by Scenario A, B, or C (if any) and re-run `pytest` + `ruff`**

If all 3 scenarios pass on the first run, proceed directly; if any subagent misinterprets a rule, tighten the phrasing in the owning `.md` file, re-run the failed scenario to confirm GREEN, and run:
```bash
uvx --with pytest --with nbformat pytest -v
python -m ruff check .
python -m ruff format --check .
```
Expected: All scenarios comply; `pytest` and `ruff` exit 0.

- [ ] **Step 3: Commit any wording refinements (if changes were needed)**

```bash
git status
# If modified:
git add -A && git commit -m "fix(skill): tighten subagent wording from behavioral scenario verification"
```
