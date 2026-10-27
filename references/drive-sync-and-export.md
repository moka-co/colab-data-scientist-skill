# Notebook Export & Google Drive Synchronization Reference

This reference document defines the exact procedures for assembling clean `.ipynb` files (preserving cell outputs without WebSocket truncation) and synchronizing artifacts to Google Drive with explicit, once-per-session user consent.

The Main Orchestrator reads this file when reaching the final export and Google Drive sync step.

---

## 1. Unified Once-Per-Session Google Drive Consent

Uploading files to a user's Google Drive modifies their cloud account and may involve proprietary or sensitive datasets. To avoid both unprompted account side effects and repetitive questions:

1. **Ask Once Per Session (`ask_question`)**:
   - If the user did not already state their Google Drive preference in their initial prompt (or during Step 1 session setup), ask **once** before the first export/sync:
     - `(Recommended) Sync the .ipynb notebook only to Google Drive (Colab Notebooks)`
     - `Sync the .ipynb notebook, modular scripts (scripts/), and logs (DATA_CLEANING_LOG.md)`
     - `Sync everything (notebook, scripts, logs, and data/intermediate/* checkpoints)`
     - `Keep local only (export .ipynb locally without uploading to Google Drive)`
2. **Remember & Honor the Choice**:
   - Record the user's choice in `session-configuration.md` and `task.md` (`drive_sync_policy`) and apply it consistently across the rest of the session and any subsequent edits—never ask contradictory offloading questions mid-pipeline.

---

## 2. Zero-Loss `.ipynb` Export via `scripts/export_to_ipynb.py`

Never call `get_cells(includeOutputs=True)` when plot outputs or large tables exist — *because* base64-encoded PNGs exceed MCP WebSocket payload limits and truncate the response.

Instead:
1. Call `get_cells(includeOutputs=False)` and save the resulting cell structure JSON to `cells.json` in your scratch directory.
2. Collect the `<cellId>=<step_output_path>` pairs recorded from each `run_code_cell` execution during the session.
3. Run `scripts/export_to_ipynb.py` (which strips terminal ANSI escape sequences, normalizes `\r` progress bars, and merges cell outputs into a valid `nbformat` v4 `.ipynb` file):
   ```bash
   python "<skill_dir>/scripts/export_to_ipynb.py" \
     --cells-json "<scratch_dir>/cells.json" \
     --out "<scratch_dir>/<NotebookName>.ipynb" \
     --name "<NotebookName>.ipynb" \
     --cell-output "<cellId_1>=<step_output_path_1>" \
     --cell-output "<cellId_2>=<step_output_path_2>"
   ```

---

## 3. Google Drive Upload, File ID Tracking & Collision Handling (`gdrive` Skill)

When the user's `drive_sync_policy` includes uploading to Google Drive, use the `gdrive` skill (`~/.gemini/config/plugins/gdrive/skills/gdrive/SKILL.md`). Never configure a third-party Google Drive MCP server in `mcp_config.json`.

### 0. Target Google Drive Path Prompting & User Superiority Rule

Before uploading or saving any file to Google Drive, the Main Orchestrator MUST invoke `ask_question` to suggest a destination path and allow the user to provide custom guidance:

1. **Suggest a Path via `ask_question`**:
   - Suggest a sensible default path: e.g. `(Recommended) Colab Notebooks/<NotebookName>.ipynb`.
   - Provide alternative project structure options while inviting the user to provide hints, a subfolder, or a full custom path.
   - Example prompt:
     ```python
     ask_question(
         questions=[{
             "question": "Where on Google Drive would you like to save '<NotebookName>.ipynb'?",
             "options": [
                 "(Recommended) Colab Notebooks/<NotebookName>.ipynb",
                 "Colab Notebooks/Projects/<NotebookName>.ipynb",
             ]
         }]
     )
     ```

2. **Absolute User Superiority Rule**:
   - **What the user says is strictly superior to what the orchestrator suggests.**
   - If the user provides a custom path, hint, or folder name, the Orchestrator MUST prioritize and honor the user's input above any default or suggestion.
   - **Full Path**: If the user enters a full path (e.g. `My Drive/Research/Experiments/Model.ipynb`), parse the folder hierarchy, locate the folder via `gdrive search`, create any missing directories via `gdrive mutate create-folder`, and save the notebook under that exact location.
   - **Hints or Keywords**: If the user gives a hint or keywords (e.g. "put it in my thesis folder" or "save to financial_models"), search Google Drive for matching folders (`gdrive search "name contains '<hint>' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"`) and upload to the matched folder.
   - **Subfolder**: If the user specifies a subfolder (e.g. `Colab Notebooks/DeepLearning`), resolve or create the subfolder inside `Colab Notebooks`.
   - The default `Colab Notebooks` root is only used if the user explicitly approves the recommended option without providing custom hints or paths.

### A. Locating the Target Folder & Checking for Name Collisions
1. Locate the confirmed target folder ID (resolved from the user's choice/hint):
   ```bash
   gdrive search "name = '<Confirmed_Folder_Name>' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
   ```
2. **Check for Existing File With the Same Name Before Uploading**:
   - Query the target folder to see if `<NotebookName>.ipynb` already exists:
     ```bash
     gdrive search "name = '<NotebookName>.ipynb' and '<FOLDER_ID>' in parents and trashed = false"
     ```
   - If a file already exists and was **not** created in the current session, ask the user (`ask_question`) whether to overwrite the existing Drive file in-place (`update <FILE_ID>`) or upload with a versioned/timestamped filename (`<NotebookName>_v2.ipynb`) to prevent accidental data loss.

### B. First Upload vs. Subsequent In-Place Updates
- **First Upload in Session**:
  ```bash
  gdrive mutate upload "<local_ipynb_path>" --mime-type "application/vnd.google.colaboratory" --parent <FOLDER_ID>
  ```
  Record the returned `<FILE_ID>` and `modifiedTime` in `task.md` and `session-configuration.md`.
- **Subsequent Edits in the Same Session**:
  - Before running `mutate update`, optionally verify `gdrive info <FILE_ID>` to ensure the file's remote `modifiedTime` has not advanced due to concurrent manual edits by the user in Google Colab. If the user edited the file concurrently in Drive, re-fetch `get_cells(includeOutputs=False)` first so their manual changes are preserved before exporting and updating:
    ```bash
    gdrive mutate update <FILE_ID> "<local_ipynb_path>" --mime-type "application/vnd.google.colaboratory"
    ```

### C. Syncing Optional Companion Artifacts (When Requested by User's Sync Policy)
- If the user selected syncing scripts/logs or intermediate checkpoints (`Polished Deliverable` mode), upload `DATA_CLEANING_LOG.md`, `scripts/clean_*.py`, and/or `data/intermediate/*` into a dedicated subfolder (`<NotebookName>_artifacts`) inside `Colab Notebooks` so the root `Colab Notebooks` folder stays clean.
