#!/usr/bin/env python3
"""Assemble a Google Colab nbformat v4 .ipynb file from get_cells and output files."""

import argparse
import json
import re
from pathlib import Path

# Regex to strip ANSI escape codes (colors, cursor movements, etc.)
ANSI_ESCAPE_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def load_json(path: str) -> dict:
    """Safely load JSON data from a given path."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def clean_output_text(text: str) -> str:
    """Clean terminal-specific escape sequences and handle carriage returns cleanly."""
    if not isinstance(text, str):
        return text

    # Remove ANSI escape sequences
    text = ANSI_ESCAPE_PATTERN.sub("", text)

    # Process carriage returns within lines (common in tqdm/progress bars)
    lines = text.split("\n")
    processed_lines = []
    for line in lines:
        if "\r" in line:
            # Keep the final state of the line after carriage returns
            segments = line.split("\r")
            if segments[-1]:
                final_segment = segments[-1]
            elif len(segments) > 1:
                final_segment = segments[-2]
            else:
                final_segment = ""
            processed_lines.append(final_segment)
        else:
            processed_lines.append(line)

    return "\n".join(processed_lines)


def sanitize_outputs(outputs: list) -> list:
    """Sanitize and normalize output objects (stream, execute_result, error, display_data)."""
    sanitized = []
    for out in outputs:
        if not isinstance(out, dict):
            sanitized.append(out)
            continue

        item = dict(out)
        out_type = item.get("output_type")

        if out_type == "stream":
            text = item.get("text", "")
            if isinstance(text, list):
                item["text"] = [clean_output_text(line) for line in text]
            elif isinstance(text, str):
                item["text"] = clean_output_text(text)
        elif out_type == "error":
            traceback = item.get("traceback", [])
            if isinstance(traceback, list):
                item["traceback"] = [clean_output_text(line) for line in traceback]
        sanitized.append(item)
    return sanitized


def normalize_source(source) -> list[str]:
    """Normalize notebook source into a list of newline-terminated strings."""
    if isinstance(source, list):
        lines = []
        for i, line in enumerate(source):
            if i < len(source) - 1 and not line.endswith("\n"):
                lines.append(line + "\n")
            else:
                lines.append(line)
        return lines
    if isinstance(source, str):
        return source.splitlines(keepends=True)
    return []


def find_matching_output(cell_id: str, code_idx: int, output_map: dict[str, str]) -> str | None:
    """Find matching output file path for a cell, supporting exact, stripped, or index matching."""
    if not output_map:
        return None

    # 1. Exact match
    if cell_id in output_map:
        return output_map[cell_id]

    # 2. Stripped / prefix match (e.g. 'cell_123' vs '123')
    cell_id_clean = cell_id.strip()
    for key, path in output_map.items():
        key_clean = key.strip()
        if (
            key_clean == cell_id_clean
            or key_clean.endswith(cell_id_clean)
            or cell_id_clean.endswith(key_clean)
        ):
            return path

    # 3. Fallback to index-based keys if mapped by index (e.g., '0', '1', 'code_0')
    idx_keys = [str(code_idx), f"code_{code_idx}", f"cell_{code_idx}"]
    for ik in idx_keys:
        if ik in output_map:
            return output_map[ik]

    return None


def build_notebook(cells_data: dict, output_map: dict[str, str], notebook_name: str) -> dict:
    """Assemble valid nbformat v4 dictionary from raw cells and output mapping."""
    raw_cells = cells_data.get("cells", cells_data if isinstance(cells_data, list) else [])
    nb_cells = []
    exec_counter = 1
    code_counter = 0

    for cell in raw_cells:
        cell_type = cell.get("cell_type", "code")
        cell_id = cell.get("id", "")
        source_lines = normalize_source(cell.get("source", []))

        if cell_type == "markdown":
            nb_cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {"id": cell_id} if cell_id else {},
                    "source": source_lines,
                }
            )
        else:
            outputs = cell.get("outputs", [])
            output_file = find_matching_output(cell_id, code_counter, output_map)
            if output_file:
                try:
                    step_out = load_json(output_file)
                    outputs = step_out.get("outputs", outputs)
                except Exception as e:
                    print(f"Warning: Failed to load output file {output_file}: {e}")

            sanitized_outputs = sanitize_outputs(outputs)

            nb_cells.append(
                {
                    "cell_type": "code",
                    "execution_count": cell.get("execution_count") or exec_counter,
                    "metadata": {"id": cell_id} if cell_id else {},
                    "outputs": sanitized_outputs,
                    "source": source_lines,
                }
            )
            exec_counter += 1
            code_counter += 1

    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "name": notebook_name},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": nb_cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Colab MCP cells and outputs to a valid .ipynb file."
    )
    parser.add_argument(
        "--cells-json", required=True, help="Path to JSON file containing get_cells output."
    )
    parser.add_argument("--out", required=True, help="Destination .ipynb file path.")
    parser.add_argument(
        "--name", default="Notebook.ipynb", help="Notebook display name in metadata."
    )
    parser.add_argument(
        "--cell-output",
        action="append",
        default=[],
        metavar="CELL_ID=PATH",
        help="Map cellId to run_code_cell output JSON file (can be passed multiple times).",
    )
    args = parser.parse_args()

    output_map: dict[str, str] = {}
    for item in args.cell_output:
        if "=" not in item:
            raise ValueError(f"Invalid --cell-output '{item}', expected CELL_ID=PATH")
        cid, cpath = item.split("=", 1)
        output_map[cid.strip()] = cpath.strip()

    cells_data = load_json(args.cells_json)
    notebook = build_notebook(cells_data, output_map, args.name)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(notebook['cells'])} cells to {out_path}")


if __name__ == "__main__":
    main()
