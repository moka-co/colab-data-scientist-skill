#!/usr/bin/env python3
"""Assemble a Google Colab nbformat v4 .ipynb file from get_cells JSON and run_code_cell output files."""

import argparse
import json
from pathlib import Path


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_source(source) -> list[str]:
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


def build_notebook(cells_data: dict, output_map: dict[str, str], notebook_name: str) -> dict:
    raw_cells = cells_data.get("cells", cells_data if isinstance(cells_data, list) else [])
    nb_cells = []
    exec_counter = 1

    for cell in raw_cells:
        cell_type = cell.get("cell_type", "code")
        cell_id = cell.get("id", "")
        source_lines = normalize_source(cell.get("source", []))

        if cell_type == "markdown":
            nb_cells.append({
                "cell_type": "markdown",
                "metadata": {"id": cell_id} if cell_id else {},
                "source": source_lines,
            })
        else:
            outputs = cell.get("outputs", [])
            if cell_id in output_map:
                step_out = load_json(output_map[cell_id])
                outputs = step_out.get("outputs", outputs)

            nb_cells.append({
                "cell_type": "code",
                "execution_count": cell.get("execution_count") or exec_counter,
                "metadata": {"id": cell_id} if cell_id else {},
                "outputs": outputs,
                "source": source_lines,
            })
            exec_counter += 1

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
    parser = argparse.ArgumentParser(description="Export Colab MCP cells and outputs to a valid .ipynb file.")
    parser.add_argument("--cells-json", required=True, help="Path to JSON file containing get_cells output.")
    parser.add_argument("--out", required=True, help="Destination .ipynb file path.")
    parser.add_argument("--name", default="Notebook.ipynb", help="Notebook display name in metadata.")
    parser.add_argument(
        "--cell-output",
        action="append",
        default=[],
        metavar="CELL_ID=PATH",
        help="Map a cellId to its saved run_code_cell output JSON file (can be passed multiple times).",
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
