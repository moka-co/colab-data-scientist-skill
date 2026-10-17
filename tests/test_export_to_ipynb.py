import json
from pathlib import Path

from scripts.export_to_ipynb import (
    build_notebook,
    clean_output_text,
    find_matching_output,
    normalize_source,
    sanitize_outputs,
)


def test_clean_output_text_ansi():
    ansi_text = "\x1b[32mSuccess\x1b[0m: Process completed!"
    cleaned = clean_output_text(ansi_text)
    assert cleaned == "Success: Process completed!"


def test_clean_output_text_carriage_return():
    cr_text = "Step 1\rStep 2\rStep 3: Done"
    cleaned = clean_output_text(cr_text)
    assert cleaned == "Step 3: Done"


def test_normalize_source():
    assert normalize_source("line1\nline2") == ["line1\n", "line2"]
    assert normalize_source(["a", "b\n"]) == ["a\n", "b\n"]
    assert normalize_source(123) == []


def test_find_matching_output():
    output_map = {
        "cell_123": "path/to/123.json",
        "456": "path/to/456.json",
        "code_2": "path/to/code2.json",
    }
    # Exact match
    assert find_matching_output("cell_123", 0, output_map) == "path/to/123.json"
    # Suffix/prefix match
    assert find_matching_output("123", 0, output_map) == "path/to/123.json"
    assert find_matching_output("cell_456", 1, output_map) == "path/to/456.json"
    # Index fallback
    assert find_matching_output("unknown_id", 2, output_map) == "path/to/code2.json"
    # No match
    assert find_matching_output("nonexistent", 99, output_map) is None


def test_sanitize_outputs():
    raw_outputs = [
        {
            "output_type": "stream",
            "text": ["\x1b[31mError\x1b[0m\n", "Progress: 50%\rProgress: 100%\n"],
        },
        {"output_type": "error", "traceback": ["\x1b[31mTraceback\x1b[0m"]},
    ]
    sanitized = sanitize_outputs(raw_outputs)
    assert sanitized[0]["text"] == ["Error\n", "Progress: 100%\n"]
    assert sanitized[1]["traceback"] == ["Traceback"]


def test_build_notebook_e2e(tmp_path: Path):
    # Prepare dummy output step file
    out_file = tmp_path / "step_out.json"
    out_file.write_text(
        json.dumps(
            {
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": ["\x1b[32mHello Colab\x1b[0m\n"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cells_data = {
        "cells": [
            {
                "id": "md1",
                "cell_type": "markdown",
                "source": "# Title",
            },
            {
                "id": "c1",
                "cell_type": "code",
                "source": "print('Hello')",
                "outputs": [],
            },
        ]
    }

    output_map = {"c1": str(out_file)}
    nb = build_notebook(cells_data, output_map, "TestNotebook.ipynb")

    assert nb["nbformat"] == 4
    assert nb["metadata"]["colab"]["name"] == "TestNotebook.ipynb"
    assert len(nb["cells"]) == 2
    assert nb["cells"][0]["cell_type"] == "markdown"
    assert nb["cells"][1]["cell_type"] == "code"
    assert nb["cells"][1]["outputs"][0]["text"] == ["Hello Colab\n"]
