import json
from pathlib import Path

import pytest

from scripts.export_to_ipynb import (
    build_notebook,
    clean_output_text,
    find_matching_output,
    normalize_source,
    sanitize_outputs,
    validate_notebook,
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


def test_sanitize_outputs_metadata_and_execution_count():
    raw_outputs = [
        {
            "output_type": "display_data",
            "data": {"image/png": "base64data", "text/plain": "<Figure>"},
        },
        {
            "output_type": "execute_result",
            "data": {"text/plain": "42"},
        },
    ]
    sanitized = sanitize_outputs(raw_outputs)
    assert sanitized[0]["metadata"] == {}
    assert sanitized[1]["metadata"] == {}
    assert sanitized[1]["execution_count"] is None


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

    # Also test with display_data and execute_result outputs missing metadata
    cells_with_display = {
        "cells": [
            {
                "id": "c2",
                "cell_type": "code",
                "source": "plt.show()",
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {"image/png": "abc"},
                    },
                    {
                        "output_type": "execute_result",
                        "data": {"text/plain": "123"},
                    },
                ],
            }
        ]
    }
    nb2 = build_notebook(cells_with_display, {}, "DisplayNB.ipynb")
    assert nb2["cells"][0]["outputs"][0]["metadata"] == {}
    assert nb2["cells"][0]["outputs"][1]["metadata"] == {}

    try:
        import nbformat

        nbformat.validate(nb)
        nbformat.validate(nb2)
    except ImportError:
        pass


@pytest.mark.parametrize(
    "invalid_nb,expected_error",
    [
        (
            {"nbformat_minor": 0, "cells": []},
            ("nbformat", "no schema"),
        ),
        (
            {"nbformat": 4, "nbformat_minor": 0, "cells": []},
            ("metadata",),
        ),
        (
            {"nbformat": 4, "nbformat_minor": 0, "metadata": {}},
            ("cells",),
        ),
        (
            {
                "nbformat": 4,
                "nbformat_minor": 0,
                "metadata": {},
                "cells": [{"source": ["x=1"]}],
            },
            ("cell_type", "oneof", "valid"),
        ),
        (
            {
                "nbformat": 4,
                "nbformat_minor": 0,
                "metadata": {},
                "cells": [{"cell_type": "markdown", "source": ["# Title"]}],
            },
            ("metadata",),
        ),
        (
            {
                "nbformat": 4,
                "nbformat_minor": 0,
                "metadata": {},
                "cells": [
                    {
                        "cell_type": "code",
                        "metadata": {},
                        "source": ["plt.show()"],
                        "execution_count": 1,
                        "outputs": [
                            {
                                "output_type": "display_data",
                                "data": {"image/png": "abc"},
                                # missing 'metadata'!
                            }
                        ],
                    }
                ],
            },
            ("metadata",),
        ),
        (
            {
                "nbformat": 4,
                "nbformat_minor": 0,
                "metadata": {},
                "cells": [
                    {
                        "cell_type": "code",
                        "metadata": {},
                        "source": ["1 + 1"],
                        "execution_count": 1,
                        "outputs": [
                            {
                                "output_type": "execute_result",
                                "data": {"text/plain": "2"},
                                # missing 'metadata'!
                            }
                        ],
                    }
                ],
            },
            ("metadata",),
        ),
        (
            {
                "nbformat": 4,
                "nbformat_minor": 0,
                "metadata": {},
                "cells": [
                    {
                        "cell_type": "code",
                        "metadata": {},
                        "source": ["print(1)"],
                        "execution_count": 1,
                        "outputs": [
                            {
                                "output_type": "stream",
                                "text": "1\n",
                                # missing 'name'!
                            }
                        ],
                    }
                ],
            },
            ("name",),
        ),
    ],
)
def test_validate_notebook_fails_on_invalid_format(invalid_nb, expected_error):
    with pytest.raises((ValueError, Exception)) as exc_info:
        validate_notebook(invalid_nb)
    err_str = str(exc_info.value).lower()
    assert any(token in err_str for token in expected_error)


def test_nbformat_validate_fails_when_output_metadata_missing():
    """Verify that nbformat.validate explicitly rejects notebooks with missing output metadata."""
    try:
        import nbformat
    except ImportError:
        pytest.skip("nbformat not installed")

    invalid_nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["plt.plot([1, 2])"],
                "execution_count": 1,
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {"image/png": "base64data"},
                        # missing 'metadata' -> must fail nbformat validation
                    }
                ],
            }
        ],
    }

    with pytest.raises(Exception) as exc_info:
        nbformat.validate(invalid_nb)

    assert "metadata" in str(exc_info.value).lower()
