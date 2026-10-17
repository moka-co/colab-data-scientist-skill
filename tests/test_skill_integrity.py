"""Integrity tests for markdown links, line-index maps, and subagent contracts."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_skill_md_conciseness_and_default_mode():
    """SKILL.md should remain concise and specify Quick Exploration as default."""
    skill_path = REPO_ROOT / "SKILL.md"
    content = skill_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    assert len(lines) <= 140, f"SKILL.md is too long ({len(lines)} lines > 140)"
    assert "Quick Exploration" in content
    assert "Polished Deliverable" in content
    assert "CELLS_TO_EXECUTE" in content


def test_internal_markdown_links_resolve():
    """All relative markdown links in repo docs must resolve to existing paths."""
    md_files = [
        REPO_ROOT / "SKILL.md",
        REPO_ROOT / "README.md",
        *(REPO_ROOT / "references").glob("*.md"),
        *(REPO_ROOT / "best-practices").glob("*.md"),
        *(REPO_ROOT / "agents").rglob("*.md"),
    ]
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        for match in link_pattern.finditer(text):
            target = match.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:", "#")) or "://" in target:
                continue
            target_path_str = target.split("#", 1)[0]
            if not target_path_str:
                continue
            resolved = (md_file.parent / target_path_str).resolve()
            assert resolved.exists(), (
                f"Broken link in {md_file.relative_to(REPO_ROOT)}: '{target}' -> {resolved}"
            )


def test_subagent_contracts():
    """Verify notebook subagents define CELLS_TO_EXECUTE and critic defines fix tiers."""
    notebook_agents = [
        "env-setup-agent.md",
        "eda-agent.md",
        "data-cleaning-agent.md",
        "model-training-agent.md",
        "inference-agent.md",
    ]
    for name in notebook_agents:
        path = REPO_ROOT / "agents" / "notebook" / name
        text = path.read_text(encoding="utf-8")
        assert "CELLS_TO_EXECUTE" in text, f"{name} missing CELLS_TO_EXECUTE contract"

    critic_text = (REPO_ROOT / "agents" / "notebook" / "critic-agent.md").read_text(
        encoding="utf-8"
    )
    assert "SAFE_AUTO_FIXES" in critic_text
    assert "ESCALATED_FIXES" in critic_text

    discovery_text = (REPO_ROOT / "agents" / "support" / "dataset-discovery-agent.md").read_text(
        encoding="utf-8"
    )
    for required_field in ["License", "Size", "Last Updated", "Colab Secrets"]:
        assert required_field in discovery_text, (
            f"dataset-discovery-agent.md missing required field '{required_field}'"
        )


def test_data_science_libraries_line_map():
    """Verify the Section-to-Line Index Map in best-practices/data-science-libraries.md."""
    lib_file = REPO_ROOT / "best-practices" / "data-science-libraries.md"
    lines = lib_file.read_text(encoding="utf-8").splitlines()

    map_row_pattern = re.compile(
        r"\|\s*(?:\*\*Part [IVX]+\*\*)?\s*\|\s*§(\d+)\s*\|[^|]+\|[^|]+\|\s*L(\d+)-L(\d+)\s*\|"
    )
    found_rows = 0
    for line in lines:
        match = map_row_pattern.search(line)
        if match:
            found_rows += 1
            sec_num = match.group(1)
            start_line = int(match.group(2))
            end_line = int(match.group(3))
            assert 1 <= start_line <= end_line <= len(lines), (
                f"Invalid line range L{start_line}-L{end_line} for §{sec_num}"
            )
            header_slice = "\n".join(lines[max(0, start_line - 2) : start_line + 2])
            assert f"## {sec_num}." in header_slice, (
                f"Expected '## {sec_num}.' near L{start_line}, found:\n{header_slice}"
            )
    assert found_rows == 28, f"Expected 28 mapped sections, found {found_rows}"


def test_no_hardcoded_user_paths():
    """Ensure no hardcoded personal Windows paths exist in repository files."""
    files = [
        REPO_ROOT / "SKILL.md",
        REPO_ROOT / "README.md",
        *(REPO_ROOT / "references").glob("*.md"),
        *(REPO_ROOT / "best-practices").glob("*.md"),
        *(REPO_ROOT / "agents").rglob("*.md"),
        *(REPO_ROOT / "scripts").glob("*.py"),
    ]
    for path in files:
        content = path.read_text(encoding="utf-8")
        assert "Users\\tech" not in content and "Users/tech" not in content, (
            f"Hardcoded user path found in {path.relative_to(REPO_ROOT)}"
        )
