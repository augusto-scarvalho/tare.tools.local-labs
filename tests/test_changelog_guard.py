from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "analysis"))

from changelog_guard import (  # noqa: E402
    ChangelogGuardError,
    check_revisions,
    is_documentation_path,
    parse_changelog,
    validate_update,
)


BASE = """# Changelog

Repository history.

## Unreleased

### Added

- Added the existing stable feature with enough descriptive context.

## 2026-08-25

### Fixed

- Fixed the historical issue without rewriting its evidence.
"""


def with_unreleased_entry(entry: str) -> str:
    return BASE.replace(
        "- Added the existing stable feature with enough descriptive context.",
        f"- {entry}\n- Added the existing stable feature with enough descriptive context.",
    )


def test_path_classification_keeps_code_and_workflows_material() -> None:
    assert is_documentation_path("docs/guide.md")
    assert is_documentation_path("README.md")
    assert is_documentation_path("CHANGELOG.md")
    assert not is_documentation_path("src/runtime.py")
    assert not is_documentation_path(".github/workflows/ci.yml")
    assert not is_documentation_path("config/research_backlog.json")


def test_material_change_requires_changelog_update() -> None:
    result = validate_update(BASE, BASE, ["src/runtime.py"])
    assert not result.ok
    assert "material changes require a CHANGELOG.md update" in result.errors


def test_material_change_accepts_new_meaningful_unreleased_entry() -> None:
    head = with_unreleased_entry(
        "Added deterministic changelog validation for material repository changes."
    )
    result = validate_update(BASE, head, ["src/runtime.py", "CHANGELOG.md"])
    assert result.ok
    assert len(result.added_entries) == 1


def test_placeholder_entry_is_rejected() -> None:
    head = with_unreleased_entry("TODO misc changes")
    result = validate_update(BASE, head, ["src/runtime.py", "CHANGELOG.md"])
    assert not result.ok
    assert any("meaningful bullet" in error for error in result.errors)
    assert any("placeholders" in error for error in result.errors)


def test_documentation_only_change_is_exempt() -> None:
    result = validate_update(BASE, BASE, ["docs/guide.md"])
    assert result.ok
    assert not result.material_paths


def test_historical_section_cannot_be_deleted_or_rewritten() -> None:
    deleted = BASE.split("## 2026-08-25", maxsplit=1)[0]
    result = validate_update(BASE, deleted, ["CHANGELOG.md"])
    assert not result.ok
    assert "historical section was deleted: ## 2026-08-25" in result.errors

    rewritten = BASE.replace("Fixed the historical issue", "Changed the historical claim")
    result = validate_update(BASE, rewritten, ["CHANGELOG.md"])
    assert not result.ok
    assert "historical section was rewritten: ## 2026-08-25" in result.errors


def test_unreleased_entry_must_be_preserved_or_moved_to_new_release() -> None:
    removed = BASE.replace(
        "- Added the existing stable feature with enough descriptive context.\n",
        "",
    )
    result = validate_update(BASE, removed, ["CHANGELOG.md"])
    assert not result.ok
    assert any("unreleased entry was deleted" in error for error in result.errors)

    moved = removed.replace(
        "## 2026-08-25",
        "## 2026-08-26\n\n### Added\n\n"
        "- Added the existing stable feature with enough descriptive context.\n\n"
        "## 2026-08-25",
    )
    result = validate_update(BASE, moved, ["CHANGELOG.md"])
    assert result.ok


def test_changelog_structure_requires_unreleased_first() -> None:
    malformed = BASE.replace("## Unreleased", "## Upcoming")
    with pytest.raises(ChangelogGuardError, match="Unreleased"):
        parse_changelog(malformed)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def test_revision_integration_uses_committed_content(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "guard@example.invalid")
    _git(tmp_path, "config", "user.name", "Changelog Guard Test")
    (tmp_path / "CHANGELOG.md").write_text(BASE, encoding="utf-8")
    (tmp_path / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "CHANGELOG.md", "runtime.py")
    _git(tmp_path, "commit", "-m", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")

    (tmp_path / "runtime.py").write_text("VALUE = 2\n", encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text(
        with_unreleased_entry("Changed the runtime value with a deterministic migration note."),
        encoding="utf-8",
    )
    _git(tmp_path, "add", "CHANGELOG.md", "runtime.py")
    _git(tmp_path, "commit", "-m", "material change")
    head = _git(tmp_path, "rev-parse", "HEAD")

    result = check_revisions(str(tmp_path), base, head)
    assert result.ok
    assert result.material_paths == ("runtime.py",)


def test_revision_integration_fails_closed_for_missing_base(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "guard@example.invalid")
    _git(tmp_path, "config", "user.name", "Changelog Guard Test")
    (tmp_path / "CHANGELOG.md").write_text(BASE, encoding="utf-8")
    _git(tmp_path, "add", "CHANGELOG.md")
    _git(tmp_path, "commit", "-m", "base")
    head = _git(tmp_path, "rev-parse", "HEAD")

    with pytest.raises(ChangelogGuardError, match="base commit is unavailable"):
        check_revisions(str(tmp_path), "1" * 40, head)
