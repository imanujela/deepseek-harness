"""Skill deployment contract: install ships skills/deepseek-harness into the
Hermes skills directory, uninstall removes it, upgrade refreshes it, and a
second deploy is idempotent.

These tests exercise the installer transaction primitives and the lifecycle
management surface directly, so no Harness Server process is started.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from deepseek_harness import installer, lifecycle
from deepseek_harness.installer import InstallPlan
from deepseek_harness.lifecycle import LifecycleConflictError, uninstall

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_NAME = "deepseek-harness"
SKILL_SOURCE = REPO_ROOT / "skills" / SKILL_NAME / "SKILL.md"


def _plan(tmp_path: Path) -> InstallPlan:
    home = tmp_path / "home"
    home.mkdir()
    hermes_home = home / ".hermes"
    env = {
        **os.environ,
        "HOME": str(home),
        "USERPROFILE": str(home),
        "HERMES_HOME": str(hermes_home),
        "HARNESS_DATA_ROOT": str(hermes_home / "oh-my-deepseek-harness"),
        "HARNESS_IMPORT_MEMORIES": "0",
        "PYTHONNOUSERSITE": "1",
    }
    return installer.build_install_plan(environ=env)


def _skill_target(plan: InstallPlan) -> Path:
    return plan.hermes_home / "skills" / SKILL_NAME


def _deploy(plan: InstallPlan) -> list[bool]:
    """Run the same directory/file transaction as execute_install, minus the server."""
    installer._preflight(plan)
    flags: list[bool] = []
    created_files: list[Path] = []
    created_directories: list[Path] = []
    for directory in plan.desired_directories:
        installer._ensure_directory(directory, created_directories)
    for path, content in plan.desired_files.items():
        flags.append(installer._atomic_create(path, content, created_files))
    return flags


def _frontmatter(text: str) -> dict:
    assert text.startswith("---")
    closing = text.index("\n---\n", 3)
    return yaml.safe_load(text[3:closing])


def test_skill_source_frontmatter_declares_the_skill() -> None:
    assert SKILL_SOURCE.is_file(), "the packaged skill source must exist"
    frontmatter = _frontmatter(SKILL_SOURCE.read_text(encoding="utf-8"))
    assert frontmatter["name"] == SKILL_NAME


def test_install_plan_targets_the_hermes_skills_directory(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    target = _skill_target(plan)
    assert plan.skill_directory == target
    assert target in plan.desired_directories
    assert target / "SKILL.md" in plan.desired_files
    assert (
        plan.desired_files[target / "SKILL.md"]
        == SKILL_SOURCE.read_text(encoding="utf-8")
    )
    payload = plan.public_dict(dry_run=True)
    assert payload["skill_deployment_path"] == str(target)


def test_install_creates_skill_entry_with_valid_frontmatter(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    assert _deploy(plan)
    deployed = _skill_target(plan) / "SKILL.md"
    assert deployed.is_file()
    assert _frontmatter(deployed.read_text(encoding="utf-8"))["name"] == SKILL_NAME


def test_second_install_is_idempotent(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    assert all(_deploy(plan))
    assert _deploy(plan) == [False] * len(plan.desired_files)


def test_uninstall_removes_the_deployed_skill(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    _deploy(plan)
    assert _skill_target(plan).is_dir()

    result = uninstall(plan)

    assert not (_skill_target(plan) / "SKILL.md").exists()
    assert not _skill_target(plan).exists()
    assert any(name.endswith("SKILL.md") for name in result["removed_managed_files"])
    assert result["changed"] is True


def test_uninstall_preserves_a_user_edited_skill_file(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    _deploy(plan)
    deployed = _skill_target(plan) / "SKILL.md"
    deployed.write_text("# user-managed skill\n", encoding="utf-8")

    with pytest.raises(LifecycleConflictError):
        uninstall(plan)

    assert deployed.read_text(encoding="utf-8") == "# user-managed skill\n"


def test_upgrade_refreshes_the_deployed_skill(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    _deploy(plan)
    deployed = _skill_target(plan) / "SKILL.md"
    deployed.write_text("stale content\n", encoding="utf-8")

    lifecycle._replace_managed_files(plan)

    assert deployed.read_text(encoding="utf-8") == SKILL_SOURCE.read_text(
        encoding="utf-8"
    )
