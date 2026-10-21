"""WIN-001 Windows runtime parity tests.

The product baseline is Linux/macOS. These tests pin the Windows behaviour the
rewrite adds: a portable Hermes-home resolution, POSIX-free permission handling,
portable process termination, and a process-identity check that works without
/proc. They are host-agnostic in intent but exercise the Windows code paths
explicitly so a regression on either platform is visible.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

import harness_server.runtime as runtime_module
import harness_server.supervisor as supervisor_module
from harness_server.runtime import (
    RuntimePaths,
    RuntimeState,
    pid_is_alive,
    process_arguments,
    process_is_owned,
    process_start_token,
    read_state,
    write_state,
)


# --------------------------------------------------------------------------
# WIN-001: Hermes home resolution is platform-aware.
# --------------------------------------------------------------------------


def test_hermes_home_env_var_wins_over_platform_default() -> None:
    from harness_server.paths import resolve_hermes_home

    resolved = resolve_hermes_home({"HERMES_HOME": "/custom/hermes"})
    assert resolved == Path("/custom/hermes")


def test_hermes_home_uses_localappdata_on_windows() -> None:
    from harness_server.paths import resolve_hermes_home

    resolved = resolve_hermes_home({"LOCALAPPDATA": r"C:\Users\test\AppData\Local"})
    assert resolved.name == "hermes"
    assert resolved.parent.name == "Local"
    assert "AppData" in str(resolved)


def test_hermes_home_uses_dot_hermes_on_posix() -> None:
    from harness_server.paths import resolve_hermes_home

    if os.name == "nt":
        pytest.skip("POSIX-only assertion: the POSIX branch never runs on Windows")
    resolved = resolve_hermes_home({"HOME": "/home/tester"})
    assert resolved.name == ".hermes"


def test_hermes_home_expands_user_in_explicit_value() -> None:
    from harness_server.paths import resolve_hermes_home

    resolved = resolve_hermes_home({"HERMES_HOME": "~/custom-hermes"})
    assert resolved.is_absolute()
    assert resolved.name == "custom-hermes"


# --------------------------------------------------------------------------
# WIN-002: file permissions are applied without POSIX-only os.fchmod.
# --------------------------------------------------------------------------


def test_write_state_works_without_posix_fchmod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """os.fchmod does not exist on Windows; write_state must not call it."""
    monkeypatch.delattr(os, "fchmod", raising=False)

    paths = RuntimePaths.from_root(tmp_path / "product")
    state = RuntimeState.create(
        instance_id="win-001",
        pid=os.getpid(),
        host="127.0.0.1",
        port=18200,
        version="3.0.0b1",
        log_path=str(paths.log_file),
        db_path=str(tmp_path / "runtime.db"),
    )

    write_state(paths, state)

    loaded = read_state(paths)
    assert loaded is not None
    assert loaded.instance_id == "win-001"


def test_runtime_paths_ensure_works_without_posix_fchmod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(os, "fchmod", raising=False)

    paths = RuntimePaths.from_root(tmp_path / "product")
    paths.ensure()

    assert paths.runtime_dir.is_dir()
    assert paths.logs_dir.is_dir()


# --------------------------------------------------------------------------
# WIN-003: termination is portable — no signal.SIGKILL on Windows.
# --------------------------------------------------------------------------


def test_terminate_signal_is_available_on_this_platform() -> None:
    """The supervisor must resolve a hard-kill signal portably."""
    kill_signal = supervisor_module.hard_kill_signal()
    assert kill_signal is not None
    if os.name == "nt":
        assert kill_signal != getattr(signal, "SIGKILL", object())


def test_hard_kill_signal_is_sigkill_on_posix() -> None:
    if os.name == "nt":
        pytest.skip("POSIX-only assertion")
    assert supervisor_module.hard_kill_signal() == signal.SIGKILL


def test_terminate_owned_uses_portable_signals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = RuntimePaths.from_root(tmp_path / "product")
    state = RuntimeState(
        schema_version=runtime_module.STATE_SCHEMA_VERSION,
        instance_id="instance-win",
        pid=424242,
        process_start_token="test:start",
        host="127.0.0.1",
        port=18200,
        version="3.0.0b1",
        started_at="2026-07-28T00:00:00+00:00",
        log_path=str(paths.log_file),
        db_path=str(tmp_path / "runtime.db"),
    )

    monkeypatch.setattr(supervisor_module, "process_is_owned", lambda candidate: True)
    sent: list[int] = []
    monkeypatch.setattr(
        supervisor_module.os, "kill", lambda pid, sent_signal: sent.append(sent_signal)
    )

    supervisor = supervisor_module.Supervisor(paths)
    monkeypatch.setattr(supervisor, "_wait_for_exit", lambda pid, timeout: True)

    supervisor._terminate_owned(state, timeout=1)

    assert sent == [signal.SIGTERM]


# --------------------------------------------------------------------------
# WIN-004: process identity works without /proc.
# --------------------------------------------------------------------------


def test_process_start_token_is_not_none_for_this_process() -> None:
    token = process_start_token(os.getpid())
    assert token is not None
    assert token.startswith(("linux:", "win:", "ps:"))


def test_process_start_token_is_stable_for_this_process() -> None:
    first = process_start_token(os.getpid())
    second = process_start_token(os.getpid())
    assert first == second


def test_process_arguments_include_the_module_invocation() -> None:
    arguments = process_arguments(os.getpid())
    assert arguments is not None
    assert any("pytest" in part or "python" in part for part in arguments)


def test_pid_is_alive_for_this_process() -> None:
    assert pid_is_alive(os.getpid()) is True


def test_pid_is_alive_for_a_dead_pid() -> None:
    assert pid_is_alive(424242) is False


def test_process_is_owned_rejects_a_foreign_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = RuntimeState(
        schema_version=runtime_module.STATE_SCHEMA_VERSION,
        instance_id="not-this-process",
        pid=os.getpid(),
        process_start_token=process_start_token(os.getpid()) or "token",
        host="127.0.0.1",
        port=18200,
        version="3.0.0b1",
        started_at="2026-07-28T00:00:00+00:00",
        log_path=str(tmp_path / "log"),
        db_path=str(tmp_path / "runtime.db"),
    )
    assert process_is_owned(state) is False


# --------------------------------------------------------------------------
# WIN-005: the installer targets the platform Hermes home, not ~/.hermes.
# --------------------------------------------------------------------------


def test_install_plan_uses_platform_hermes_home(tmp_path: Path) -> None:
    from deepseek_harness.installer import build_install_plan

    plan = build_install_plan(
        environ={"LOCALAPPDATA": str(tmp_path / "AppData" / "Local")},
        data_root=str(tmp_path / "data"),
    )

    assert plan.hermes_home.name == "hermes"
    assert "AppData" in str(plan.hermes_home)


def test_install_plan_writes_plugin_paths_under_platform_home(
    tmp_path: Path,
) -> None:
    from deepseek_harness.installer import build_install_plan

    plan = build_install_plan(
        environ={"LOCALAPPDATA": str(tmp_path / "AppData" / "Local")},
        data_root=str(tmp_path / "data"),
    )

    plugin_files = [
        path
        for path in plan.desired_files
        if path.parent.name in {"deepseek-harness", "deepseek-context"}
    ]
    assert plugin_files, "the plan must deploy both plugin adapters"
    for path in plugin_files:
        assert str(plan.hermes_home) in str(path)


def test_install_plan_does_not_hardcode_dot_hermes(tmp_path: Path) -> None:
    from deepseek_harness.installer import build_install_plan

    plan = build_install_plan(
        environ={"HOME": str(tmp_path), "LOCALAPPDATA": str(tmp_path / "AppData" / "Local")},
        data_root=str(tmp_path / "data"),
    )

    assert ".hermes" not in str(plan.hermes_home)


# --------------------------------------------------------------------------
# WIN-006: the plugin hooks read the platform Hermes home.
# --------------------------------------------------------------------------


def test_plugin_memory_paths_use_platform_home(tmp_path: Path) -> None:
    from deepseek_harness.paths import hermes_home_from_environ

    home = hermes_home_from_environ({"LOCALAPPDATA": str(tmp_path / "AppData" / "Local")})
    assert home.name == "hermes"
    assert ".hermes" not in str(home)


def test_runtime_config_default_paths_use_platform_home(
    tmp_path: Path,
) -> None:
    from harness_server.config import RuntimeConfig

    config = RuntimeConfig.from_env(
        {
            "LOCALAPPDATA": str(tmp_path / "AppData" / "Local"),
        }
    )

    assert ".hermes" not in config.db_path
    assert ".hermes" not in config.memories_dir
    assert "AppData" in config.db_path
