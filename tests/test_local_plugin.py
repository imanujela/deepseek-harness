"""LOCAL-001: the plugin runtime must be pure-local.

Hard guarantees under test:
- ``deepseek_harness.tools`` imports without ever loading ``httpx`` or
  ``fastapi`` into the process.
- tools.py source contains no ``Supervisor``, no ``httpx``, no ``time.sleep``
  and no ``_ensure_server_running`` — no tool handler path may block on,
  poll, or spawn a server.
- All nine runtime handlers execute against a temp database through the
  in-process HarnessDomain singleton and return JSON strings.
- Doctor exits 0 with no API key, no hermes executable and no running
  server; it never prints a secret value when a key IS set.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
TOOLS_SOURCE = SRC_DIR / "deepseek_harness" / "tools.py"

FAKE_SECRET = "fake-local-plugin-secret-that-must-not-appear"

# conftest.py provides this stub for pytest; a bare subprocess needs its own.
_AGENT_STUB = (
    "import sys, types\n"
    "agent = types.ModuleType('agent')\n"
    "ctx = types.ModuleType('agent.context_engine')\n"
    "class _CE: pass\n"
    "ctx.ContextEngine = _CE\n"
    "agent.context_engine = ctx\n"
    "sys.modules.setdefault('agent', agent)\n"
    "sys.modules.setdefault('agent.context_engine', ctx)\n"
)


def _run_isolated_python(body: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC_DIR), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    for volatile in ("HARNESS_DB_PATH", "HARNESS_PORT", "HARNESS_HOST"):
        env.pop(volatile, None)
    return subprocess.run(
        [sys.executable, "-c", _AGENT_STUB + body],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


# ── (1) importing tools pulls in no HTTP/server stack ──────────


@pytest.mark.parametrize("forbidden_module", ["httpx", "fastapi"])
def test_importing_tools_never_loads_server_stack(forbidden_module: str) -> None:
    result = _run_isolated_python(
        "import deepseek_harness.tools, sys\n"
        f"assert '{forbidden_module}' not in sys.modules, "
        f"[m for m in sys.modules if '{forbidden_module}' in m]\n"
    )
    assert result.returncode == 0, result.stderr


def test_importing_tools_does_not_eagerly_open_a_database() -> None:
    result = _run_isolated_python(
        "import deepseek_harness.tools as t, sys\n"
        "assert t._DOMAIN is None, 'the domain singleton must be lazy'\n"
    )
    assert result.returncode == 0, result.stderr


# ── (2) tools.py source is free of server/spawn/sleep wiring ───


def test_tools_source_contains_no_server_or_sleep_wiring() -> None:
    source = TOOLS_SOURCE.read_text(encoding="utf-8")
    for forbidden in ("httpx", "fastapi", "Supervisor", "time.sleep", "_ensure_server_running"):
        assert forbidden not in source, f"tools.py must not contain {forbidden!r}"


# ── (3) handlers run in-process against HarnessDomain ──────────


@pytest.fixture()
def local_tools(tmp_path, monkeypatch):
    from deepseek_harness import tools
    from harness_server.domain import HarnessDomain

    domain = HarnessDomain(db_path=str(tmp_path / "local-tools.db"))
    monkeypatch.setattr(tools, "_domain", lambda: domain)
    return tools, domain


def test_domain_singleton_is_lazy_and_cached(monkeypatch) -> None:
    from deepseek_harness import tools

    calls: list[dict] = []

    class SpyDomain:
        db_path = "spy://memory"

        def __init__(self, *args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(tools, "HarnessDomain", SpyDomain)
    monkeypatch.setattr(tools, "_DOMAIN", None)
    first = tools._domain()
    second = tools._domain()
    assert first is second
    assert len(calls) == 1
    # GATE-001: the singleton now injects the memory admission gate (the
    # caller owns construction of the classifier dependency).
    assert calls[0]["args"] == ()
    assert calls[0]["kwargs"] == {"admission": tools._memory_admission}


def test_all_nine_handlers_return_json_strings_against_temp_db(local_tools, tmp_path) -> None:
    tools, domain = local_tools

    created = json.loads(tools._tool_plan_create(task_description="Ship the local plugin"))
    assert "error" not in created, created
    plan_id = created["plan_id"]
    step = created["steps"][0]

    updated = json.loads(
        tools._tool_plan_update_step(step_id=step["step_id"], status="completed")
    )
    assert updated["status"] == "updated"

    cascaded = json.loads(
        tools._tool_plan_cascade(plan_id=plan_id, modified_step_id=step["step_id"])
    )
    assert "affected_steps" in cascaded

    status = json.loads(tools._tool_plan_status(plan_id=plan_id))
    assert status["plan_id"] == plan_id

    tagged = json.loads(tools._tool_memory_tag(content="must never use raw SQL"))
    assert tagged["layer"]

    queried = json.loads(tools._tool_memory_query(limit=10))
    assert "entries" in queried

    filtered = json.loads(tools._tool_memory_filter(lambda_value=0.5))
    assert filtered["lambda_value"] == 0.5

    checkpoint = json.loads(
        tools._tool_checkpoint_create(
            plan_id=plan_id,
            plan_steps=created["steps"],
            completed_step_ids=[step["step_id"]],
        )
    )
    checkpoint_id = checkpoint["checkpoint"]["checkpoint_id"]

    reviewed = json.loads(tools._tool_checkpoint_review(checkpoint_id=checkpoint_id))
    assert reviewed["review_result"]["alignment"]

    # everything landed in the temp DB — no server was contacted anywhere
    health = domain.health()
    assert health["plans"] == 1
    assert health["db_path"] == str(tmp_path / "local-tools.db")


def test_handler_failures_return_json_error_strings_without_raising(local_tools) -> None:
    tools, _ = local_tools
    missing = json.loads(tools._tool_plan_status(plan_id="does-not-exist"))
    assert "error" in missing
    bad_lambda = json.loads(tools._tool_memory_filter(lambda_value=2.5))
    assert "error" in bad_lambda


# ── (4)+(5) doctor exits 0 without key/hermes/server, never leaks secrets ──


@pytest.fixture()
def installed_local_tree(tmp_path):
    """A fake installed deployment: adapters + config + db, no server, no CLI."""
    from deepseek_harness.installer import build_install_plan
    from harness_server.domain import HarnessDomain

    home = tmp_path / "home"
    hermes_home = home / ".hermes"
    data_root = hermes_home / "oh-my-deepseek-harness"
    env = {
        "HOME": str(home),
        "USERPROFILE": str(home),
        "HERMES_HOME": str(hermes_home),
    }
    plan = build_install_plan(environ=env, data_root=str(data_root))
    for path, content in plan.desired_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    HarnessDomain(db_path=plan.runtime_config.db_path).health()  # creates schema
    return plan, env


def _diagnose(env, data_root, **overrides):
    from deepseek_harness import doctor

    options: dict = {
        "environ": env,
        "data_root": str(data_root),
        "python_version": (3, 12, 10),
        "dependency_finder": lambda _name: object(),
        "hermes_locator": lambda _name: None,
        "hermes_version_reader": lambda _executable: None,
        "port_probe": lambda _host, _port: False,
        "endpoint_probe": lambda _base, _path: pytest.fail(
            "pure-local Doctor must not probe server endpoints without managed state"
        ),
    }
    options.update(overrides)
    return doctor.diagnose(**options)


def _check(report, check_id: str) -> dict:
    return next(item for item in report.to_dict()["checks"] if item["id"] == check_id)


def test_doctor_exits_zero_without_key_hermes_or_server(installed_local_tree) -> None:
    plan, env = installed_local_tree
    assert "DEEPSEEK_API_KEY" not in env and "OPENROUTER_API_KEY" not in env
    report = _diagnose(env, plan.data_root)
    assert report.exit_code == 0, json.dumps(report.to_dict(), ensure_ascii=False)


def test_doctor_missing_hermes_executable_is_a_pass_with_note(installed_local_tree) -> None:
    plan, env = installed_local_tree
    report = _diagnose(env, plan.data_root)
    hermes = _check(report, "hermes")
    assert hermes["status"] == "pass"
    assert hermes["details"]["detected"] == "not_found"


def test_doctor_hermes_newer_than_matrix_passes(installed_local_tree) -> None:
    plan, env = installed_local_tree
    report = _diagnose(
        env,
        plan.data_root,
        hermes_locator=lambda _name: "/fake/bin/hermes",
        hermes_version_reader=lambda _exe: "0.22.2",
    )
    hermes = _check(report, "hermes")
    assert hermes["status"] == "pass"
    assert hermes["details"]["detected"] == "0.22.2"
    assert report.exit_code == 0, json.dumps(report.to_dict(), ensure_ascii=False)


def test_doctor_hermes_below_minimum_still_fails(installed_local_tree) -> None:
    from deepseek_harness.doctor import DOCTOR_UNSUPPORTED

    plan, env = installed_local_tree
    report = _diagnose(
        env,
        plan.data_root,
        hermes_locator=lambda _name: "/fake/bin/hermes",
        hermes_version_reader=lambda _exe: "0.18.0",
    )
    assert report.exit_code == DOCTOR_UNSUPPORTED
    assert _check(report, "hermes")["status"] == "fail"


def test_doctor_absent_server_is_note_not_failure(installed_local_tree) -> None:
    plan, env = installed_local_tree
    report = _diagnose(env, plan.data_root)
    server = _check(report, "server")
    port = _check(report, "port")
    assert server["status"] == "pass"
    assert port["status"] == "pass"
    provider = _check(report, "provider")
    assert provider["status"] == "pass"
    assert "API key" in provider["message"]


def test_doctor_never_prints_secret_values(installed_local_tree) -> None:
    from deepseek_harness.doctor import render_human

    plan, env = installed_local_tree
    for secret in (
        {"DEEPSEEK_API_KEY": FAKE_SECRET},
        {"OPENROUTER_API_KEY": FAKE_SECRET},
    ):
        report = _diagnose({**env, **secret}, plan.data_root)
        assert report.exit_code == 0
        assert _check(report, "provider")["status"] == "pass"
        rendered_json = json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True)
        assert FAKE_SECRET not in rendered_json
        assert FAKE_SECRET not in render_human(report)
