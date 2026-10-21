"""Skill router tests — session skill choice + mechanical per-child injection.

Contract under test:

* ``discover`` reads Superpowers plugin skills and the user skills tree from the
  filesystem, tolerating broken ``SKILL.md`` files.
* ``choose`` picks 3-5 skills, always starting with brainstorming +
  writing-for-agents, then intent and keyword matches.
* ``inject_delegate_args`` rewrites every ``delegate_task`` context through a
  pre_tool_call ``modify`` result, idempotently, without touching other keys.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from deepseek_harness import skill_router as sr

MARKER = "[superpowers-skills: dsh]"


# ── helpers ─────────────────────────────────────────────


def _write_skill(root: Path, dirname: str, name: str, description: str = "desc") -> Path:
    path = root / dirname / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\nbody\n",
        encoding="utf-8",
    )
    return path


def _pool(*names: str) -> dict:
    return {
        name: sr.SkillInfo(
            name=name,
            description=f"{name} description",
            path=f"/fake/{name}/SKILL.md",
            source="superpowers",
        )
        for name in names
    }


@pytest.fixture
def skill_tree(tmp_path: Path) -> Path:
    root = tmp_path / "skills"
    _write_skill(root, "brainstorming", "brainstorming", "Refine ideas first")
    _write_skill(root, "systematic-debugging", "systematic-debugging", "Debug deeply")
    _write_skill(root, "writing-for-agents", "writing-for-agents", "Write docs agents read")
    # A file with no frontmatter must not abort discovery.
    broken = root / "orphan" / "SKILL.md"
    broken.parent.mkdir(parents=True)
    broken.write_text("no frontmatter here\n", encoding="utf-8")
    return root


# ── (a) discovery ───────────────────────────────────────


class TestDiscover:
    def test_reads_frontmatter_and_falls_back_to_directory_name(self, skill_tree):
        found = sr.discover([str(skill_tree)])

        assert set(found) == {
            "brainstorming",
            "systematic-debugging",
            "writing-for-agents",
            "orphan",
        }
        assert found["brainstorming"].description == "Refine ideas first"
        # no frontmatter -> directory name, empty description, no crash
        assert found["orphan"].name == "orphan"
        assert found["orphan"].description == ""
        assert found["orphan"].source == "user"

    def test_unreadable_file_is_skipped_not_fatal(self, skill_tree, monkeypatch):
        real_open = Path.read_text

        def flaky(self, *args, **kwargs):
            if self.name == "SKILL.md" and self.parent.name == "systematic-debugging":
                raise PermissionError("denied")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", flaky)
        found = sr.discover([str(skill_tree)])

        assert "systematic-debugging" not in found
        assert "brainstorming" in found

    def test_missing_roots_return_empty_dict(self, tmp_path):
        assert sr.discover([str(tmp_path / "nope")]) == {}


# ── (b) choose ──────────────────────────────────────────


TRIO = ["brainstorming", "writing-for-agents", "using-superpowers"]


def _full_pool():
    return _pool(
        "brainstorming",
        "writing-for-agents",
        "using-superpowers",
        "verification-before-completion",
        "test-driven-development",
        "systematic-debugging",
        "writing-plans",
        "dispatching-parallel-agents",
        "subagent-driven-development",
        "pdf-extraction",
    )


class TestChoose:
    def test_fixed_trio_leads_for_empty_and_trivial_prompts(self):
        pool = _pool(
            "brainstorming",
            "writing-for-agents",
            "using-superpowers",
            "verification-before-completion",
            "test-driven-development",
        )

        for prompt in ("", "hi"):
            names = [item["name"] for item in sr.choose(prompt, skills=pool)]
            assert names[:3] == TRIO, prompt
            assert len(names) >= 3

    def test_verification_is_the_fourth_always_on(self):
        names = [
            item["name"] for item in sr.choose("hi", skills=_full_pool())
        ]

        assert names[:3] == TRIO
        assert names[3] == "verification-before-completion"

    def test_intent_and_keywords_extend_the_trio(self):
        chosen = sr.choose("please refactor the parser", intent="refactor", skills=_full_pool())
        names = [item["name"] for item in chosen]

        assert names[:3] == TRIO
        assert "test-driven-development" in names
        assert len(chosen) >= 3
        assert len(chosen) <= 5

    def test_bug_prompt_pulls_systematic_debugging(self):
        names = [
            item["name"]
            for item in sr.choose(
                "traceback: the parser crashes on empty input", skills=_full_pool()
            )
        ]

        assert names[:3] == TRIO
        assert "systematic-debugging" in names
        assert len(names) <= 5

    def test_dedupes_when_trio_and_matches_overlap(self):
        names = [
            item["name"]
            for item in sr.choose("plan the bug fix, then verify", skills=_full_pool())
        ]

        assert len(names) == len(set(names))
        assert names[:3] == TRIO
        assert names.count("brainstorming") == 1

    def test_floor_fills_from_whatever_pool_exists(self):
        pool = _pool("alpha", "beta", "gamma")

        names = [item["name"] for item in sr.choose("hi", skills=pool)]

        assert len(names) == 3
        assert names == ["alpha", "beta", "gamma"]

    def test_never_exceeds_available_pool(self):
        pool = _pool("brainstorming", "writing-for-agents")

        assert len(sr.choose("anything", skills=pool)) == 2


# ── (b2) discovery-driven scoring ───────────────────────


class TestScoreSkills:
    def _pdf_pool(self):
        return {
            "pdf-extraction": sr.SkillInfo(
                name="pdf-extraction",
                description="Fill and merge PDF forms",
                path="/fake/pdf-extraction/SKILL.md",
                source="user",
            ),
            "unrelated": sr.SkillInfo(
                name="unrelated",
                description="Compile kernel modules with make",
                path="/fake/unrelated/SKILL.md",
                source="user",
            ),
        }

    def test_prompt_selects_skill_by_description_not_by_table(self):
        matches = sr.score_skills("merge these pdf forms", self._pdf_pool())

        assert [item["name"] for item in matches] == ["pdf-extraction"]
        assert matches[0]["score"] >= 0.25

    def test_unrelated_description_shares_no_tokens_and_is_not_a_match(self):
        matches = sr.score_skills("merge these pdf forms", self._pdf_pool())

        assert "unrelated" not in [item["name"] for item in matches]

    def test_choose_picks_a_discovered_skill_outside_every_table(self):
        pool = {**_full_pool(), **self._pdf_pool()}

        chosen = sr.choose("merge these pdf forms", skills=pool)
        names = [item["name"] for item in chosen]

        assert names[:3] == TRIO
        assert "pdf-extraction" in names
        assert "unrelated" not in names
        assert len(chosen) <= 5

    def test_results_are_stable_and_score_sorted(self):
        pool = {
            "zeta": sr.SkillInfo("zeta", "merge pdf forms", "/z", "user"),
            "alpha": sr.SkillInfo("alpha", "merge pdf forms", "/a", "user"),
        }

        matches = sr.score_skills("merge pdf forms", pool)

        assert [item["name"] for item in matches] == ["alpha", "zeta"]


# ── (c) bootstrap ───────────────────────────────────────


class TestBootstrap:
    def test_empty_pool_returns_none(self):
        assert sr.bootstrap("build an auth service", "architecture", skills={}) is None

    def test_context_is_short_and_names_selected_skills(self):
        pool = _pool(
            "brainstorming",
            "writing-for-agents",
            "using-superpowers",
            "verification-before-completion",
            "writing-plans",
            "subagent-driven-development",
        )

        out = sr.bootstrap("design a migration architecture", "architecture", skills=pool)

        assert out is not None
        text = out["context"]
        assert text.startswith("[superpowers-skills] load before acting:")
        assert "brainstorming" in text
        assert "writing-for-agents" in text
        assert "using-superpowers" in text
        assert "skill_view" in text
        assert len(text.split()) <= 80

    def test_never_none_when_pool_has_at_least_three_skills(self):
        out = sr.bootstrap("hi", skills=_pool("alpha", "beta", "gamma"))

        assert out is not None
        for name in ("alpha", "beta", "gamma"):
            assert name in out["context"]


# ── (d) delegate_task injection ─────────────────────────


class TestInjectDelegateArgs:
    def test_batch_mode_rewrites_every_context_once(self):
        pool = _pool(
            "systematic-debugging",
            "test-driven-development",
            "verification-before-completion",
            "writing-for-agents",
        )
        args = {
            "tasks": [
                {
                    "goal": "fix the failing parser bug",
                    "context": "repo at /x",
                    "output_schema": {"type": "object"},
                },
                {
                    "goal": "write docs for the CLI",
                    "context": "",
                    "output_schema": {"type": "string"},
                },
            ]
        }

        out = sr.inject_delegate_args("delegate_task", args, skills=pool)

        assert out["action"] == "modify"
        tasks = out["args"]["tasks"]
        assert tasks[0]["goal"] == args["tasks"][0]["goal"]
        assert tasks[0]["output_schema"] == {"type": "object"}
        assert tasks[0]["context"].count(sr.MARKER) == 1
        assert tasks[0]["context"].endswith("repo at /x")
        assert tasks[1]["output_schema"] == {"type": "string"}
        assert tasks[1]["context"].startswith(sr.MARKER)
        assert tasks[1]["context"].count(sr.MARKER) == 1
        assert "repo at /x" not in tasks[1]["context"]
        # input untouched
        assert args["tasks"][0]["context"] == "repo at /x"

        # idempotent: already-marked contexts are a no-op
        assert sr.inject_delegate_args("delegate_task", {"tasks": tasks}, skills=pool) is None

    def test_single_mode_preserves_other_keys(self):
        pool = _pool(
            "test-driven-development",
            "verification-before-completion",
            "writing-for-agents",
        )
        args = {
            "goal": "implement retry logic",
            "context": "repo at /x",
            "role": "leaf",
            "output_schema": {"type": "object"},
        }

        out = sr.inject_delegate_args("delegate_task", args, skills=pool)

        assert out["action"] == "modify"
        assert set(out["args"]) == {"context"}
        assert sr.MARKER == "[superpowers-skills: dsh]"
        assert out["args"]["context"].count(sr.MARKER) == 1
        assert out["args"]["context"].endswith("repo at /x")
        assert args["context"] == "repo at /x"

        again = sr.inject_delegate_args(
            "delegate_task", {"goal": args["goal"], "context": out["args"]["context"]}, skills=pool
        )
        assert again is None

    def test_non_delegate_tools_are_untouched(self):
        pool = _pool("brainstorming", "writing-for-agents", "verification-before-completion")

        assert sr.inject_delegate_args("read_file", {"path": "x"}, skills=pool) is None
        assert sr.inject_delegate_args("terminal", {"command": "ls"}, skills=pool) is None

    def test_discovery_failure_returns_none(self, monkeypatch):
        def boom(*args, **kwargs):
            raise OSError("no skills tree")

        monkeypatch.setattr(sr, "discover", boom)

        assert sr.inject_delegate_args("delegate_task", {"goal": "fix bug"}) is None


# ── (e) child contract ──────────────────────────────────


class TestChildContract:
    def test_contract_has_marker_skill_names_and_verification_obligation(self):
        pool = _pool(
            "systematic-debugging",
            "test-driven-development",
            "verification-before-completion",
            "writing-for-agents",
        )

        text = sr.child_contract("fix the crash in the parser", skills=pool)

        assert text.startswith(MARKER)
        assert "systematic-debugging" in text
        assert "verify" in text.lower()
        assert "skill_view" in text
        assert "failing test" in text


# ── (f) purity ──────────────────────────────────────────


class TestPurity:
    def test_source_has_no_network_or_sleep(self):
        src = Path(sr.__file__).read_text(encoding="utf-8")

        for forbidden in ("httpx", "fastapi", "time.sleep", "asyncio.sleep", "requests"):
            assert forbidden not in src, f"skill_router.py must not use {forbidden}"

    def test_fresh_import_does_not_load_heavy_deps(self):
        path = Path(sr.__file__).resolve()
        code = (
            "import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('skill_router_probe', r'{path}')\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "sys.modules['skill_router_probe'] = mod\n"
            "spec.loader.exec_module(mod)\n"
            "bad = [n for n in ('httpx', 'fastapi') if n in sys.modules]\n"
            "print(mod.ALWAYS, bad)\n"
        )

        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(path.parent),
        )

        assert proc.returncode == 0, proc.stderr
        assert "httpx" not in proc.stdout and "fastapi" not in proc.stdout
        assert (
            "('brainstorming', 'writing-for-agents', 'using-superpowers')" in proc.stdout
        )


# ── (g) register wiring ─────────────────────────────────


class FakeCtx:
    def __init__(self):
        self.hooks = []
        self.tools = []

    def register_hook(self, name, handler):
        self.hooks.append((name, handler))

    def register_tool(self, name, toolset, schema, handler, **kwargs):
        self.tools.append(name)


class TestWiring:
    def test_register_adds_both_hooks_without_new_tools(self):
        import deepseek_harness

        ctx = FakeCtx()
        deepseek_harness.register(ctx)

        pre_llm = [handler for name, handler in ctx.hooks if name == "pre_llm_call"]
        pre_tool = [handler for name, handler in ctx.hooks if name == "pre_tool_call"]

        assert sr.on_pre_llm_call in pre_llm
        assert sr.on_pre_tool_call in pre_tool
        assert len(ctx.tools) == 9

    def test_hooks_never_raise(self):
        assert sr.on_pre_llm_call(junk=object()) is None
        assert sr.on_pre_llm_call(is_first_turn=False, user_message="fix bug") is None
        assert sr.on_pre_tool_call(tool_name="delegate_task", args=None) is None
        assert sr.on_pre_tool_call(junk=object()) is None

    def test_pre_llm_call_acts_only_on_the_first_real_turn(self, monkeypatch):
        pool = _pool(
            "brainstorming",
            "writing-for-agents",
            "writing-plans",
            "subagent-driven-development",
        )
        monkeypatch.setattr(sr, "discover", lambda *a, **k: pool)

        assert sr.on_pre_llm_call(is_first_turn=False, user_message="design an architecture") is None
        assert sr.on_pre_llm_call(is_first_turn=True, user_message="") is None

        out = sr.on_pre_llm_call(is_first_turn=True, user_message="design an architecture")

        assert out is not None and out["context"].startswith("[superpowers-skills]")
