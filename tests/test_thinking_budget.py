"""I-18 ThinkingBudget: ULTRA tier + mechanical yield-floor governor.

Contract under test:
  * Ultra sits ABOVE max/xhigh/high, but the wire can only carry
    "max"/"high"/"medium" -> xhigh and ultra collapse to "max" on the wire.
  * review() is the cut rule: a thought survives only if it is actionable
    AND not a near-duplicate (Jaccard >= 0.8) of an already kept thought.
  * fewer than `floor` surviving steps => CONVERGE (stop thinking).
  * record() moves the tier only after `hysteresis` consecutive rounds.
  * the module is pure stdlib: no httpx/fastapi, no sleeps, no I/O.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODULE = REPO / "src" / "deepseek_harness" / "thinking_budget.py"

from deepseek_harness.thinking_budget import (  # noqa: E402
    TIERS,
    ThinkingGovernor,
    plan,
    wire_effort,
)


class TestTierOrdering:
    def test_ultra_above_max_above_high(self):
        assert TIERS.index("ultra") > TIERS.index("max") > TIERS.index("high")
        assert TIERS.index("xhigh") > TIERS.index("high")
        assert TIERS.index("xhigh") < TIERS.index("max")

    def test_architecture_plans_ultra(self):
        assert plan("architecture").tier == "ultra"

    def test_unknown_intent_falls_back_to_medium(self):
        assert plan("").tier == "medium"

    def test_wire_effort_is_honest_about_the_api(self):
        assert wire_effort("ultra") == "max"
        assert wire_effort("xhigh") == "max"
        assert wire_effort("high") == "high"
        assert wire_effort("medium") == "medium"
        assert wire_effort("off") is None

    def test_plan_carries_budget_and_floor(self):
        p = plan("architecture")
        assert p.budget_tokens == 24000
        assert p.yield_floor == 2
        assert "ultra" in p.protocol.lower() or len(p.protocol) > 0


class TestReviewCutRule:
    def test_duplicate_actionable_thought_is_cut(self):
        g = ThinkingGovernor()
        t = "Add a retry wrapper around the http client for transient failures"
        r = g.review([t, t])
        assert r["kept"] == [t]
        assert r["cut"] == [t]
        assert r["valuable"] == 1
        assert r["verdict"] == "CONVERGE"

    def test_non_actionable_thought_is_cut(self):
        g = ThinkingGovernor()
        r = g.review(
            [
                "It might be interesting to consider whether the design is elegant",
                "Add a retry wrapper around the http client",
            ]
        )
        assert r["valuable"] == 1
        assert len(r["cut"]) == 1

    def test_two_distinct_actionable_thoughts_continue(self):
        g = ThinkingGovernor()
        r = g.review(
            [
                "Add a retry wrapper around the http client",
                "Extract the backoff policy into its own module",
            ]
        )
        assert r["verdict"] == "CONTINUE"
        assert len(r["steps"]) == 2
        assert r["valuable"] == 2
        assert r["floor"] == 2
        assert r["cut"] == []

    def test_single_actionable_thought_converges(self):
        g = ThinkingGovernor()
        r = g.review(["Wire the governor into on_pre_llm_call"])
        assert r["verdict"] == "CONVERGE"
        assert r["valuable"] == 1

    def test_step_marker_counts_as_actionable(self):
        g = ThinkingGovernor()
        r = g.review(["Step: measure the token delta on the wire"])
        assert r["valuable"] == 1


class TestRecordHysteresis:
    def test_two_converges_demote_one_tier(self):
        g = ThinkingGovernor()
        g.plan("new")  # high
        assert g.record({"verdict": "CONVERGE", "valuable": 1})["action"] == "hold"
        out = g.record({"verdict": "CONVERGE", "valuable": 0})
        assert out["action"] == "demote"
        assert out["tier"] == "medium"

    def test_two_continues_at_ultra_hold_at_cap(self):
        g = ThinkingGovernor()
        g.plan("architecture")  # ultra
        g.record({"verdict": "CONTINUE", "valuable": 3})
        out = g.record({"verdict": "CONTINUE", "valuable": 3})
        assert out["action"] == "hold"
        assert out["tier"] == "ultra"

    def test_two_continues_at_high_promote_to_xhigh(self):
        g = ThinkingGovernor()
        g.plan("refactor")  # high
        g.record({"verdict": "CONTINUE", "valuable": 3})
        out = g.record({"verdict": "CONTINUE", "valuable": 3})
        assert out["action"] == "promote"
        assert out["tier"] == "xhigh"

    def test_continue_at_floor_does_not_promote(self):
        g = ThinkingGovernor()
        g.plan("refactor")  # high
        g.record({"verdict": "CONTINUE", "valuable": 2})
        out = g.record({"verdict": "CONTINUE", "valuable": 2})
        assert out["action"] == "hold"
        assert out["tier"] == "high"


class TestModulePurity:
    def test_no_http_or_web_frameworks_imported(self):
        # Isolate-load the module file: importing the package would run
        # __init__.py, whose deps are not this module's deps.
        code = (
            "import importlib.util, sys;"
            f"spec = importlib.util.spec_from_file_location('tb', {str(MODULE)!r});"
            "m = importlib.util.module_from_spec(spec);"
            "sys.modules['tb'] = m;"  # dataclasses needs the module registered
            "spec.loader.exec_module(m);"
            "assert 'httpx' not in sys.modules, 'httpx imported';"
            "assert 'fastapi' not in sys.modules, 'fastapi imported'"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr

    def test_no_sleeps_in_source(self):
        assert "time.sleep" not in MODULE.read_text(encoding="utf-8")
