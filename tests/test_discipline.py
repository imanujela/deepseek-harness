"""Discipline tests — RevisionGate (anti-churn) and StopRule (stopping criterion).

These encode two failures documented in the research:

* arXiv 2310.01798 — intrinsic self-correction without external feedback
  degrades reasoning (GPT-3.5 GSM8K 75.9% -> 74.7% after two rounds;
  CommonSenseQA 75.8% -> 41.8%), and models flip correct answers to wrong
  more often than the reverse.  An unevidenced "this new answer is better"
  switch is therefore net-negative.
* arXiv 2608.22856 — answer churn is cross-snapshot disagreement minus
  same-snapshot repeat disagreement; a model can silently change its answer
  with every input fixed.
* HF deepseek-ai/DeepSeek-V4.1-Flash#39 and arXiv 2607.01641 — an agent
  reaches the answer by turn 6-12 and then spends 20-40 turns verifying with
  no stopping criterion (2,946,619 input tokens where the edit was 3% of
  spend; "the feedback path is not effectively bounded").

So: switching answers requires external evidence, and deliberation ends once
a deliverable exists unless a falsifiable open question is named.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from deepseek_harness.discipline import RevisionGate, StopRule

EVIDENCE = {"type": "test_run", "payload": "pytest tests/test_discipline.py -> 12 passed"}


# ── RevisionGate: refusals ──────────────────────────────


class TestRevisionGateRefuses:
    def test_switch_without_evidence_is_refused(self):
        gate = RevisionGate()
        gate.commit("a1", "The answer is 42.")
        out = gate.propose("a1", "The answer is 43.", None)

        assert out["accepted"] is False
        assert out["reason"] == "no_external_evidence"
        assert out["churn"] is True
        assert out["standing"] == gate.standing_hash("a1")
        assert gate.standing_text("a1") == "The answer is 42."

    def test_evidence_of_unlisted_type_is_refused(self):
        gate = RevisionGate()
        gate.commit("a1", "Retry with a lock.")
        out = gate.propose(
            "a1", "Retry without a lock.", {"type": "vibes", "payload": "feels better"}
        )

        assert out["accepted"] is False
        assert out["reason"] == "no_external_evidence"

    def test_evidence_with_empty_payload_is_refused(self):
        gate = RevisionGate()
        gate.commit("a1", "Use a queue.")
        for empty in (None, "", [], {}):
            out = gate.propose(
                "a1", "Use a stack.", {"type": "tool_result", "payload": empty}
            )
            assert out["accepted"] is False, empty
            assert out["reason"] == "no_external_evidence"

    def test_restatement_is_not_a_new_answer(self):
        gate = RevisionGate()
        gate.commit("a1", "The parser should reject empty input early.")
        out = gate.propose(
            "a1", "The parser should reject empty input, early.", EVIDENCE
        )

        assert out["accepted"] is False
        assert out["reason"] == "restatement"
        assert gate.version("a1") == 1


# ── RevisionGate: acceptance ────────────────────────────


class TestRevisionGateAccepts:
    def test_evidenced_switch_is_accepted_and_versions(self):
        gate = RevisionGate()
        gate.commit("a1", "Cache in process memory.")
        out = gate.propose("a1", "Cache in Redis because the benchmark p99 regressed.", EVIDENCE)

        assert out == {"accepted": True, "version": 2}
        assert gate.standing_text("a1").startswith("Cache in Redis")
        assert gate.version("a1") == 2

    def test_custom_evidence_types_are_honoured(self):
        gate = RevisionGate(evidence_types=("user_instruction",))
        gate.commit("a1", "Ship on Friday.")
        refused = gate.propose("a1", "Ship on Monday.", EVIDENCE)
        accepted = gate.propose(
            "a1", "Ship on Monday.", {"type": "user_instruction", "payload": "move it"}
        )

        assert refused["accepted"] is False
        assert accepted == {"accepted": True, "version": 2}


# ── RevisionGate: churn metric ──────────────────────────


class TestChurnRate:
    def test_rate_is_zero_before_any_proposal(self):
        assert RevisionGate().churn_rate() == {
            "switches": 0,
            "evidenced": 0,
            "unevidenced": 0,
            "rate": 0.0,
        }

    def test_counts_evidenced_and_unevidenced_switches(self):
        gate = RevisionGate()
        gate.commit("a1", "Start with the narrow fix.")
        gate.propose("a1", "Rewrite the module.", None)  # unevidenced
        gate.propose("a1", "Rewrite the subsystem.", {"type": "guess", "payload": "x"})  # unevidenced
        gate.propose("a1", "Rewrite the parser behind a flag.", EVIDENCE)  # evidenced
        gate.propose("a1", "Rewrite the parser, behind a flag.", EVIDENCE)  # restatement, not a switch

        churn = gate.churn_rate()

        assert churn["switches"] == 3
        assert churn["evidenced"] == 1
        assert churn["unevidenced"] == 2
        assert churn["rate"] == pytest.approx(2 / 3)


# ── StopRule ────────────────────────────────────────────


class TestStopRule:
    def test_actions_before_deliverable_are_admitted(self):
        rule = StopRule()
        assert rule.admit("grep the failing test") == {"allow": True}
        assert rule.mark_deliverable("patch to auth.py") is None
        assert rule.status()["deliverable"] is True

    def test_action_after_deliverable_without_question_is_refused(self):
        rule = StopRule()
        rule.mark_deliverable("patch to auth.py")
        out = rule.admit("run the full suite again")

        assert out["allow"] is False
        assert out["reason"] == "deliverable_exists_no_open_question"
        assert out["say"] == "state what is missing or stop"

    def test_action_after_deliverable_naming_a_question_is_admitted(self):
        rule = StopRule()
        rule.mark_deliverable("patch to auth.py")
        assert rule.admit("does the 401 path still return a WWW-Authenticate header?") == {
            "allow": True
        }

    def test_action_with_falsifiable_because_clause_is_admitted(self):
        rule = StopRule()
        rule.mark_deliverable("patch to auth.py")
        out = rule.admit("re-run CI because we must verify the flaky test is gone")
        assert out["allow"] is True

    def test_unfalsifiable_because_clause_is_refused(self):
        rule = StopRule()
        rule.mark_deliverable("patch to auth.py")
        out = rule.admit("look deeper because it might be nicer")
        assert out["allow"] is False

    def test_soft_cap_turns_refusal_into_state_what_is_missing(self):
        """Past the cap the refusal must demand the missing sentence, not just 'stop'."""
        from deepseek_harness.discipline import StopRule

        rule = StopRule(soft_cap=2)
        rule.mark_deliverable("patch")
        # counter increments before the check: action 1 is under cap, action 2 hits it
        assert rule.admit("run the suite again")["reason"] == "deliverable_exists_no_open_question"
        over = rule.admit("verify the config hot-reloads")
        assert over["allow"] is False
        assert over["reason"] == "deliverable_exists_over_cap"
        assert "state what is missing" in over["say"]
        # a capped turn that DOES name the open question is still admitted
        assert rule.admit("does the gateway accept the new value?")["allow"] is True

    def test_status_reports_action_count_and_cap(self):
        rule = StopRule(soft_cap=3)
        for _ in range(3):
            rule.admit("read a file")
        status = rule.status()

        assert status == {
            "deliverable": False,
            "actions_since": 3,
            "cap": 3,
            "over_cap": True,
        }


# ── Purity + wiring ─────────────────────────────────────


class TestPurityAndWiring:
    def test_module_source_has_no_network_or_sleep(self):
        src = Path(__file__).resolve().parent.parent / "src" / "deepseek_harness" / "discipline.py"
        text = src.read_text(encoding="utf-8")
        for forbidden in ("httpx", "fastapi", "time.sleep", "requests", "asyncio.sleep"):
            assert forbidden not in text, f"discipline.py must not use {forbidden}"

    def test_imports_in_a_fresh_interpreter(self):
        src = Path(__file__).resolve().parent.parent / "src"
        env = {**os.environ, "PYTHONPATH": str(src)}
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import deepseek_harness.discipline as d;"
                "print(d.RevisionGate().churn_rate()['rate'], d.StopRule().status()['cap'])",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "0.0 12"

    def test_register_adds_hooks_without_new_tools_and_hook_never_raises(self):
        import deepseek_harness

        class FakeCtx:
            def __init__(self):
                self.hooks = []
                self.tools = []

            def register_hook(self, name, handler):
                self.hooks.append((name, handler))

            def register_tool(self, name, toolset, schema, handler, **kwargs):
                self.tools.append(name)

        deepseek_harness.stop_rule._deliverable = None  # isolate module-level state
        ctx = FakeCtx()
        deepseek_harness.register(ctx)

        pre_llm = [h for n, h in ctx.hooks if n == "pre_llm_call"]
        assert len(pre_llm) == 6  # 5 baseline + skill router (registered before stop rule)
        assert len(ctx.tools) == 9
        assert deepseek_harness.revision_gate is not None
        assert deepseek_harness.stop_rule is not None

        # before a deliverable: silent; after: injects the stop directive
        assert all(h(is_first_turn=False) is None for h in pre_llm[-1:])
        deepseek_harness.stop_rule.mark_deliverable("patch")
        out = pre_llm[-1](is_first_turn=False)
        assert isinstance(out, dict) and "stop" in out["context"].lower()
        assert pre_llm[-1](is_first_turn=True) is None

        # garbage kwargs must not raise
        assert pre_llm[-1](is_first_turn=False, junk=object()) is not None
        deepseek_harness.stop_rule._deliverable = None  # leave no global behind
