"""Discipline primitives: RevisionGate (anti-churn) and StopRule (stop criterion).

Why these exist — the research, not intuition:

1. arXiv 2310.01798 ("Large Language Models Cannot Self-Correct Reasoning Yet")
   proves intrinsic self-correction without external feedback DEGRADES
   reasoning: GPT-3.5 on GSM8K 75.9% -> 74.7% after two rounds, CommonSenseQA
   75.8% -> 41.8%. Models flip correct answers to wrong more often than the
   reverse. A self-initiated "this new answer is better" switch is therefore
   net-negative unless external evidence backs it. RevisionGate refuses the
   switch and records the refusal so the unevidenced rate can be driven to 0.

2. arXiv 2608.22856 defines answer churn as cross-snapshot disagreement minus
   same-snapshot repeat disagreement: a model can change its answer with every
   input fixed, invisibly to accuracy metrics. RevisionGate.churn_rate() makes
   that visible per answer.

3. HF discussion deepseek-ai/DeepSeek-V4.1-Flash#39: an agent reaches the
   answer by turn 6-12, then spends 20-40 turns verifying with NO stopping
   criterion — 2,946,619 input tokens where the actual edit was 3% of spend
   (cf. arXiv 2607.01641, "Infinite Agentic Loops: the feedback path is not
   effectively bounded"). StopRule ends deliberation once a deliverable exists
   and only admits work that names a falsifiable open question.

Pure stdlib, no I/O, no sleeps: both classes are deterministic state machines so
the tool layer and the pre_llm_call hook can share them safely.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_EVIDENCE_TYPES = (
    "tool_result",
    "test_run",
    "citation",
    "user_instruction",
)

# A "because"/"until" clause only counts as an open question when it names a
# check that could come back false.
_CHECK_VERBS = frozenset(
    {
        "check",
        "compare",
        "confirm",
        "fails",
        "fail",
        "grep",
        "inspect",
        "measure",
        "query",
        "read",
        "reproduce",
        "run",
        "test",
        "validate",
        "verify",
    }
)

_CLAUSE = re.compile(r"\b(?:because|until)\b(.*)$", re.IGNORECASE | re.DOTALL)
_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set:
    return set(_WORD.findall((text or "").lower()))


def _jaccard(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _digest(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _payload_filled(payload: Any) -> bool:
    if payload is None:
        return False
    if isinstance(payload, str):
        return bool(payload.strip())
    try:
        return len(payload) > 0
    except TypeError:
        return bool(payload)


def _external_evidence(evidence: Any, allowed: Sequence[str]) -> bool:
    if not isinstance(evidence, dict):
        return False
    if evidence.get("type") not in allowed:
        return False
    return _payload_filled(evidence.get("payload"))


class RevisionGate:
    """Refuse answer switches that rest on no external evidence.

    ``commit`` records the standing answer. ``propose`` accepts a switch only
    when it carries evidence of an allowed type with a non-empty payload; a
    near-identical rewording is classified as a restatement, not a new answer.
    """

    def __init__(
        self, evidence_types: Sequence[str] = DEFAULT_EVIDENCE_TYPES
    ) -> None:
        self.evidence_types = tuple(evidence_types)
        self._answers: Dict[str, Dict[str, Any]] = {}
        self._switches = 0
        self._evidenced = 0
        self._unevidenced = 0

    # ── state ──────────────────────────────────────────

    def commit(
        self, answer_id: str, answer_text: str, *, evidence: Any = None
    ) -> Dict[str, Any]:
        """Record (or re-record) the standing answer; returns the new version."""
        previous = self._answers.get(answer_id)
        version = (previous["version"] + 1) if previous else 1
        if previous is not None and previous["text"] != answer_text:
            # An explicit commit is an operator decision, not a self-switch:
            # it is counted as evidenced and never as churn.
            self._switches += 1
            self._evidenced += 1
        self._answers[answer_id] = {
            "text": answer_text,
            "version": version,
            "evidence": evidence,
        }
        logger.debug(
            "[discipline] commit %s v%s (%d chars)", answer_id, version, len(answer_text or "")
        )
        return {"committed": True, "answer_id": answer_id, "version": version}

    def standing_text(self, answer_id: str) -> Optional[str]:
        record = self._answers.get(answer_id)
        return record["text"] if record else None

    def standing_hash(self, answer_id: str) -> Optional[str]:
        record = self._answers.get(answer_id)
        return _digest(record["text"]) if record else None

    def version(self, answer_id: str) -> int:
        record = self._answers.get(answer_id)
        return record["version"] if record else 0

    # ── the lock ───────────────────────────────────────

    def propose(
        self, answer_id: str, new_text: str, evidence: Any = None
    ) -> Dict[str, Any]:
        """Try to replace the standing answer. Returns the verdict dict."""
        standing = self._answers.get(answer_id)
        if standing is None:
            logger.debug("[discipline] propose for uncommitted answer %s", answer_id)
            return {
                "accepted": False,
                "reason": "no_standing_answer",
                "standing": None,
                "churn": False,
            }

        if _jaccard(standing["text"], new_text) >= 0.8:
            logger.debug("[discipline] restatement on %s, not a switch", answer_id)
            return {
                "accepted": False,
                "reason": "restatement",
                "standing": _digest(standing["text"]),
                "churn": False,
            }

        if not _external_evidence(evidence, self.evidence_types):
            self._switches += 1
            self._unevidenced += 1
            logger.debug(
                "[discipline] refused unevidenced switch on %s (evidence=%r)",
                answer_id,
                evidence,
            )
            return {
                "accepted": False,
                "reason": "no_external_evidence",
                "standing": _digest(standing["text"]),
                "churn": True,
            }

        self._switches += 1
        self._evidenced += 1
        version = standing["version"] + 1
        self._answers[answer_id] = {
            "text": new_text,
            "version": version,
            "evidence": evidence,
        }
        logger.debug("[discipline] accepted evidenced switch on %s -> v%s", answer_id, version)
        return {"accepted": True, "version": version}

    def churn_rate(self) -> Dict[str, Any]:
        """Switches, evidenced, unevidenced, and the rate to drive to zero."""
        total = self._switches
        rate = (total - self._evidenced) / total if total else 0.0
        return {
            "switches": total,
            "evidenced": self._evidenced,
            "unevidenced": total - self._evidenced,
            "rate": rate,
        }


class StopRule:
    """End deliberation once a deliverable exists.

    Before ``mark_deliverable`` any action is admitted. After it, an action is
    admitted only when it names a falsifiable open question — a trailing "?"
    or a because/until clause containing a check verb.
    """

    REFUSAL = {
        "allow": False,
        "reason": "deliverable_exists_no_open_question",
        "say": "state what is missing or stop",
    }
    # Once the soft cap is spent, "keep going" stops being an option: the only
    # admissible continuation is naming what is still missing. V4.1-Flash's
    # documented failure is 20-40 extra verification turns with no bound, so the
    # cap converts an unbounded loop into one required sentence.
    CAP_REFUSAL = {
        "allow": False,
        "reason": "deliverable_exists_over_cap",
        "say": "soft cap reached: state what is missing or unverified in one line, or stop",
    }

    def __init__(self, soft_cap: int = 12) -> None:
        self.soft_cap = soft_cap
        self._deliverable: Optional[str] = None
        self._actions_since = 0

    def mark_deliverable(self, what: str) -> None:
        self._deliverable = what
        self._actions_since = 0
        logger.debug("[discipline] deliverable marked: %s", what)

    def _names_open_question(self, action: str) -> bool:
        text = action or ""
        if "?" in text:
            return True
        clause = _CLAUSE.search(text)
        if clause is None:
            return False
        return bool(_words(clause.group(1)) & _CHECK_VERBS)

    def admit(self, action: str) -> Dict[str, Any]:
        self._actions_since += 1
        if self._deliverable is None:
            return {"allow": True}
        if self._names_open_question(action):
            return {"allow": True}
        logger.debug("[discipline] refused post-deliverable action: %s", action)
        if self._actions_since >= self.soft_cap:
            return dict(self.CAP_REFUSAL)
        return dict(self.REFUSAL)

    def status(self) -> Dict[str, Any]:
        return {
            "deliverable": self._deliverable is not None,
            "actions_since": self._actions_since,
            "cap": self.soft_cap,
            "over_cap": self._actions_since >= self.soft_cap,
        }
