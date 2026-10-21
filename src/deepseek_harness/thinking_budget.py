"""I-18 ThinkingBudget: an ULTRA tier plus a mechanical yield floor.

Why this is not just a bigger tier badge
---------------------------------------
Raising the effort number buys *more* thinking, not *better-bounded* thinking.
This module adds the missing half: a cut rule. Thinking continues only while it
keeps producing actionable steps above a floor; the moment the yield drops, the
governor says CONVERGE and the session's tier is demoted after a short
hysteresis window. Ultra is therefore a protocol (branches, pre-mortem, cut
after every branch), not a parameter.

Wire honesty
------------
The DeepSeek API accepts only ``reasoning_effort`` values "max", "high" and
"medium". There is no "xhigh" and no "ultra" on the wire. ``wire_effort``
therefore maps both to "max"; the extra depth above max is carried by the
in-context protocol string injected by I-17, never by a pretend API field.
``_TIER_TOKEN_BUDGET`` is an advisory *context* budget for sizing the reasoning
window, not an API parameter either.

Purity: stdlib only (dataclasses, re, typing). No I/O, no network, no clocks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Set

TIERS = ("off", "low", "medium", "high", "xhigh", "max", "ultra")

# Advisory context budget per tier — NOT an API field (see module docstring).
_TIER_TOKEN_BUDGET = {
    "off": 0,
    "low": 800,
    "medium": 2000,
    "high": 5000,
    "xhigh": 9000,
    "max": 14000,
    "ultra": 24000,
}

# Intent (I-10 router vocabulary) -> tier. architecture/research sit ABOVE max.
INTENT_TIER = {
    "simple": "medium",
    "spec_driven": "medium",
    "refactor": "high",
    "new": "high",
    "medium": "high",
    "architecture": "ultra",
    "research": "ultra",
    "collaboration": "xhigh",
}

DEFAULT_TIER = "medium"

# Adjectives / conversational fillers that look like content but name no step.
IMPERATIVE_VERBS = frozenset(
    {
        "add", "remove", "rename", "move", "split", "extract", "wire", "fix",
        "write", "delete", "replace", "run", "verify", "measure", "refactor",
        "implement", "test", "document", "pin", "raise", "lower", "drop",
    }
)
STEP_MARKERS = ("->", "=>", "step:", "next:", "todo:")

ULTRA_PROTOCOL = (
    "ULTRA protocol. Branch first: name three divergent candidate approaches, "
    "score each one line (cost, risk, reversibility), and commit only after all "
    "three are on the table. Every branch must end in a step -- verb plus object "
    "-- or be cut. Run the cut rule after each branch: a thought that repeats an "
    "already-kept thought (Jaccard 0.8) dies without discussion. Yield floor is "
    "two surviving steps. Fewer than two means STOP THINKING: take the best step "
    "and act, or ask exactly one question. Before committing, run one "
    "adversarial pre-mortem on the winning branch: name the failure mode and the "
    "check that would catch it. No fourth branch, no restating."
)

HIGH_PROTOCOL = (
    "HIGH protocol. Name two candidate approaches and pick one with a reason. "
    "Every branch must end in a step (verb + object) or be cut. Yield floor is "
    "two surviving steps; below that, stop thinking and act. Run one pre-mortem "
    "line on the chosen path."
)

ULTRA_WORDS = len(ULTRA_PROTOCOL.split())


def wire_effort(tier: str) -> str | None:
    """Map a local tier onto what the API can actually accept.

    "ultra" and "xhigh" have no API representation and collapse to "max";
    "low"/"off" have no API representation either and return None.
    The surplus above max lives in the protocol text, not on the wire.
    """
    if tier in ("ultra", "xhigh", "max"):
        return "max"
    if tier == "high":
        return "high"
    if tier == "medium":
        return "medium"
    return None


@dataclass(frozen=True)
class EffortPlan:
    """Immutable plan for one intent: tier, advisory budget, floor, protocol."""

    tier: str
    budget_tokens: int
    yield_floor: int
    protocol: str


def _tier_for(intent: str) -> str:
    return INTENT_TIER.get((intent or "").strip().lower(), DEFAULT_TIER)


def _protocol_for(tier: str) -> str:
    if tier == "ultra":
        return ULTRA_PROTOCOL
    if tier in ("xhigh", "max"):
        return HIGH_PROTOCOL
    return ""


def plan(intent: str) -> EffortPlan:
    """Build the effort plan for an I-10 intent; unknown/empty -> medium."""
    tier = _tier_for(intent)
    return EffortPlan(
        tier=tier,
        budget_tokens=_TIER_TOKEN_BUDGET[tier],
        yield_floor=2,
        protocol=_protocol_for(tier),
    )


_WORD_RE = re.compile(r"[a-z0-9]+")
_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)


def _tokens(text: str) -> Set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _head_words(text: str, n: int = 12) -> List[str]:
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    return cleaned.split()[:n]


def _is_actionable(text: str) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in STEP_MARKERS):
        return True
    return any(word in IMPERATIVE_VERBS for word in _head_words(text))


class ThinkingGovernor:
    """Stateful cut rule: per-round yield review plus hysteretic tier moves."""

    def __init__(self, floor: int = 2, hysteresis: int = 2) -> None:
        self._floor = max(0, int(floor))
        self._hysteresis = max(1, int(hysteresis))
        self._tier = DEFAULT_TIER
        self._converge_streak = 0
        self._continue_streak = 0

    @property
    def tier(self) -> str:
        return self._tier

    def plan(self, intent: str) -> EffortPlan:
        """Plan for `intent` and adopt its tier as the session's current tier."""
        p = plan(intent)
        self._tier = p.tier
        return p

    def review(self, thoughts: Sequence[str]) -> Dict[str, Any]:
        """Cut non-actionable and near-duplicate thoughts; score the yield."""
        kept: List[str] = []
        cut: List[str] = []
        kept_tokens: List[Set[str]] = []

        for thought in thoughts or []:
            text = (thought or "").strip()
            if not text:
                cut.append(text)
                continue

            tokens = _tokens(text)
            if any(_jaccard(tokens, prev) >= 0.8 for prev in kept_tokens):
                cut.append(text)          # "duplicate" — dies without discussion
                continue
            if not _is_actionable(text):
                cut.append(text)          # no verb, no step marker
                continue

            kept.append(text)
            kept_tokens.append(tokens)

        valuable = len(kept)
        return {
            "kept": kept,
            "cut": cut,
            "steps": list(kept),
            "valuable": valuable,
            "floor": self._floor,
            "verdict": "CONTINUE" if valuable >= self._floor else "CONVERGE",
        }

    def record(self, verdict: Dict[str, Any]) -> Dict[str, Any]:
        """Advance the hysteresis counters and move the tier if warranted."""
        kind = str((verdict or {}).get("verdict", "")).upper()
        valuable = int((verdict or {}).get("valuable", 0) or 0)

        if kind == "CONVERGE":
            self._converge_streak += 1
            self._continue_streak = 0
            if self._converge_streak >= self._hysteresis:
                streak = self._converge_streak
                self._converge_streak = 0
                return self._shift(-1, "demote", streak)
            return {"tier": self._tier, "action": "hold", "streak": self._converge_streak}

        if kind == "CONTINUE":
            self._continue_streak += 1
            self._converge_streak = 0
            # Promotion needs more than break-even yield: floor + 1.
            if valuable < self._floor + 1:
                return {"tier": self._tier, "action": "hold", "streak": self._continue_streak}
            if self._continue_streak >= self._hysteresis:
                streak = self._continue_streak
                self._continue_streak = 0
                return self._shift(1, "promote", streak)
            return {"tier": self._tier, "action": "hold", "streak": self._continue_streak}

        return {"tier": self._tier, "action": "hold", "streak": 0}

    def _shift(self, delta: int, action: str, streak: int) -> Dict[str, Any]:
        idx = TIERS.index(self._tier)
        new_idx = max(0, min(len(TIERS) - 1, idx + delta))
        moved = new_idx != idx
        self._tier = TIERS[new_idx]
        return {
            "tier": self._tier,
            "action": action if moved else "hold",
            "streak": streak,
        }
