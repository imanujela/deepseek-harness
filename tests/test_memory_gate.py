"""GATE-001: the memory_hygiene gate is attached to the actual write path.

Evidence for why a prompt-level rule is not enough: HF discussion
``deepseek-ai/DeepSeek-V4.1-Flash#46`` measured the same conversation writing
13 KB / 9 memory files on V4 Flash versus 161 KB / 11 files on V4.1 Flash —
12x — and the excess was not facts: decision logs retelling every step,
"open questions", self-advice, 109 "TBC" placeholders for facts the user never
gave, and file preambles. System-prompt rules had zero effect, so the gate
belongs on the WRITE path.

Contract under test:
- ``HarnessStorage(db_path, admission=callable)`` refuses non-fact writes with
  ``MemoryRefused(reason, kind)`` and enforces a per-instance character budget;
- ``HarnessStorage`` without ``admission`` keeps the old behaviour (library
  users and existing tests are untouched);
- ``HarnessDomain(admission=...)`` forwards the callable to its storage;
- ``deepseek_harness.tools._domain()`` wires ``memory_hygiene.admit`` in.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from harness_server.domain import HarnessDomain
from harness_server.models import MemoryEntry, MemoryLayer
from harness_server.storage import HarnessStorage, MemoryRefused

from deepseek_harness import memory_hygiene, tools

#: A real chronicle from the V4.1 Flash pattern: process retold as prose.
CHRONICLE = (
    "I then created the schema, verified the migration, and re-ran the tests."
)
#: A terse fact with a concrete value (the shape the gate is meant to keep).
FACT = "Build time is 12.5s now."


def _entry(content: str, tags: list[str] | None = None) -> MemoryEntry:
    return MemoryEntry(
        content=content,
        tags=tags or ["fact"],
        layer=MemoryLayer.PATTERN,
    )


def _admission(text: str, *, used_chars: int = 0) -> dict:
    """Plain wrapper around the shipped classifier, as tools.py wires it."""
    return memory_hygiene.admit(text, used_chars=used_chars)


# ════════════════════════════════════════════════════════════════
# (a) storage with admission refuses a chronicle, stores a fact
# ════════════════════════════════════════════════════════════════


def test_storage_refuses_chronicle_and_keeps_fact(tmp_path: Path) -> None:
    store = HarnessStorage(str(tmp_path / "gate.db"), admission=_admission)

    with pytest.raises(MemoryRefused) as refused:
        store.insert_memory(_entry(CHRONICLE), source="session.md")

    assert refused.value.kind == "chronicle"
    assert "chronicle" in refused.value.reason
    assert store.count_total_memories() == 0

    store.insert_memory(_entry(FACT), source="session.md")

    rows = store.query_memories(tags=["fact"])
    assert [row.content for row in rows] == [FACT]


def test_refusal_message_names_the_kind(tmp_path: Path) -> None:
    """Requirement (4): a surfacing error must name the refused kind."""
    store = HarnessStorage(str(tmp_path / "gate.db"), admission=_admission)

    with pytest.raises(MemoryRefused) as refused:
        store.insert_memory(_entry(CHRONICLE))

    assert "chronicle" in str(refused.value)


# ════════════════════════════════════════════════════════════════
# (b) no admission callable → no gate (no regression for library users)
# ════════════════════════════════════════════════════════════════


def test_storage_without_admission_stores_chronicle(tmp_path: Path) -> None:
    store = HarnessStorage(str(tmp_path / "plain.db"))

    store.insert_memory(_entry(CHRONICLE), source="session.md")

    rows = store.query_memories(tags=["fact"])
    assert [row.content for row in rows] == [CHRONICLE]


# ════════════════════════════════════════════════════════════════
# (c) per-instance character budget
# ════════════════════════════════════════════════════════════════


def test_budget_refuses_after_admitted_chars_exceed_budget(tmp_path: Path) -> None:
    def admission(text: str, *, used_chars: int = 0) -> dict:
        return memory_hygiene.admit(text, budget_chars=50, used_chars=used_chars)

    store = HarnessStorage(str(tmp_path / "budget.db"), admission=admission)
    store.insert_memory(_entry(FACT))  # admitted; running total grows

    with pytest.raises(MemoryRefused) as refused:
        store.insert_memory(_entry("Tests: 308 passed at 2025-01-15."))

    assert refused.value.kind == "fact"
    assert "budget overflow" in refused.value.reason
    assert store.count_total_memories() == 1


# ════════════════════════════════════════════════════════════════
# (d) HarnessDomain forwards the admission callable
# ════════════════════════════════════════════════════════════════


def test_domain_forwards_admission_to_constructed_storage(tmp_path: Path) -> None:
    domain = HarnessDomain(
        db_path=str(tmp_path / "domain-gate.db"), admission=_admission
    )

    with pytest.raises(MemoryRefused) as refused:
        domain.storage.insert_memory(_entry(CHRONICLE), source="session.md")

    assert refused.value.kind == "chronicle"
    assert domain.storage.count_total_memories() == 0


def test_domain_attaches_admission_to_accepted_storage(tmp_path: Path) -> None:
    storage = HarnessStorage(str(tmp_path / "accepted.db"))

    domain = HarnessDomain(storage=storage, admission=_admission)

    with pytest.raises(MemoryRefused) as refused:
        domain.storage.insert_memory(_entry(CHRONICLE))

    assert refused.value.kind == "chronicle"


# ════════════════════════════════════════════════════════════════
# (e) the plugin factory wires memory_hygiene.admit
# ════════════════════════════════════════════════════════════════


def test_plugin_domain_factory_wires_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "tools-gate.db"
    real_domain = tools.HarnessDomain

    def factory(**kwargs):
        return real_domain(db_path=str(db), **kwargs)

    monkeypatch.setattr(tools, "HarnessDomain", factory)
    monkeypatch.setattr(tools, "_DOMAIN", None)

    domain = tools._domain()

    assert domain.storage._admission is not None
    assert domain.storage._admission is tools._memory_admission
    with pytest.raises(MemoryRefused) as refused:
        domain.storage.insert_memory(_entry(CHRONICLE), source="session.md")
    assert refused.value.kind == "chronicle"

    domain.storage.insert_memory(_entry(FACT), source="session.md")
    assert [row.content for row in domain.storage.query_memories(tags=["fact"])] == [
        FACT
    ]


def test_plugin_admission_wrapper_returns_hygiene_verdict() -> None:
    verdict = tools._memory_admission(CHRONICLE, used_chars=0)

    assert verdict["admit"] is False
    assert verdict["kind"] == "chronicle"
    assert verdict["reason"]


# ════════════════════════════════════════════════════════════════
# (4) refusals surface through the tool dispatch error shape
# ════════════════════════════════════════════════════════════════


def test_importing_tools_does_not_pay_for_the_classifier() -> None:
    """The admission import is lazy: tool import cost stays unchanged."""
    script = (
        "import sys; import deepseek_harness.tools; "
        "assert 'deepseek_harness.memory_hygiene' not in sys.modules, "
        "'memory_hygiene imported eagerly'"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )

    assert completed.returncode == 0, completed.stderr


def test_tool_dispatch_surfaces_refusal_with_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No memory-storing handler is registered yet (MEM-001 pending), but the
    shared dispatch seam already has to make a refusal visible instead of
    swallowing it: ``{"error": ...}`` with the refused kind in the message."""

    def refuse(method: str, path: str, body: dict) -> dict:
        raise MemoryRefused("refused: kind=chronicle; narrates process", "chronicle")

    monkeypatch.setattr(tools, "_route", refuse)

    result = tools._call_server("POST", "/memory/store", json={"content": CHRONICLE})

    assert "error" in result
    assert "chronicle" in result["error"]
    assert "memory refused" in result["error"]
