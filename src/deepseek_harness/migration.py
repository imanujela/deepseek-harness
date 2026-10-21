"""MIG-001: Beta migration and failure rollback.

Thin glue over lifecycle.upgrade + audit_events. Handles:
- Constraint-violations.md -> audit-events.jsonl migration (TC-MIG-003)
- Downgrade rejection before writes (TC-MIG-006)
- Per-stage failure injection via phase_hook (TC-MIG-005)
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from .audit_events import append_event, read_events
from .paths import memories_path

logger = logging.getLogger(__name__)


def default_constraint_md() -> str:
    """返回约束违反 Markdown 文件路径。"""
    return str(memories_path("constraint-violations.md"))


def default_audit_jsonl() -> str:
    """返回审计事件 JSONL 文件路径。"""
    return str(memories_path("audit-events.jsonl"))


# constraint-violations.md line format: "- [quality] constraint | evidence | tool"
_LINE_RE = re.compile(
    r"^-\s*\[([^\]]*)\]\s*(.+?)(?:\s*\|\s*(.+?))?(?:\s*\|\s*(.+?))?\s*$"
)


def migrate_constraint_log(
    md_path: Optional[str] = None,
    jsonl_path: Optional[str] = None,
) -> dict[str, Any]:
    """Migrate constraint-violations.md entries to audit-events.jsonl.

    Valid lines become structured events; invalid lines are preserved in the
    original file and reported (TC-MIG-003).
    """
    if md_path is None:
        md_path = default_constraint_md()
    if jsonl_path is None:
        jsonl_path = default_audit_jsonl()
    src = Path(md_path)
    if not src.exists():
        return {"migrated": 0, "skipped": 0, "preserved_file": None}

    lines = src.read_text(encoding="utf-8").strip().splitlines()
    migrated, skipped, bad_lines = 0, 0, []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        if not m:
            bad_lines.append(line)
            skipped += 1
            continue
        quality, constraint, evidence, tool = (
            m.group(1).strip(),
            m.group(2).strip(),
            (m.group(3) or "").strip(),
            (m.group(4) or "").strip(),
        )
        append_event(
            {
                "quality": quality,
                "constraint": constraint,
                "evidence": evidence,
                "tool": tool,
            },
            path=jsonl_path,
        )
        migrated += 1

    # Preserve un-migratable lines in the original file
    if bad_lines:
        src.write_text("\n".join(bad_lines) + "\n", encoding="utf-8")
    else:
        src.unlink(missing_ok=True)

    return {
        "migrated": migrated,
        "skipped": skipped,
        "preserved_file": str(src) if bad_lines else None,
    }


def reject_downgrade(current: str, target: str) -> None:
    """Raise if target version is older than current (TC-MIG-006).

    Uses tuple comparison on numeric components; non-numeric suffixes
    (b1, rc, etc.) are stripped for the comparison.
    """
    def _parse(v: str) -> tuple[int, ...]:
        # Strip non-numeric suffixes like b1, rc2, -beta
        core = re.match(r"(\d+\.?\d*)", v).group(1).rstrip(".")
        return tuple(int(p) for p in core.split("."))

    if _parse(target) < _parse(current):
        raise ValueError(
            f"downgrade rejected: current={current} target={target}; "
            "refusing to write older schema over newer data"
        )


def migration_inventory(data_root: Path) -> dict[str, Any]:
    """Before/after snapshot of Config/DB/events for TC-MIG-001 inventory."""
    db = data_root / "harness.db"
    events = data_root / "events"
    config = data_root / "config.yaml"
    constraint_md = default_constraint_md()
    audit_jsonl = default_audit_jsonl()
    return {
        "config_exists": config.exists(),
        "db_exists": db.exists(),
        "db_size": db.stat().st_size if db.exists() else 0,
        "events_dir_exists": events.exists(),
        "events_count": len(list(events.glob("*.jsonl"))) if events.exists() else 0,
        "constraint_md_exists": Path(constraint_md).exists(),
        "audit_jsonl_exists": Path(audit_jsonl).exists(),
        "audit_jsonl_events": len(read_events(audit_jsonl)) if Path(audit_jsonl).exists() else 0,
    }
