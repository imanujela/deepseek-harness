"""DOMAIN-001: HarnessDomain 纯进程内服务测试。

覆盖 domain.py 的契约：
- HarnessDomain 直接操作 SQLite，不需要任何 HTTP 服务 / FastAPI
- 方法返回形状与现有 HTTP 端点响应一致（工具层可直接复用同一份 JSON）
- 缺失实体抛 ValueError（HTTP 适配层再翻译成 404/422）
- 导入 harness_server.domain 不会拉起 fastapi
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from harness_server.domain import HarnessDomain
from harness_server.models import MemoryEntry, MemoryLayer

TASK = "设计数据库 schema、实现 API、编写测试"


@pytest.fixture()
def domain(tmp_path: Path) -> HarnessDomain:
    """指向临时数据库的域服务实例。"""
    return HarnessDomain(db_path=str(tmp_path / "domain.db"))


# ════════════════════════════════════════════════════════════════
# Plan Engine
# ════════════════════════════════════════════════════════════════


def test_create_plan_returns_plan_id_and_steps(domain: HarnessDomain) -> None:
    """create_plan 返回 plan_id 和 3-8 个步骤。"""
    result = domain.create_plan(TASK)

    assert result["plan_id"]
    assert 3 <= len(result["steps"]) <= 8
    assert all(step["text"] for step in result["steps"])


def test_create_plan_rejects_empty_description(domain: HarnessDomain) -> None:
    """空任务描述被拒绝（HTTP 适配层翻译成 422）。"""
    with pytest.raises(ValueError):
        domain.create_plan("   ")


def test_plan_status_returns_steps_and_graph(domain: HarnessDomain) -> None:
    """plan_status 回读已创建的步骤与依赖图。"""
    created = domain.create_plan(TASK)

    status = domain.plan_status(created["plan_id"])

    assert status["plan_id"] == created["plan_id"]
    assert len(status["steps"]) == len(created["steps"])
    assert status["adjacency_list"]
    assert status["created_at"] and status["updated_at"]


def test_plan_status_unknown_plan_raises_value_error(domain: HarnessDomain) -> None:
    """未知 plan 抛 ValueError（HTTP 适配层翻译成 404）。"""
    with pytest.raises(ValueError):
        domain.plan_status("ghost-plan")


def test_update_plan_step_on_missing_step_raises_value_error(
    domain: HarnessDomain,
) -> None:
    """更新不存在的步骤抛 ValueError。"""
    with pytest.raises(ValueError):
        domain.update_plan_step("ghost-step", text="不存在的步骤")


def test_update_plan_step_returns_status_and_cascade(domain: HarnessDomain) -> None:
    """更新步骤返回 {status, step_id, cascade}。"""
    created = domain.create_plan(TASK)
    step_id = created["steps"][0]["step_id"]

    result = domain.update_plan_step(step_id, status="in_progress")

    assert result["status"] == "updated"
    assert result["step_id"] == step_id
    assert result["cascade"] is not None
    assert "affected_steps" in result["cascade"]
    assert "action" in result["cascade"]


def test_plan_cascade_reports_affected_steps(domain: HarnessDomain) -> None:
    """plan_cascade 返回受影响步骤集合。"""
    created = domain.create_plan(TASK)
    step_id = created["steps"][0]["step_id"]

    result = domain.plan_cascade(created["plan_id"], step_id)

    assert "affected_steps" in result
    assert "action" in result


def test_plan_cascade_unknown_step_raises_value_error(domain: HarnessDomain) -> None:
    """级联起点不在该 Plan 内时抛 ValueError。"""
    created = domain.create_plan(TASK)

    with pytest.raises(ValueError):
        domain.plan_cascade(created["plan_id"], "ghost-step")


# ════════════════════════════════════════════════════════════════
# Memory Tagger
# ════════════════════════════════════════════════════════════════


def test_memory_tag_classifies_chinese_constraint(domain: HarnessDomain) -> None:
    """中文约束句被分类到 constraint 层。"""
    result = domain.memory_tag("用户明确要求所有技术文档必须使用中文")

    assert result["layer"] == MemoryLayer.CONSTRAINT.value
    assert result["tags"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["content"] == "用户明确要求所有技术文档必须使用中文"


def test_memory_query_filters_by_layer(domain: HarnessDomain) -> None:
    """memory_query 按层级过滤。"""
    domain.storage.insert_memory(
        MemoryEntry(content="约束记忆内容", tags=["约束"], layer=MemoryLayer.CONSTRAINT),
        source="c.md",
    )
    domain.storage.insert_memory(
        MemoryEntry(content="偏好记忆内容", tags=["偏好"], layer=MemoryLayer.PREFERENCE),
        source="p.md",
    )

    result = domain.memory_query(layer=MemoryLayer.CONSTRAINT)

    assert result["total"] == 1
    assert result["entries"][0]["layer"] == MemoryLayer.CONSTRAINT.value


def test_memory_filter_lambda_zero_returns_only_constraint(
    domain: HarnessDomain,
) -> None:
    """λ=0.0 只返回 constraint 层条目。"""
    domain.storage.insert_memory(
        MemoryEntry(content="约束记忆内容", tags=["约束"], layer=MemoryLayer.CONSTRAINT),
        source="c.md",
    )
    domain.storage.insert_memory(
        MemoryEntry(content="偏好记忆内容", tags=["偏好"], layer=MemoryLayer.PREFERENCE),
        source="p.md",
    )

    result = domain.memory_filter(0.0)

    assert result["included_layers"] == ["constraint"]
    assert result["lambda_value"] == 0.0
    assert result["total"] == 1
    assert all(e["layer"] == MemoryLayer.CONSTRAINT.value for e in result["entries"])


# ════════════════════════════════════════════════════════════════
# Checkpoint Review
# ════════════════════════════════════════════════════════════════


def test_checkpoint_create_then_review_round_trips(domain: HarnessDomain) -> None:
    """checkpoint_create → checkpoint_review 闭环。"""
    created = domain.create_plan(TASK)
    plan_steps = [
        {
            "step_id": step["step_id"],
            "text": step["text"],
            "status": step["status"],
        }
        for step in created["steps"]
    ]

    created_cp = domain.checkpoint_create(
        plan_id=created["plan_id"],
        plan_steps=plan_steps,
        completed_step_ids=[plan_steps[0]["step_id"]],
    )
    checkpoint_id = created_cp["checkpoint"]["checkpoint_id"]
    assert created_cp["checkpoint"]["plan_id"] == created["plan_id"]

    review = domain.checkpoint_review(checkpoint_id)

    assert review["checkpoint_id"] == checkpoint_id
    result = review["review_result"]
    assert result["alignment"] in ("aligned", "partial", "deviated")
    assert result["progress_status"] in ("on_track", "at_risk", "behind")
    assert result["unexpected_impact"] in ("none", "low", "medium", "high")
    assert 0.0 <= result["confidence"] <= 1.0


def test_checkpoint_create_rejects_empty_plan_id(domain: HarnessDomain) -> None:
    """空 plan_id 被拒绝（HTTP 适配层翻译成 422）。"""
    with pytest.raises(ValueError):
        domain.checkpoint_create(plan_id="", plan_steps=[], completed_step_ids=[])


def test_checkpoint_review_unknown_id_raises_value_error(
    domain: HarnessDomain,
) -> None:
    """审查不存在的 Checkpoint 抛 ValueError（HTTP 适配层翻译成 404）。"""
    with pytest.raises(ValueError):
        domain.checkpoint_review("cp_ghost")


# ════════════════════════════════════════════════════════════════
# 纯本地契约：不依赖 FastAPI / HTTP
# ════════════════════════════════════════════════════════════════


def test_importing_domain_does_not_import_fastapi() -> None:
    """在全新子进程里导入 domain 不会拉起 fastapi。"""
    script = (
        "import sys; import harness_server.domain; "
        "assert 'fastapi' not in sys.modules, "
        "sorted(m for m in sys.modules if m.startswith('fastapi'))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
