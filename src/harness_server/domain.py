"""Harness 领域逻辑 — 纯 Python 进程内服务

本模块从 ``harness_server/app.py`` 中抽出全部领域逻辑，不依赖任何 HTTP 框架：

- Memory Tagger 的层级分类规则（LayerRule / LAYER_RULES / classify / extract_tags）
- Plan Engine 的级联修正引擎（cascade_correct / detect_cycle / build_adjacency_list）
- Plan 任务分解与步骤模型构造（_decompose_task / _steps_to_model）
- Checkpoint 快照审查的四个维度判断（_check_alignment_from_snapshot 等）
- Hermes Memory 的 .md 导入（import_hermes_memories / _import_md_file）

在这一切之上，``HarnessDomain`` 提供一个**纯进程内**的服务类：它直接持有一个
``HarnessStorage``，把原来 HTTP 端点里的业务逻辑暴露成普通方法。工具层可以
直接调用这些方法拿到与 HTTP 端点完全一致的 JSON，不需要启动任何服务。

错误约定：所有可预期的失败都抛 ``ValueError``（``HarnessDomainError`` 的子类），
由 HTTP 适配层翻译成 422 / 404 / 409。模块本身不 import fastapi。
"""

from __future__ import annotations

import glob
import logging
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .models import (
    AssociationStrength,
    Checkpoint,
    CompletedStep,
    CreateCheckpointRequest,
    MemoryEntry,
    MemoryLayer,
    OKRPlanStep,
    PlanStatus,
    ReviewResult,
    generate_checkpoint_id,
    now_iso,
)
from .storage import HarnessStorage, default_db_path

logger = logging.getLogger("harness-server")

# ════════════════════════════════════════════════════════════════
# 错误类型：全部是 ValueError，HTTP 适配层据此选择状态码
# ════════════════════════════════════════════════════════════════


class HarnessDomainError(ValueError):
    """领域层可预期错误的基类。

    继承 ``ValueError``，这样调用方既可以捕获具体的子类，也可以统一按
    ``ValueError`` 处理；HTTP 适配层把它翻译成 4xx 而不是 500。
    """


class DomainNotFound(HarnessDomainError):
    """目标实体不存在 → HTTP 404。"""


class DomainConflict(HarnessDomainError):
    """状态冲突（如检测到循环依赖）→ HTTP 409。"""


# ════════════════════════════════════════════════════════════════
# 第一部分：Memory Tagger 标签分类逻辑
# （迁移自 memory-tagger/tagger.py）
# ════════════════════════════════════════════════════════════════


@dataclass
class LayerRule:
    """单条层级匹配规则

    通过关键词匹配判断记忆内容属于哪个层级。

    Attributes:
        layer: 该规则对应的记忆层级
        keywords: 触发该层级的关键词列表
        weight: 匹配权重，用于置信度计算（权重越高，该层级优先级越高）
    """

    layer: MemoryLayer
    keywords: List[str]
    weight: float = 1.0


# 层级匹配规则表（关键词按优先级排列）
# 每条规则中的关键词越多，匹配的置信度越高
# 这个表是 Memory Tagger 的核心——它定义了如何从文本内容推断记忆层级
LAYER_RULES: Dict[MemoryLayer, LayerRule] = {
    # 约束层：安全/合规相关，权重最高（2.0），因为约束不可遗忘
    MemoryLayer.CONSTRAINT: LayerRule(
        layer=MemoryLayer.CONSTRAINT,
        keywords=[
            "不能", "不要", "必须", "禁止", "严禁",
            "不得", "不可", "不允许", "一定不要", "绝对不要",
            "should not", "must not", "never",
            "禁止使用", "禁用", "不允许使用",
        ],
        weight=2.0,
    ),
    # 偏好层：技术选择和习惯
    MemoryLayer.PREFERENCE: LayerRule(
        layer=MemoryLayer.PREFERENCE,
        keywords=[
            "喜欢", "偏好", "习惯", "倾向于",
            "prefer", "preferred", "favorite",
            "喜欢用", "习惯用", "偏好使用",
            "更倾向于", "更加喜欢",
        ],
        weight=1.0,
    ),
    # 风格层：写作/编码风格
    MemoryLayer.STYLE: LayerRule(
        layer=MemoryLayer.STYLE,
        keywords=[
            "风格", "格式", "方式", "样式",
            "style", "format", "manner",
            "写作风格", "编码风格", "语言风格",
            "格式要求", "排版风格",
        ],
        weight=1.0,
    ),
    # 决策层：历史决策和结论
    MemoryLayer.DECISION: LayerRule(
        layer=MemoryLayer.DECISION,
        keywords=[
            "决定", "结论", "确认", "选定",
            "decide", "decision", "conclusion", "confirmed",
            "最终决定", "最终结论", "达成一致",
            "我们决定", "决定采用", "确定使用",
        ],
        weight=1.0,
    ),
    # 模式层：成功经验和规律
    MemoryLayer.PATTERN: LayerRule(
        layer=MemoryLayer.PATTERN,
        keywords=[
            "每次", "经常", "总是", "往往",
            "usually", "always", "often", "typically",
            "每次都要", "通常需要", "一般都是",
            "常规做法", "习惯做法",
        ],
        weight=1.0,
    ),
}


def _find_matches(content: str) -> List[Tuple[MemoryLayer, int]]:
    """在内容中查找所有匹配的关键词

    遍历所有层级的规则，统计每个层级匹配到的关键词数量。

    Args:
        content: 要分析的记忆内容文本

    Returns:
        匹配结果列表，每项是 (层级, 匹配关键词数)，只返回匹配数 > 0 的层级
    """
    matches: List[Tuple[MemoryLayer, int]] = []
    for layer, rule in LAYER_RULES.items():
        count = 0
        for kw in rule.keywords:
            # 不区分大小写的全文匹配
            if re.search(re.escape(kw), content, re.IGNORECASE):
                count += 1
        if count > 0:
            matches.append((layer, count))
    return matches


def classify(content: str) -> Tuple[MemoryLayer, List[str], float]:
    """对记忆内容进行层级分类

    这是 Memory Tagger 的核心函数。通过关键词匹配判断内容属于哪个层级。

    算法流程：
    1. 查找所有匹配的关键词
    2. 按加权匹配数排序（score = count × weight）
    3. 取分数最高的层级作为分类结果
    4. 置信度 = 最高分 / 总分

    Args:
        content: 记忆内容文本

    Returns:
        元组 (layer, matched_keywords, confidence):
        - layer: 最匹配的记忆层级（无匹配时默认 PATTERN）
        - matched_keywords: 命中的关键词列表
        - confidence: 分类置信度 [0, 1]
    """
    matches = _find_matches(content)

    if not matches:
        # 无匹配时默认归入 pattern 层（可降级到其他默认层）
        return MemoryLayer.PATTERN, [], 0.0

    # 按加权匹配数排序：score = count * weight
    scored = [
        (layer, count, LAYER_RULES[layer].weight * count)
        for layer, count in matches
    ]
    scored.sort(key=lambda x: x[2], reverse=True)

    best_layer, best_count, best_score = scored[0]
    total_score = sum(s for _, _, s in scored)

    # 提取命中层级的关键词
    matched_kws: List[str] = []
    for kw in LAYER_RULES[best_layer].keywords:
        if re.search(re.escape(kw), content, re.IGNORECASE):
            matched_kws.append(kw)

    # 置信度 = 最高分 / 总分（单一规则匹配时接近 1.0）
    confidence = best_score / total_score if total_score > 0 else 0.0

    # 至少匹配到 1 个关键词时置信度不低于 0.3
    if best_count > 0 and confidence < 0.3:
        confidence = max(confidence, 0.3)

    return best_layer, matched_kws, round(confidence, 4)


def extract_tags(content: str) -> List[str]:
    """从内容中提取标签（关键词级别的标记）

    与 classify 不同，这个函数返回所有层级命中的关键词（不区分层级），
    用于给记忆打多个标签方便后续检索。

    Args:
        content: 记忆内容文本

    Returns:
        去重后的关键词列表
    """
    tags: List[str] = []
    for rule in LAYER_RULES.values():
        for kw in rule.keywords:
            if re.search(re.escape(kw), content, re.IGNORECASE):
                tags.append(kw)
    return tags


def detect_task_type(task_prompt: str) -> str:
    """检测任务类型：convergent / divergent / mixed

    通过关键词匹配判断任务属于哪种类型，用于推荐合适的 λ 值。

    - convergent（收敛）：修复/部署/排查等工程任务，需要全部历史约束
    - divergent（发散）：构思/设计/写作等创意任务，不注入偏好避免限制
    - mixed（混合）：方案/策略/分析等，平衡约束与自由度

    Args:
        task_prompt: 任务描述文本

    Returns:
        任务类型字符串："convergent" / "divergent" / "mixed"
    """
    # 任务类型关键词映射表
    config_lookup = {
        "convergent": ["修复", "部署", "排查", "调试", "构建", "发布", "合并", "删除"],
        "divergent": ["构思", "设计", "写作", "批判", "探索", "头脑风暴", "创意", "灵感"],
        "mixed": ["方案", "策略", "分析", "评估", "对比", "评审", "规划"],
    }

    scores = {"convergent": 0, "divergent": 0, "mixed": 0}
    for task_type, keywords in config_lookup.items():
        for kw in keywords:
            if re.search(re.escape(kw), task_prompt, re.IGNORECASE):
                scores[task_type] += 1

    # 无法识别时默认 mixed
    if all(v == 0 for v in scores.values()):
        return "mixed"

    return max(scores, key=scores.get)


# ════════════════════════════════════════════════════════════════
# 第二部分：Plan Engine 级联修正引擎
# （迁移自 plan-engine/engine.py）
# ════════════════════════════════════════════════════════════════


def cascade_correct(
    plan_id: str,
    modified_step_id: str,
    plan_steps: Dict[str, OKRPlanStep],
    store: HarnessStorage,
) -> Dict:
    """级联修正引擎核心算法

    当某个步骤被修改后，沿依赖图 BFS 遍历，按关联强度分级处理影响。

    算法流程：
    1. 从 modified_step_id 出发
    2. 反向查找：谁依赖我？（搜索 dependency_ids 包含当前步骤的步骤）
    3. 正向查找：我有哪些子步骤？（搜索 parent_id 等于当前步骤的步骤）
    4. 按关联强度分级处理：
       - STRONG → pending_review（需人工审查）
       - MODERATE → pending_review（浅审查）
       - WEAK → notify（仅通知，不改变状态）
    5. 持久化状态变更

    Args:
        plan_id: Plan 唯一标识
        modified_step_id: 被修改的步骤 ID（级联修正的起点）
        plan_steps: Plan 的所有步骤字典 {step_id: OKRPlanStep}
        store: 存储层实例（用于持久化状态变更）

    Returns:
        字典 {"affected_steps": [...], "action": {step_id: "pending_review"|"notify"}}
    """
    affected: Dict[str, str] = {}
    visited: set = set()
    queue: List[str] = [modified_step_id]

    while queue:
        current_id = queue.pop(0)
        if current_id in visited:
            continue
        visited.add(current_id)

        # 反向查找：搜索所有依赖此步骤的步骤（谁依赖我？）
        for sid, s in plan_steps.items():
            if current_id in s.dependency_ids:
                if sid not in visited:
                    queue.append(sid)

                # 按关联强度分级处理
                if s.association_strength == AssociationStrength.STRONG:
                    affected[sid] = "pending_review"
                    # 已完成步骤不得被级联降级（TC-PLAN-009：不静默改已完成）
                    if s.status != PlanStatus.COMPLETED:
                        s.status = PlanStatus.PENDING_REVIEW
                elif s.association_strength == AssociationStrength.MODERATE:
                    affected[sid] = "pending_review"
                    if s.status != PlanStatus.COMPLETED:
                        s.status = PlanStatus.PENDING_REVIEW
                else:
                    # WEAK 关联仅通知，不改变状态
                    affected[sid] = "notify"

        # 正向查找：搜索子步骤（我依赖谁？向下游传播）
        for sid, s in plan_steps.items():
            if s.parent_id == current_id:
                if sid not in visited:
                    queue.append(sid)
                affected[sid] = "notify"

    # 持久化受影响步骤的状态变更（只有 pending_review 需要写库，notify 不改状态）
    # 已完成步骤不得被降级：写库时用 WHERE status != 'completed' 守卫（TC-PLAN-009）
    if affected:
        conn = store._connection()
        try:
            for sid, action in affected.items():
                if action == "pending_review":
                    conn.execute(
                        "UPDATE steps SET status = ? "
                        "WHERE step_id = ? AND status != ?",
                        (PlanStatus.PENDING_REVIEW.value, sid, PlanStatus.COMPLETED.value),
                    )
            conn.commit()
        finally:
            conn.close()

        # 更新 Plan 时间戳
        store.update_plan_timestamp(plan_id)

    return {"affected_steps": list(affected.keys()), "action": affected}


def detect_cycle(steps: List[OKRPlanStep]) -> bool:
    """使用拓扑排序（Kahn 算法）检测循环依赖

    如果依赖图中存在环，返回 True。

    算法：
    1. 构建入度表和邻接表
    2. 将入度为 0 的节点入队
    3. 每次出队一个节点，将其邻居入度减 1
    4. 如果最终访问的节点数 ≠ 总节点数，说明有环

    Args:
        steps: 步骤列表

    Returns:
        True 表示存在循环依赖，False 表示无环
    """
    # 构建入度表 + 邻接表
    in_degree: Dict[str, int] = {}
    adj: Dict[str, List[str]] = {}
    step_ids = {s.step_id for s in steps}

    for s in steps:
        in_degree.setdefault(s.step_id, 0)
        adj.setdefault(s.step_id, [])

    for s in steps:
        for dep_id in s.dependency_ids:
            # 只统计在同一 plan 内的依赖
            if dep_id in step_ids:
                adj.setdefault(dep_id, []).append(s.step_id)
                in_degree[s.step_id] = in_degree.get(s.step_id, 0) + 1

    # Kahn 算法：BFS 拓扑排序
    queue = [sid for sid, deg in in_degree.items() if deg == 0]
    visited_count = 0

    while queue:
        node = queue.pop(0)
        visited_count += 1
        for neighbor in adj.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # 如果访问的节点数 ≠ 总节点数，说明有环
    return visited_count != len(steps)


def build_adjacency_list(steps: List[OKRPlanStep]) -> Dict[str, List[str]]:
    """构建依赖图邻接表（正向）

    邻接表格式: {step_id: [依赖于它的 step_id 列表]}
    例如: {"s1": ["s2", "s3"]} 表示 s2 和 s3 都依赖 s1

    Args:
        steps: 步骤列表

    Returns:
        邻接表字典
    """
    adj: Dict[str, List[str]] = {}
    for s in steps:
        adj.setdefault(s.step_id, [])
    for s in steps:
        for dep_id in s.dependency_ids:
            adj.setdefault(dep_id, []).append(s.step_id)
    return adj


# ════════════════════════════════════════════════════════════════
# 第三部分：Plan 辅助函数
# （迁移自 plan-engine/server.py）
# ════════════════════════════════════════════════════════════════


def _decompose_task(task_description: str) -> List[Dict]:
    """将任务描述分解为 3-8 个步骤

    这是一个基于规则的任务分解器（不调用 LLM）：
    1. 优先按序号列表分割（"1. xxx\n2. yyy"）
    2. 如果结果太少或太多，按句号/分号分割
    3. 如果还是太少，按逗号分割
    4. 最终保证在 3-8 个步骤之间

    Args:
        task_description: 任务描述文本

    Returns:
        步骤字典列表，每项 {"text": "...", "key": "S1"}
    """
    lines = [l.strip() for l in task_description.strip().split("\n") if l.strip()]
    steps_raw = []

    # 尝试按序号分割（"1. xxx" 或 "1、xxx"）
    numbered = False
    for line in lines:
        m = re.match(r"^(\d+)[.、．\s]+(.+)$", line)
        if m:
            numbered = True
            steps_raw.append(m.group(2))
        elif not numbered and line:
            steps_raw.append(line)

    # 如果结果太少或太多，按句子重新分割
    if len(steps_raw) < 3 or len(steps_raw) > 8:
        sentences = re.split(r"[。；;]", task_description)
        steps_raw = [s.strip() for s in sentences if len(s.strip()) > 4]

    # 如果还是太少，按逗号分割
    if len(steps_raw) < 3:
        chunks = re.split(r"[，,]", task_description)
        steps_raw = [c.strip() for c in chunks if len(c.strip()) > 4]

    # 保证在 3-8 个步骤之间
    if len(steps_raw) < 3:
        steps_raw = [f"步骤 {i+1}: {task_description}" for i in range(3)]
    if len(steps_raw) > 8:
        steps_raw = steps_raw[:8]

    # 生成步骤字典
    result = []
    for i, text in enumerate(steps_raw):
        result.append({
            "text": text.strip(),
            "key": f"S{i+1}",
        })
    return result


def _steps_to_model(steps_data: List[Dict]) -> List[OKRPlanStep]:
    """将步骤字典列表转换为 OKRPlanStep 对象列表

    自动设置：
    - step_id（UUID）
    - dependency_ids（按顺序链式链接：step_2 依赖 step_1）
    - association_strength（按位置：第 1 步 STRONG，第 2 步 MODERATE，之后 WEAK）

    Args:
        steps_data: 步骤字典列表

    Returns:
        OKRPlanStep 对象列表
    """
    steps: List[OKRPlanStep] = []
    for i, sd in enumerate(steps_data):
        step_id = str(uuid.uuid4())
        dep_ids = []
        # 按顺序链接：每个步骤依赖前一个步骤
        if i > 0:
            dep_ids.append(steps[i - 1].step_id)

        # 关联强度：第 1 步 STRONG，第 2 步 MODERATE，之后 WEAK
        strength = AssociationStrength.STRONG
        if i >= 2:
            strength = AssociationStrength.WEAK
        elif i >= 1:
            strength = AssociationStrength.MODERATE

        step = OKRPlanStep(
            step_id=step_id,
            text=sd["text"],
            key=sd.get("key", ""),
            status=PlanStatus.PENDING,
            dependency_ids=dep_ids,
            association_strength=strength,
        )
        steps.append(step)

    return steps


# ════════════════════════════════════════════════════════════════
# 第四部分：Checkpoint 辅助函数
# （迁移自 checkpoint-review/server.py）
# ════════════════════════════════════════════════════════════════


def _build_completed_summary(request: CreateCheckpointRequest) -> List[CompletedStep]:
    """从请求中提取已完成的步骤摘要

    Args:
        request: 创建 Checkpoint 的请求

    Returns:
        已完成步骤的摘要列表
    """
    step_map = {s.step_id: s for s in request.plan_steps}
    summary: List[CompletedStep] = []
    for sid in request.completed_step_ids:
        step = step_map.get(sid)
        if step:
            summary.append(CompletedStep(
                step_id=sid,
                text=step.text,
                result="",
            ))
    return summary


def _build_remaining_plan(request: CreateCheckpointRequest) -> List[str]:
    """提取剩余（未完成）的 Plan 步骤描述

    Args:
        request: 创建 Checkpoint 的请求

    Returns:
        剩余步骤描述列表，格式如 "[step_id] 步骤文本"
    """
    completed_set = set(request.completed_step_ids)
    return [
        f"[{s.step_id}] {s.text}" for s in request.plan_steps
        if s.step_id not in completed_set
    ]


def _build_current_goal(request: CreateCheckpointRequest) -> str:
    """推断当前正在执行的目标

    取第一个未完成的步骤作为当前目标；如果全部完成则返回最后一步。

    Args:
        request: 创建 Checkpoint 的请求

    Returns:
        当前目标描述文本
    """
    completed_set = set(request.completed_step_ids)
    for s in request.plan_steps:
        if s.step_id not in completed_set:
            return s.text
    # 全部完成 → 取最后一步的文本
    if request.plan_steps:
        return request.plan_steps[-1].text
    return ""


def _check_alignment_from_snapshot(checkpoint: Checkpoint) -> str:
    """基于快照检查目标对齐程度

    审查时没有原始 plan_steps，用 completed_steps 和 unexpected_findings 判断：
    - 有 completed_steps 但无 remaining_plan → 全部完成 → aligned
    - 有 completed_steps 且有 remaining_plan → 正在执行 → aligned
    - 有 unexpected_findings → 可能走偏了 → partial

    Args:
        checkpoint: Checkpoint 快照

    Returns:
        对齐程度: "aligned" | "partial" | "deviated"
    """
    if not checkpoint.completed_steps_summary:
        return "aligned"  # 还没开始，无所谓偏离

    if checkpoint.unexpected_findings:
        # 有意外发现 → 可能走偏了
        return "partial"

    return "aligned"


def _check_progress_from_snapshot(checkpoint: Checkpoint) -> str:
    """基于快照检查进度状态

    用已完成步骤数 / (已完成 + 剩余) 的比率估算：
    - ratio >= 0.5 → on_track
    - 0.25 <= ratio < 0.5 → at_risk
    - ratio < 0.25 → behind

    Args:
        checkpoint: Checkpoint 快照

    Returns:
        进度状态: "on_track" | "at_risk" | "behind"
    """
    done = len(checkpoint.completed_steps_summary)
    remaining = len(checkpoint.remaining_plan)
    total = done + remaining

    if total == 0:
        return "on_track"

    ratio = done / total
    if ratio >= 0.5:
        return "on_track"
    elif ratio >= 0.25:
        return "at_risk"
    else:
        return "behind"


def _compute_unexpected_impact(checkpoint: Checkpoint) -> str:
    """评估意外发现的影响程度

    判断规则：
    - 无意外发现 → none
    - 包含严重关键词（安全/漏洞/崩溃等）→ high
    - 发现条数 >= 3 → medium
    - 其他 → low

    Args:
        checkpoint: Checkpoint 快照

    Returns:
        影响程度: "none" | "low" | "medium" | "high"
    """
    findings = checkpoint.unexpected_findings or []
    if not findings:
        return "none"

    # 严重关键词列表（中英文）
    severe_keywords = {"security", "vulnerability", "crash", "data loss",
                       "安全", "漏洞", "崩溃", "数据丢失"}
    combined = " ".join(f.lower() for f in findings)
    for kw in severe_keywords:
        if kw.lower() in combined:
            return "high"

    if len(findings) >= 3:
        return "medium"

    return "low"


def _adjustments_from_snapshot(
    checkpoint: Checkpoint,
    alignment: str,
    progress: str,
    impact: str,
) -> List[str]:
    """从快照生成剩余计划调整建议

    根据审查结果生成可操作的调整建议列表。

    Args:
        checkpoint: Checkpoint 快照
        alignment: 目标对齐程度
        progress: 进度状态
        impact: 意外发现影响

    Returns:
        调整建议字符串列表
    """
    adjustments: List[str] = []

    if alignment == "partial":
        adjustments.append("部分步骤可能偏离原始目标，建议复审已完成的步骤内容")

    if progress == "behind":
        adjustments.append("进度落后，建议考虑缩减范围或增加资源")

    if impact == "high":
        adjustments.append("安全/合规风险较高，建议暂停并优先处理意外发现")
    elif impact == "medium":
        adjustments.append("存在多项意外发现，建议评估后再继续")

    if checkpoint.remaining_plan:
        adjustments.append(
            f"剩余 {len(checkpoint.remaining_plan)} 步: "
            f"{'; '.join(checkpoint.remaining_plan[:3])}"
            f"{'...' if len(checkpoint.remaining_plan) > 3 else ''}"
        )

    return adjustments


def _compute_confidence(alignment: str, progress: str, impact: str) -> float:
    """基于快照信息完整度计算审查置信度

    置信度反映审查结果的可信程度：
    - aligned + none impact → 0.90（一切顺利，信息明确）
    - partial + high impact → 0.70（情况复杂，信息不足）
    - deviated → 0.60（偏离严重，需人工介入）
    - 其他 → 0.85（默认置信度）

    Args:
        alignment: 目标对齐程度
        progress: 进度状态（当前未使用，预留扩展）
        impact: 意外发现影响

    Returns:
        置信度 [0, 1]
    """
    if alignment == "aligned" and impact == "none":
        return 0.90
    if alignment == "partial" and impact == "high":
        return 0.70
    if alignment == "deviated":
        return 0.60
    return 0.85


# ════════════════════════════════════════════════════════════════
# 第五部分：Hermes Memory 导入逻辑
# （从 memory-tagger/storage.py 迁移到此处，因为需要 classify/extract_tags）
# ════════════════════════════════════════════════════════════════


def import_hermes_memories(
    memories_dir: str,
    store: HarnessStorage,
) -> Tuple[int, int, int]:
    """从 Hermes Memory 目录导入 .md 文件

    读取指定目录下的所有 .md 文件，将内容分类后批量插入数据库。
    这个函数放在 server.py 而不是 storage.py，因为它需要 classify() 和
    extract_tags() 函数（避免 storage.py ↔ server.py 循环导入）。

    解析策略：
    - 按空行分割段落
    - 跳过标题行（# 开头）、空行、过短片段
    - 每段落用 classify() 分类，用 extract_tags() 提取标签
    - 批量插入数据库

    Args:
        memories_dir: Hermes Memory 目录路径
        store: 存储层实例

    Returns:
        元组 (imported, skipped, total_files):
        - imported: 成功插入的条数
        - skipped: 跳过的文件数
        - total_files: 扫描的文件总数
    """
    if not os.path.isdir(memories_dir):
        logger.warning("Hermes Memory 目录不存在: %s", memories_dir)
        return 0, 0, 0

    md_files = glob.glob(os.path.join(memories_dir, "*.md"))
    total_files = len(md_files)

    imported = 0
    skipped = 0

    for filepath in md_files:
        try:
            ok, count = _import_md_file(filepath, store)
            if ok:
                imported += count
            else:
                skipped += 1
        except Exception as e:
            logger.warning("导入文件失败 %s: %s", filepath, e)
            skipped += 1

    logger.info(
        "Hermes 导入完成: %d 条插入, %d 文件跳过, 共 %d 文件",
        imported, skipped, total_files,
    )
    return imported, skipped, total_files


def _import_md_file(filepath: str, store: HarnessStorage) -> Tuple[bool, int]:
    """导入单个 .md 文件中的记忆

    Args:
        filepath: .md 文件路径
        store: 存储层实例

    Returns:
        元组 (ok, count): ok=是否成功处理, count=插入条数
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    # 按空行分割成段落
    blocks = re.split(r"\n\s*\n", text)
    entries: List[MemoryEntry] = []
    source_name = os.path.basename(filepath)

    for block in blocks:
        block = block.strip()
        # 跳过空行、标题行、过短片段
        if not block or len(block) < 10:
            continue
        if block.startswith("#"):
            continue

        # 用 tagger 函数分类和提取标签
        tags = extract_tags(block)
        layer, _, _ = classify(block)

        entry = MemoryEntry(
            content=block[:500],  # 截断超长内容
            tags=tags,
            layer=layer,
            created_at=datetime.now(timezone.utc),
        )
        entries.append(entry)

    if entries:
        count = store.bulk_insert_memories(entries, source=source_name)
        return True, count

    return True, 0


# ════════════════════════════════════════════════════════════════
# 第六部分：HarnessDomain —— 纯进程内服务
# ════════════════════════════════════════════════════════════════

#: 记忆查询/过滤的默认条数上限（与 HTTP 请求模型一致）
DEFAULT_MEMORY_LIMIT = 50
#: 记忆查询/过滤的条数上限（与 HTTP 请求模型一致）
MAX_MEMORY_LIMIT = 500


def _coerce_status(value: Any) -> PlanStatus:
    """把状态值转成 PlanStatus 枚举

    接受枚举本身或字符串（"in_progress"），非法值抛 ValueError。
    """
    if isinstance(value, PlanStatus):
        return value
    try:
        return PlanStatus(value)
    except ValueError as exc:
        raise ValueError(f"非法步骤状态: {value}") from exc


def _coerce_strength(value: Any) -> AssociationStrength:
    """把关联强度转成 AssociationStrength 枚举"""
    if isinstance(value, AssociationStrength):
        return value
    try:
        return AssociationStrength(value)
    except ValueError as exc:
        raise ValueError(f"非法关联强度: {value}") from exc


def _coerce_layer(value: Any) -> Optional[MemoryLayer]:
    """把层级转成 MemoryLayer 枚举，None 原样返回"""
    if value is None or isinstance(value, MemoryLayer):
        return value
    try:
        return MemoryLayer(value)
    except ValueError as exc:
        raise ValueError(f"非法记忆层级: {value}") from exc


def _step_to_dict(step: OKRPlanStep) -> Dict[str, Any]:
    """把 OKRPlanStep 转成可直接 JSON 序列化的字典"""
    return step.model_dump(mode="json")


def _memory_to_dict(entry: MemoryEntry) -> Dict[str, Any]:
    """把 MemoryEntry 转成可直接 JSON 序列化的字典"""
    return entry.model_dump(mode="json")


def _normalize_plan_step(step: Any) -> Dict[str, Any]:
    """把各种形态的 plan 步骤归一化成字典

    支持 dict、DictStep、OKRPlanStep 三种输入。
    """
    if isinstance(step, dict):
        return dict(step)
    dump = getattr(step, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    raise ValueError(f"无法解析的 plan 步骤: {step!r}")


class HarnessDomain:
    """Harness 领域服务 —— 纯进程内，不经过任何 HTTP。

    直接持有一个 :class:`~harness_server.storage.HarnessStorage`，把原来
    FastAPI 端点里的业务逻辑暴露成普通方法。返回结构与 HTTP 端点的响应
    模型完全一致（JSON 层面逐字段相同），因此工具层既可以走 HTTP，也可以
    直接调用本类。

    使用方法::

        domain = HarnessDomain()                        # 用默认数据库
        domain = HarnessDomain(db_path="/tmp/h.db")     # 指定数据库
        domain = HarnessDomain(storage=store)           # 复用已有存储层

    错误约定：实体不存在抛 :class:`DomainNotFound`（ValueError 子类），
    参数非法抛 :class:`ValueError`，状态冲突抛 :class:`DomainConflict`。
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        *,
        storage: Optional[HarnessStorage] = None,
        admission=None,
    ) -> None:
        """初始化域服务

        Args:
            db_path: SQLite 数据库路径；None 时用 ``default_db_path()``
            storage: 直接复用已有存储层（优先于 db_path）
            admission: 可选的记忆写入闸门 ``admission(text, *, used_chars)``，
                由调用方构造（工具层注入 ``memory_hygiene.admit`` 的包装）。
                构造存储层时原样传入；复用已有 ``storage`` 时挂到它上面。
                None 表示不设闸门，与旧行为一致。

                域层只转发这个依赖，不 import 分类器：``harness_server`` 不能
                反向依赖 ``deepseek_harness``。HF 讨论
                ``deepseek-ai/DeepSeek-V4.1-Flash#46`` 显示同一会话在 V4.1
                Flash 上写出 161 KB / 11 个记忆文件，而 V4 Flash 是 13 KB /
                9 个（12 倍），且系统提示词规则对膨胀零效果——闸门必须落在
                写入路径上。
        """
        if storage is not None:
            self.storage = storage
            if admission is not None:
                self.storage._admission = admission
        else:
            self.storage = HarnessStorage(
                db_path=db_path or default_db_path(), admission=admission
            )
        self.db_path = self.storage.db_path

    # ── Plan Engine ──────────────────────────────────────────

    def create_plan(self, task_description: str) -> Dict[str, Any]:
        """从任务描述创建 OKR PlanStep 列表

        对应 ``POST /plan/create``。

        Args:
            task_description: 任务描述文本

        Returns:
            {"plan_id": str, "steps": [步骤字典, ...]}

        Raises:
            ValueError: 任务描述为空
            DomainConflict: 检测到循环依赖
        """
        if not task_description or not task_description.strip():
            raise ValueError("任务描述不能为空")

        # 分解任务为步骤
        steps_data = _decompose_task(task_description)
        steps = _steps_to_model(steps_data)

        # 检测循环依赖
        if detect_cycle(steps):
            raise DomainConflict("检测到循环依赖，请重新描述任务")

        # 生成 plan_id 并持久化
        plan_id = str(uuid.uuid4())
        self.storage.create_plan(plan_id)
        self.storage.insert_steps(plan_id, steps)

        logger.info("创建 Plan: %s, 步骤数: %d", plan_id, len(steps))

        return {
            "plan_id": plan_id,
            "steps": [_step_to_dict(s) for s in steps],
        }

    def update_plan_step(self, step_id: str, **fields: Any) -> Dict[str, Any]:
        """更新步骤状态/内容 + 自动级联修正

        对应 ``PUT /plan/step/{step_id}``。

        Args:
            step_id: 步骤唯一标识
            **fields: 要更新的字段（text / key / status / parent_id /
                dependency_ids / association_strength），只更新传入的非 None 值

        Returns:
            {"status": "updated", "step_id": str, "cascade": {...} | None}

        Raises:
            DomainNotFound: 步骤不存在或所属 Plan 不存在
            ValueError: 状态值非法或状态转换非法
        """
        # 验证步骤存在
        existing = self.storage.get_step(step_id)
        if not existing:
            raise DomainNotFound(f"步骤不存在: {step_id}")

        # 获取 plan_id
        plan_id = self.storage.get_step_plan_id(step_id)
        if not plan_id:
            raise DomainNotFound(f"步骤所属 Plan 不存在: {step_id}")

        payload = {k: v for k, v in fields.items() if v is not None}
        if "status" in payload:
            payload["status"] = _coerce_status(payload["status"])
        if "association_strength" in payload:
            payload["association_strength"] = _coerce_strength(
                payload["association_strength"]
            )

        # 更新字段
        updated = self.storage.update_step(step_id, **payload)

        if updated:
            self.storage.update_plan_timestamp(plan_id)
            logger.info("步骤更新成功: %s", step_id)

            # 自动触发级联修正（如果状态变更或内容变更）
            if payload.get("status") or payload.get("text"):
                steps = self.storage.get_steps(plan_id)
                steps_dict = {s.step_id: s for s in steps}

                cascade_result = cascade_correct(
                    plan_id=plan_id,
                    modified_step_id=step_id,
                    plan_steps=steps_dict,
                    store=self.storage,
                )

                return {
                    "status": "updated",
                    "step_id": step_id,
                    "cascade": cascade_result,
                }

        return {"status": "updated", "step_id": step_id, "cascade": None}

    def plan_cascade(self, plan_id: str, modified_step_id: str) -> Dict[str, Any]:
        """级联修正引擎

        对应 ``POST /plan/cascade``。

        Args:
            plan_id: Plan 唯一标识
            modified_step_id: 被修改的步骤 ID（级联修正的起点）

        Returns:
            {"affected_steps": [step_id, ...], "action": {step_id: 动作}}

        Raises:
            DomainNotFound: Plan 不存在、无步骤或起点步骤不属于该 Plan
        """
        steps = self.storage.get_steps(plan_id)
        if not steps:
            raise DomainNotFound(f"Plan 不存在或无步骤: {plan_id}")

        # 验证 modified_step_id 存在
        step_ids = {s.step_id for s in steps}
        if modified_step_id not in step_ids:
            raise DomainNotFound(f"步骤不存在于该 Plan: {modified_step_id}")

        steps_dict = {s.step_id: s for s in steps}
        result = cascade_correct(
            plan_id=plan_id,
            modified_step_id=modified_step_id,
            plan_steps=steps_dict,
            store=self.storage,
        )

        logger.info(
            "级联修正: plan=%s, modified=%s, affected=%d",
            plan_id, modified_step_id, len(result["affected_steps"]),
        )

        return result

    def plan_status(
        self, plan_id: str, include_archived: bool = False,
    ) -> Dict[str, Any]:
        """获取 Plan 状态和依赖图

        对应 ``GET /plan/status/{plan_id}``。

        Args:
            plan_id: Plan 唯一标识
            include_archived: 是否包含已归档 Plan（默认 False，TC-PLAN-012）

        Returns:
            {"plan_id", "steps", "adjacency_list", "created_at", "updated_at"}

        Raises:
            DomainNotFound: Plan 不存在
        """
        meta = self.storage.get_plan_meta(plan_id, include_archived=include_archived)
        if not meta:
            raise DomainNotFound(f"Plan 不存在: {plan_id}")

        steps = self.storage.get_steps(plan_id)
        adjacency_list = build_adjacency_list(steps)

        return {
            "plan_id": plan_id,
            "steps": [_step_to_dict(s) for s in steps],
            "adjacency_list": adjacency_list,
            "created_at": meta["created_at"],
            "updated_at": meta["updated_at"],
        }

    # ── Memory Tagger ────────────────────────────────────────

    def memory_tag(self, content: str) -> Dict[str, Any]:
        """输入内容 → 返回层级标签

        对应 ``POST /memory/tag``。

        Args:
            content: 记忆内容文本

        Returns:
            {"content", "tags", "layer", "confidence"}

        Raises:
            ValueError: 内容为空
        """
        if not content:
            raise ValueError("记忆内容不能为空")

        layer, matched_kws, confidence = classify(content)
        tags = extract_tags(content)

        return {
            "content": content,
            "tags": tags,
            "layer": layer.value,
            "confidence": confidence,
        }

    def memory_query(
        self,
        tags: Optional[Iterable[str]] = None,
        layer: Optional[Any] = None,
        limit: int = DEFAULT_MEMORY_LIMIT,
    ) -> Dict[str, Any]:
        """按标签/层级查询记忆

        对应 ``POST /memory/query``。

        Args:
            tags: 标签筛选列表（匹配任意一个）
            layer: 层级筛选（MemoryLayer 或层级名字符串）
            limit: 返回条数上限

        Returns:
            {"entries": [记忆字典, ...], "total": int}

        Raises:
            ValueError: 层级非法或 limit 越界
        """
        limit = _validate_limit(limit)
        tag_list = list(tags) if tags else None
        entries = self.storage.query_memories(
            tags=tag_list,
            layer=_coerce_layer(layer),
            limit=limit,
        )
        return {
            "entries": [_memory_to_dict(e) for e in entries],
            "total": len(entries),
        }

    def memory_filter(
        self,
        lambda_value: float,
        limit: int = DEFAULT_MEMORY_LIMIT,
    ) -> Dict[str, Any]:
        """按 λ 值过滤记忆

        对应 ``POST /memory/filter``。

        Args:
            lambda_value: λ 值 ∈ [0, 1]（0.0=仅约束层，1.0=全部）
            limit: 返回条数上限

        Returns:
            {"entries", "total", "lambda_value", "included_layers"}

        Raises:
            ValueError: λ 越界或 limit 越界
        """
        limit = _validate_limit(limit)
        if not 0.0 <= float(lambda_value) <= 1.0:
            raise ValueError(f"lambda 值必须在 [0, 1] 之间: {lambda_value}")

        entries = self.storage.filter_by_lambda(float(lambda_value), limit=limit)
        layers = self.storage.get_layers_for_lambda(float(lambda_value))

        return {
            "entries": [_memory_to_dict(e) for e in entries],
            "total": len(entries),
            "lambda_value": float(lambda_value),
            "included_layers": layers,
        }

    # ── Checkpoint Review ────────────────────────────────────

    def checkpoint_create(
        self,
        plan_id: str,
        plan_steps: Sequence[Any],
        completed_step_ids: Sequence[str],
        unexpected_findings: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """从 Plan 状态提取结构化 Checkpoint 快照

        对应 ``POST /checkpoint/create``。

        Args:
            plan_id: Plan 唯一标识
            plan_steps: Plan 的完整步骤列表（dict / DictStep / OKRPlanStep）
            completed_step_ids: 已完成的步骤 ID 列表
            unexpected_findings: 执行过程中发现的非预期信息

        Returns:
            {"checkpoint": {Checkpoint 字典, ...}}

        Raises:
            ValueError: plan_id 为空或 completed_step_ids 不属于该 Plan
            DomainNotFound: Plan 不存在
        """
        if not plan_id:
            raise ValueError("plan_id 不能为空")

        request = CreateCheckpointRequest(
            plan_id=plan_id,
            plan_steps=[_normalize_plan_step(s) for s in plan_steps],
            completed_step_ids=list(completed_step_ids or []),
            unexpected_findings=list(unexpected_findings or []),
        )

        # 自动计算编号
        next_num = self.storage._get_next_checkpoint_number(plan_id)

        # 构建摘要信息
        completed_summary = _build_completed_summary(request)
        remaining = _build_remaining_plan(request)
        current_goal = _build_current_goal(request)

        # 构建 Checkpoint 对象
        checkpoint = Checkpoint(
            checkpoint_id=generate_checkpoint_id(),
            plan_id=plan_id,
            created_at=now_iso(),
            checkpoint_number=next_num,
            current_goal=current_goal,
            completed_steps_summary=completed_summary,
            unexpected_findings=request.unexpected_findings,
            remaining_plan=remaining,
            purpose="step_threshold",
        )

        # 持久化
        try:
            self.storage.insert_checkpoint(checkpoint)
        except ValueError as exc:
            raise DomainNotFound(str(exc)) from exc

        logger.info(
            "Checkpoint 创建成功: %s (plan=%s, #%d)",
            checkpoint.checkpoint_id,
            checkpoint.plan_id,
            checkpoint.checkpoint_number,
        )

        return {"checkpoint": checkpoint.model_dump(mode="json")}

    def checkpoint_review(self, checkpoint_id: str) -> Dict[str, Any]:
        """对 Checkpoint 执行四维审查

        对应 ``POST /checkpoint/review/{checkpoint_id}``。

        Args:
            checkpoint_id: Checkpoint 唯一标识

        Returns:
            {"checkpoint_id": str, "review_result": {alignment, progress_status,
             unexpected_impact, adjustments, confidence}}

        Raises:
            DomainNotFound: Checkpoint 不存在
        """
        checkpoint = self.storage.get_checkpoint(checkpoint_id)
        if checkpoint is None:
            raise DomainNotFound(f"Checkpoint 不存在: {checkpoint_id}")

        # 基于快照做四维审查
        alignment = _check_alignment_from_snapshot(checkpoint)
        progress = _check_progress_from_snapshot(checkpoint)
        impact = _compute_unexpected_impact(checkpoint)
        adjustments = _adjustments_from_snapshot(checkpoint, alignment, progress, impact)

        review_result = ReviewResult(
            alignment=alignment,
            progress_status=progress,
            unexpected_impact=impact,
            adjustments=adjustments,
            confidence=_compute_confidence(alignment, progress, impact),
        )

        logger.info(
            "审查完成: %s → alignment=%s, progress=%s, impact=%s",
            checkpoint_id,
            alignment,
            progress,
            impact,
        )

        return {
            "checkpoint_id": checkpoint_id,
            "review_result": review_result.model_dump(mode="json"),
        }

    def checkpoint_chain(self, plan_id: str) -> Dict[str, Any]:
        """获取 Plan 的完整 Checkpoint 链

        对应 ``GET /checkpoint/chain/{plan_id}``。

        Args:
            plan_id: Plan 唯一标识

        Returns:
            {"plan_id", "checkpoints": [Checkpoint 字典, ...], "total": int}
        """
        checkpoints = self.storage.get_checkpoint_chain(plan_id)
        return {
            "plan_id": plan_id,
            "checkpoints": [c.model_dump(mode="json") for c in checkpoints],
            "total": len(checkpoints),
        }

    # ── 运行时探针 ───────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """健康检查 —— 返回三组表的统计信息

        对应 ``GET /health``。

        Returns:
            {"status", "plans", "steps", "memories", "checkpoints", "db_path"}
        """
        conn = self.storage._connection()
        try:
            plan_count = conn.execute("SELECT COUNT(*) as cnt FROM plans").fetchone()["cnt"]
            step_count = conn.execute("SELECT COUNT(*) as cnt FROM steps").fetchone()["cnt"]
            memory_count = conn.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()["cnt"]
            checkpoint_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM checkpoints"
            ).fetchone()["cnt"]
            return {
                "status": "ok",
                "plans": plan_count,
                "steps": step_count,
                "memories": memory_count,
                "checkpoints": checkpoint_count,
                "db_path": self.storage.db_path,
            }
        finally:
            conn.close()


def _validate_limit(limit: int) -> int:
    """校验记忆查询/过滤的条数上限"""
    try:
        value = int(limit)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"limit 必须是整数: {limit}") from exc
    if value < 1 or value > MAX_MEMORY_LIMIT:
        raise ValueError(
            f"limit 必须在 1~{MAX_MEMORY_LIMIT} 之间: {limit}"
        )
    return value


__all__ = [
    "DEFAULT_MEMORY_LIMIT",
    "MAX_MEMORY_LIMIT",
    "DomainConflict",
    "DomainNotFound",
    "HarnessDomain",
    "HarnessDomainError",
    "LAYER_RULES",
    "LayerRule",
    "_adjustments_from_snapshot",
    "_build_completed_summary",
    "_build_current_goal",
    "_build_remaining_plan",
    "_check_alignment_from_snapshot",
    "_check_progress_from_snapshot",
    "_compute_confidence",
    "_compute_unexpected_impact",
    "_decompose_task",
    "_find_matches",
    "_import_md_file",
    "_steps_to_model",
    "build_adjacency_list",
    "cascade_correct",
    "classify",
    "detect_cycle",
    "detect_task_type",
    "extract_tags",
    "import_hermes_memories",
]
