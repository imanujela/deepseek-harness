"""Harness Server — 合并后的 FastAPI MCP 服务（薄 HTTP 适配层）

全部领域逻辑位于 ``harness_server.domain``（纯 Python，不依赖 HTTP 框架）。
本文件只做三件事：

1. 加载 config.yaml、构造存储层、执行启动期 Hermes Memory 导入；
2. 把 FastAPI 端点委托给 :class:`~harness_server.domain.HarnessDomain`；
3. 把领域层抛出的 ValueError（DomainNotFound / DomainConflict）翻译成
   与原实现逐端点一致的 HTTPException 状态码（404 / 409 / 422 / 500）。

本文件将 3 个原服务的 FastAPI 应用合并为一个，提供 10 个核心端点 + 1 个健康检查：

Plan Engine（I-06 OKR PlanStep DAG + 级联修正）：
  1. POST /plan/create           — 从任务描述创建 OKR PlanStep 列表
  2. PUT  /plan/step/{step_id}   — 更新步骤状态/内容 + 自动级联修正
  3. POST /plan/cascade          — 级联修正引擎
  4. GET  /plan/status/{plan_id} — 获取 Plan 状态和依赖图

Memory Tagger（I-12 Memory 标签 + λ 过滤）：
  5. POST /memory/tag            — 输入内容 → 返回层级标签
  6. POST /memory/query          — 按标签/层级查询记忆
  7. POST /memory/filter         — 按 λ 值过滤记忆

Checkpoint Review（I-11 Checkpoint 快照审查）：
  8. POST /checkpoint/create             — 从 Plan 状态提取结构化 Checkpoint 快照
  9. POST /checkpoint/review/{id}        — 对 Checkpoint 执行审查
 10. GET  /checkpoint/chain/{plan_id}    — 获取 Plan 的完整 Checkpoint 链

健康检查：
 11. GET  /health               — 返回服务状态和存储统计

数据持久化：单 SQLite (~/.hermes/mcp/harness.db)
启动端口：8200
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

import yaml
from fastapi import FastAPI, HTTPException

# 使用相对导入（harness_server 包内部）
from .models import (
    CascadeRequest,
    CascadeResponse,
    ChainResponse,
    CreateCheckpointRequest,
    CreateCheckpointResponse,
    CreatePlanRequest,
    CreatePlanResponse,
    ErrorResponse,
    FilterRequest,
    FilterResponse,
    PlanStatusResponse,
    QueryRequest,
    QueryResponse,
    ReviewRequest,
    ReviewResponse,
    TagRequest,
    TagResponse,
    UpdateStepRequest,
)
from .config import default_memories_dir
from .storage import HarnessStorage, default_db_path

# ── 领域兼容重导出 ──────────────────────────────────────────────
# 领域逻辑已迁入 harness_server.domain；以下名字仍从 harness_server.app
# 导入（tests/、deepseek_harness.cli、harness_server.server 依赖这些旧
# 导入路径），由 domain 提供唯一实现，本模块不再复制任何逻辑。
from .domain import (
    DEFAULT_MEMORY_LIMIT,
    MAX_MEMORY_LIMIT,
    DomainConflict,
    DomainNotFound,
    HarnessDomain,
    HarnessDomainError,
    LAYER_RULES,
    LayerRule,
    _adjustments_from_snapshot,
    _build_completed_summary,
    _build_current_goal,
    _build_remaining_plan,
    _check_alignment_from_snapshot,
    _check_progress_from_snapshot,
    _compute_confidence,
    _compute_unexpected_impact,
    _decompose_task,
    _find_matches,
    _import_md_file,
    _steps_to_model,
    build_adjacency_list,
    cascade_correct,
    classify,
    detect_cycle,
    detect_task_type,
    extract_tags,
    import_hermes_memories,
)

# ════════════════════════════════════════════════════════════════
# 日志配置
# ════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("harness-server")

# ════════════════════════════════════════════════════════════════
# 配置加载（从 config.yaml 读取 Memory Tagger 相关配置）
# ════════════════════════════════════════════════════════════════

# config.yaml 与 server.py 在同一目录下
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    """加载 config.yaml 配置文件

    如果文件不存在或解析失败，返回空字典（使用代码中的默认值）。

    Returns:
        配置字典
    """
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("config.yaml 未找到，使用默认配置")
        return {}
    except yaml.YAMLError as e:
        logger.warning("config.yaml 解析失败: %s，使用默认配置", e)
        return {}


config = load_config()

# 从配置中提取 Memory Tagger 相关设置
storage_config = config.get("storage", {})
TASK_LAMBDA_MAP: Dict[str, dict] = {}
task_mappings = config.get("task_type_mappings", {})
for task_type, mapping in task_mappings.items():
    if isinstance(mapping, dict):
        TASK_LAMBDA_MAP[task_type] = mapping

DEFAULT_LAMBDA = config.get("default_lambda", 0.5)
DEFAULT_LAMBDA_LABEL = config.get("default_lambda_label", "mixed")

# ════════════════════════════════════════════════════════════════
# 存储层初始化（单 SQLite，3 组表）
# ════════════════════════════════════════════════════════════════

# 数据库路径：优先用 config.yaml 中的配置，其次用环境变量，最后用默认路径
db_path = os.path.expanduser(
    storage_config.get("db_path") or default_db_path()
)
storage = HarnessStorage(db_path=db_path)


# ════════════════════════════════════════════════════════════════
# 启动时导入 Hermes Memory（如果配置开启）
# ════════════════════════════════════════════════════════════════

if storage_config.get("import_on_startup", True):
    memories_dir = os.path.expanduser(
        storage_config.get("hermes_memories_dir") or default_memories_dir()
    )
    imported, skipped, total = import_hermes_memories(memories_dir, storage)
    logger.info("启动导入: %d 插入, %d 跳过, 共 %d 文件", imported, skipped, total)


# ════════════════════════════════════════════════════════════════
# FastAPI 应用
# ════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Harness MCP Server",
    description=(
        "合并服务：Plan Engine（I-06）+ Memory Tagger（I-12）+ Checkpoint Review（I-11）。"
        "提供 OKR PlanStep DAG 管理级联修正、Memory 标签分类与 λ 过滤、"
        "Checkpoint 快照审查三大功能。"
    ),
    version="1.0.0",
)
# 注意：不加 CORS middleware（本地服务不需要）


def _domain() -> HarnessDomain:
    """按当前模块级 ``storage`` 解析领域服务实例。

    ``server.create_app`` 会在导入后替换模块属性 ``storage``（注入独立
    存储层），因此端点必须每次读取最新的全局 ``storage``，而不是在模块级
    缓存一个 domain 单例。HarnessDomain 的构造只是持有引用，代价可忽略。
    """
    return HarnessDomain(storage=storage)


# ════════════════════════════════════════════════════════════════
# Plan Engine 端点（/plan 前缀）
# ════════════════════════════════════════════════════════════════


@app.post(
    "/plan/create",
    response_model=CreatePlanResponse,
    responses={422: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    summary="从任务描述创建 OKR PlanStep 列表",
    description="输入任务描述 → 自动分解为 3-8 个步骤 → 返回 OKRPlanStep 列表",
)
async def create_plan(req: CreatePlanRequest):
    """POST /plan/create — 委托 HarnessDomain.create_plan

    任务描述为空 → 422；检测到循环依赖 → 409；其余异常 → 500。
    """
    try:
        result = _domain().create_plan(req.task_description)
    except DomainConflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("create_plan 错误: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return CreatePlanResponse(**result)


@app.put(
    "/plan/step/{step_id}",
    response_model=dict,
    responses={404: {"model": ErrorResponse}},
    summary="更新步骤状态/内容",
    description="更新步骤后自动触发级联修正",
)
async def update_step(step_id: str, req: UpdateStepRequest):
    """PUT /plan/step/{step_id} — 委托 HarnessDomain.update_plan_step

    步骤或其所属 Plan 不存在 → 404；其余异常 → 500（与原实现一致：
    非法状态转换 / DAG 校验失败等 storage ValueError 仍走 500）。
    状态或文本变更时自动触发级联修正。
    """
    try:
        return _domain().update_plan_step(
            step_id,
            text=req.text,
            key=req.key,
            status=req.status,
            parent_id=req.parent_id,
            dependency_ids=req.dependency_ids,
            association_strength=req.association_strength,
        )
    except DomainNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("update_step 错误: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/plan/cascade",
    response_model=CascadeResponse,
    responses={404: {"model": ErrorResponse}},
    summary="级联修正引擎",
    description="从 modified_step_id 出发 BFS 遍历，按关联强度分级处理影响",
)
async def cascade(req: CascadeRequest):
    """POST /plan/cascade — 委托 HarnessDomain.plan_cascade

    Plan 不存在/无步骤或起点步骤不属于该 Plan → 404；其余异常 → 500。
    """
    try:
        result = _domain().plan_cascade(req.plan_id, req.modified_step_id)
    except DomainNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("cascade 错误: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return CascadeResponse(**result)


@app.get(
    "/plan/status/{plan_id}",
    response_model=PlanStatusResponse,
    responses={404: {"model": ErrorResponse}},
    summary="获取 Plan 状态和依赖图",
    description="返回完整 Plan + 依赖图邻接表",
)
async def get_plan_status(
    plan_id: str, include_archived: bool = False,
):
    """GET /plan/status/{plan_id} — 委托 HarnessDomain.plan_status

    默认不返回已归档 Plan（TC-PLAN-012），include_archived=true 可查。
    Plan 不存在 → 404；其余异常 → 500。
    """
    try:
        result = _domain().plan_status(plan_id, include_archived=include_archived)
    except DomainNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("get_plan_status 错误: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return PlanStatusResponse(**result)


# ════════════════════════════════════════════════════════════════
# Memory Tagger 端点（/memory 前缀）
# ════════════════════════════════════════════════════════════════


@app.post(
    "/memory/tag",
    response_model=TagResponse,
    responses={422: {"model": ErrorResponse}},
    summary="输入内容 → 返回层级标签",
    description="对单条记忆内容进行层级分类，输出标签、层级和置信度",
)
async def tag_memory(req: TagRequest):
    """POST /memory/tag — 委托 HarnessDomain.memory_tag

    使用关键词规则引擎（LAYER_RULES）对内容分类，返回最匹配的层级、
    命中的关键词列表和置信度。空内容由请求模型（min_length=1）拦为 422。
    """
    try:
        result = _domain().memory_tag(req.content)
    except Exception as e:
        logger.error("tag_memory 错误: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return TagResponse(**result)


@app.post(
    "/memory/query",
    response_model=QueryResponse,
    responses={422: {"model": ErrorResponse}},
    summary="按标签/层级查询记忆",
    description="从持久化存储中按标签和/或层级检索记忆条目",
)
async def query_memory(req: QueryRequest):
    """POST /memory/query — 委托 HarnessDomain.memory_query

    tags 和 layer 都是可选的筛选条件，同时使用时取交集。
    """
    try:
        result = _domain().memory_query(
            tags=req.tags,
            layer=req.layer,
            limit=req.limit,
        )
    except Exception as e:
        logger.error("query_memory 错误: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return QueryResponse(**result)


@app.post(
    "/memory/filter",
    response_model=FilterResponse,
    responses={422: {"model": ErrorResponse}},
    summary="按 λ 值过滤记忆",
    description=(
        "λ ∈ [0, 1]：0.0=仅约束层, 0.5=约束+偏好+决策, 1.0=全部。"
        "从持久化存储按 λ 值过滤记忆。"
    ),
)
async def filter_memory(req: FilterRequest):
    """POST /memory/filter — 委托 HarnessDomain.memory_filter

    λ 值决定注入哪些层级的记忆；λ 越界由请求模型（ge/le）拦为 422。
    """
    try:
        result = _domain().memory_filter(req.lambda_value, limit=req.limit)
    except Exception as e:
        logger.error("filter_memory 错误: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return FilterResponse(**result)


# ════════════════════════════════════════════════════════════════
# Checkpoint Review 端点（/checkpoint 前缀）
# ════════════════════════════════════════════════════════════════


@app.post(
    "/checkpoint/create",
    response_model=CreateCheckpointResponse,
    responses={422: {"model": ErrorResponse}},
    summary="从 Plan 状态提取结构化 Checkpoint 快照",
    description=(
        "输入 Plan 步骤列表和已完成步骤 ID，生成结构化 Checkpoint 快照。"
        "自动递增 checkpoint_number。"
    ),
)
async def create_checkpoint(req: CreateCheckpointRequest):
    """POST /checkpoint/create — 委托 HarnessDomain.checkpoint_create

    plan_id 为空 → 422；Plan 不存在 → 404（DomainNotFound，与领域层
    错误契约一致）；其余异常 → 500。
    """
    try:
        result = _domain().checkpoint_create(
            plan_id=req.plan_id,
            plan_steps=req.plan_steps,
            completed_step_ids=req.completed_step_ids,
            unexpected_findings=req.unexpected_findings,
        )
    except DomainNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("create_checkpoint 错误: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return CreateCheckpointResponse(**result)


@app.post(
    "/checkpoint/review/{checkpoint_id}",
    response_model=ReviewResponse,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    summary="对 Checkpoint 执行轻量上下文审查",
    description=(
        "从存储读取 Checkpoint 快照，对其执行四维审查："
        "目标对齐、进度偏差、意外发现影响、剩余计划调整建议。"
        "仅读快照，不读完整执行日志（上下文递减原则）。"
    ),
)
async def review_checkpoint(checkpoint_id: str, req: ReviewRequest):  # noqa: ARG001
    """POST /checkpoint/review/{checkpoint_id} — 委托 HarnessDomain.checkpoint_review

    审查基于快照做确定性判断（不调用 LLM）。Checkpoint 不存在 → 404；
    其余异常 → 500。
    """
    try:
        result = _domain().checkpoint_review(checkpoint_id)
    except DomainNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("review_checkpoint 错误: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return ReviewResponse(**result)


@app.get(
    "/checkpoint/chain/{plan_id}",
    response_model=ChainResponse,
    responses={422: {"model": ErrorResponse}},
    summary="获取 Plan 的完整 Checkpoint 链",
    description="返回按 checkpoint_number 升序排列的 Checkpoint 列表",
)
async def get_checkpoint_chain(plan_id: str):
    """GET /checkpoint/chain/{plan_id} — 委托 HarnessDomain.checkpoint_chain

    返回指定 Plan 的所有 Checkpoint，按编号升序排列。
    """
    try:
        result = _domain().checkpoint_chain(plan_id)
    except Exception as e:
        logger.error("get_checkpoint_chain 错误: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return ChainResponse(**result)


# ════════════════════════════════════════════════════════════════
# 健康检查端点
# ════════════════════════════════════════════════════════════════


@app.get(
    "/health",
    summary="健康检查",
    description="返回服务状态和三组表的存储统计",
)
async def health_check():
    """GET /health — 委托 HarnessDomain.health

    返回三组表的统计信息：plans/steps/memories/checkpoints 的数量。
    """
    return _domain().health()


# ════════════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════════════


def main():
    """启动 FastAPI 服务

    端口：8200（通过环境变量 HARNESS_PORT 可覆盖）
    主机：127.0.0.1（通过环境变量 HARNESS_HOST 可覆盖）
    """
    import uvicorn

    host = os.environ.get("HARNESS_HOST", "127.0.0.1")
    port = int(os.environ.get("HARNESS_PORT", "8200"))

    logger.info("启动 Harness MCP Server — %s:%d", host, port)
    logger.info("数据库路径: %s", storage.db_path)

    uvicorn.run(
        "harness_server.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


__all__ = [
    # 应用与存储（harness_server.server 按名读取 app/storage）
    "app",
    "storage",
    "config",
    "load_config",
    "main",
    # 领域兼容重导出（tests/、deepseek_harness.cli 按旧路径导入）
    "DEFAULT_LAMBDA",
    "DEFAULT_LAMBDA_LABEL",
    "DEFAULT_MEMORY_LIMIT",
    "DomainConflict",
    "DomainNotFound",
    "HarnessDomain",
    "HarnessDomainError",
    "LAYER_RULES",
    "LayerRule",
    "MAX_MEMORY_LIMIT",
    "TASK_LAMBDA_MAP",
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


if __name__ == "__main__":
    main()
