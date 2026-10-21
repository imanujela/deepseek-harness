"""
I-17 推理强度控制（Reasoning Effort Control）。

在 pre_llm_call hook 中根据 I-10 意图路由的分类结果，
通过文本提示注入推理深度指引，控制 DeepSeek V4 模型的推理深度。

设计限制：Hermes pre_llm_call hook 协议仅支持通过返回 {"context": "..."}
注入文本到 user message，无法直接设置 API 端的 reasoning_effort 参数。
因此本插件改为通过上下文文本指引模型调整推理强度，而非直接操作 API 参数。

Spike 验证结论：
- DeepSeek API 接受 reasoning_effort 参数（"max"/"high"/"medium"）
- 设 "max" 时推理 token 约 3x 于不设时（282 vs 91）
- 因 Hermes hook 架构限制，改为文本提示注入

映射规则（意图类型 → 推理强度指引，I-18 起由 thinking_budget 提供）：
  - architecture / research → ultra（wire 仍是 "max"，超出部分由切除协议承载）
  - collaboration → xhigh（wire 同为 "max"）
  - refactor / new / medium → high（短协议）
  - simple / spec_driven → 不注入

本文件保留 _INTENT_EFFORT_MAP / _get_reasoning_effort_level 作为对外契约，
hook 内部改为调用 I-18 ThinkingBudget（见 thinking_budget.py）。

关联: docs/innovations/17-reasoning-effort-control.md
      docs/innovations/18-thinking-budget-yield-floor.md
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# 意图类型到 reasoning_effort 的映射
_INTENT_EFFORT_MAP: dict[str, str] = {
    # 高复杂度任务 → max reasoning
    "architecture": "max",
    "research": "max",
    "collaboration": "max",
    # 中等复杂度任务 → high reasoning
    "refactor": "high",
    "new": "high",
    "medium": "high",
    # 简单任务 → 不设置（使用模型默认值）
    "simple": "",
    "spec_driven": "",
}

def _get_reasoning_effort_level(intent: str) -> Optional[str]:
    """根据 I-10 意图类型获取对应的 reasoning_effort 级别。

    Args:
        intent: I-10 意图路由的分类结果（如 "architecture", "refactor" 等）

    Returns:
        "max", "high", 或 None（表示不设置，使用模型默认值）
    """
    effort = _INTENT_EFFORT_MAP.get(intent, "")
    return effort if effort else None


def on_pre_llm_call(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """pre_llm_call hook：注入推理强度上下文。

    依赖 I-10 意图路由的分类结果（classify_intent），
    再经 I-18 ThinkingBudget 得出 tier 与协议文本。

    仅在首轮调用且存在 user_message 时执行分类和注入。
    architecture/research → ultra（wire 仍为 "max"，超出部分由协议承载），
    collaboration → xhigh（wire 同为 "max"），
    refactor/new/medium → high 短协议，
    simple/spec_driven → 不注入。

    Args:
        **kwargs: Hermes Plugin 系统传入的上下文。
            关键字段：
            - is_first_turn: bool — 是否为首轮调用
            - user_message: str — 用户消息原文
            - model: str — 当前使用的模型名称

    Returns:
        dict with 'context' key 包含推理强度指引与协议，
        或 None（非首轮/简单任务/无法处理时跳过）。
        异常时静默捕获并记录日志，不阻断流程。
    """
    try:
        is_first = kwargs.get("is_first_turn", False)
        user_message = kwargs.get("user_message", "")

        if not is_first or not user_message:
            return None

        from .intent_router import classify_intent  # 延迟导入
        from .thinking_budget import plan, wire_effort  # I-18

        result = classify_intent(user_message)
        intent = result.get("intent", "spec_driven")

        effort_plan = plan(intent)

        if effort_plan.tier in ("ultra", "xhigh"):
            header = (
                f"[I-17 ThinkingBudget] tier={effort_plan.tier} "
                f"wire={wire_effort(effort_plan.tier)} "
                f"floor={effort_plan.yield_floor} 高复杂度"
            )
            return {"context": f"{header}\n{effort_plan.protocol}"}

        if effort_plan.tier == "high":
            return {
                "context": (
                    "[I-17 推理强度] 中等复杂度（重构/新功能/功能修改）\n"
                    f"{effort_plan.protocol}"
                )
            }

        return None

    except ImportError:
        logger.debug("[I-17] intent_router not available, skipping")
    except Exception as e:
        logger.error("[I-17] Error: %s", e, exc_info=True)

    return None
