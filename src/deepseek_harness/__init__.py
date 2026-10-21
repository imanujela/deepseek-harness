"""数字分身 Plugin — 使用 Hermes Plugin Hook 系统，不改一行核心代码。"""

import logging

from .gate import on_pre_llm_call as on_cognitive_gate
from .assessor import on_post_tool_call
from .learner import on_session_end
from .subagent_watch import on_subagent_start, on_subagent_stop
from .intent_router import on_pre_llm_call as on_intent_router
from .reasoning_effort import on_pre_llm_call as on_reasoning_effort
from .latest_reminder import on_pre_llm_call as on_latest_reminder
from .skill_router import on_pre_llm_call as on_skill_router
from .skill_router import on_pre_tool_call as on_skill_injection
from .discipline import RevisionGate, StopRule
from .tools import register_all_tools

logger = logging.getLogger(__name__)

# Shared state for the tool layer and the pre_llm_call hook below.
revision_gate = RevisionGate()
stop_rule = StopRule()

# arXiv 2607.01641: the feedback path is not effectively bounded. Once a
# deliverable exists, further turns must justify themselves.
STOP_DIRECTIVE = (
    "[discipline] A deliverable already exists. Do not keep verifying it. "
    "Stop now, or name one falsifiable open question (ending in '?', or a "
    "'because ... verify/check/test' clause) that still blocks delivery."
)


def on_stop_rule(**kwargs):
    """pre_llm_call: inject the stop directive after a deliverable exists.

    Never raises, never sleeps; non-first turns only.
    """
    try:
        if kwargs.get("is_first_turn", False):
            return None
        if not stop_rule.status()["deliverable"]:
            return None
        return {"context": STOP_DIRECTIVE}
    except Exception:  # pragma: no cover - defensive: hooks must never break a turn
        logger.warning("[discipline] stop-rule hook failed", exc_info=True)
        return None


def register(ctx):
    """注册所有 Plugin Hook handler + 工具。"""
    # Hook 注册
    ctx.register_hook("pre_llm_call", on_cognitive_gate)
    ctx.register_hook("pre_llm_call", on_intent_router)
    ctx.register_hook("pre_llm_call", on_reasoning_effort)
    ctx.register_hook("pre_llm_call", on_latest_reminder)
    ctx.register_hook("pre_llm_call", on_skill_router)
    ctx.register_hook("pre_llm_call", on_stop_rule)
    # Subagent skill contracts travel through pre_tool_call: the host ignores
    # subagent_start's return value, but shallow-merges a "modify" result here.
    ctx.register_hook("pre_tool_call", on_skill_injection)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("on_session_end", on_session_end)
    ctx.register_hook("subagent_start", on_subagent_start)
    ctx.register_hook("subagent_stop", on_subagent_stop)

    # 工具注册（I-06 规划引擎 + I-12 记忆标签 + I-11 快照审查）
    register_all_tools(ctx)
