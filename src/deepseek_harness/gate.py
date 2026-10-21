"""
认知门控引擎（Cognitive Gate Engine）。

在 pre_llm_call hook 中被 Hermes Plugin 系统调用，
在每轮 LLM 调用前注入认知检查提醒、双向原语和范围控制提醒。

工作模式：
  - 首轮：注入完整 MAP.md + L1/L2 + I-02 双向原语 + I-08 范围控制
  - 后续轮：仅注入 L1/L2 + 简短 I-02 + 简短 I-08（避免上下文膨胀）
"""

import logging
import os
import re
from typing import Any, Dict, Optional

from .paths import map_path

logger = logging.getLogger(__name__)


_I02_FULL = """[I-02 双向原语]
你可以在需要时使用以下原语主动声明需求：

/propose_skill <name> <trigger> <steps>
  - 用途：完成任务后提议固化为可复用 Skill
  - 示例：/propose_skill python-migrate-setup-to-pyproject "检测到 setup.py" "1.读取setup.py 2.生成pyproject.toml 3.验证"
  - 注意：只建议有价值的可复用模式

/trigger_self_review <target> <reason>
  - 用途：在关键决策前请求独立审查
  - 示例：/trigger_self_review "数据库迁移计划" "涉及生产数据，影响范围大"
  - 注意：用于高风险决策"""

_I02_BRIEF = (
    "[I-02] 如需提议 Skill 或请求审查，请使用 "
    "/propose_skill 或 /trigger_self_review"
)

_I08_FULL = """[I-08 范围控制]
需求对齐层：不得超出用户显式声明的功能范围。如果任务描述中有"不要 X""只做 Y"等排除项，严格遵守。
技术规划层：新增步骤前先判断是"必要依赖"还是"范围蔓延"。
  判断标准：不加这个步骤，已承诺的 PlanStep 能否完成？
  → 能完成 = 范围蔓延，拒绝
  → 不能完成 = 必要依赖，可接受"""

_I08_BRIEF = (
    "[I-08] 注意范围控制：不加步骤能完成 = 拒绝"
)

# ── I-01: 硬约束横切状态通道 ──
# SES-001: 约束按 session_id 隔离存储，不再使用模块级全局集合
from .session_policy import SessionPolicyStore

_HARD_CONSTRAINT_PATTERN = re.compile(
    r'(?:不能|不要|不得|禁止|严禁|不允许|千万别|绝对不|必须)[^，。；、！？\n]{2,60}'
)

# 兼容层：保留 _current_hard_constraints 名称供旧测试读取，
# 但实际存储已迁移到 SessionPolicyStore
_current_hard_constraints: set[str] = set()


# ── 约束钉节奏器（ConstraintRot）──
# 硬约束被提取进 SessionPolicyStore 后，若后续轮次不再出现，模型会在几轮内违反它
# （实测：违规率 0% → 30-59%）。每 _PIN_EVERY 轮把最多 _PIN_MAX 条原样钉回，
# 约 47 tokens，且注入在 user message 侧，不动缓存前缀。
_PIN_EVERY = 6
_PIN_MAX = 7
_PIN_MAX_CHARS = 60
_TURN_COUNTS: dict[str, int] = {}


def _pinned_constraints(session_id: str, *, turn_id: str = "") -> str:
    """每 ``_PIN_EVERY`` 轮返回一段约束钉文本，其它轮返回空串。

    参数:
        session_id: 当前会话 ID（为空时回落到模块级兼容集合）
        turn_id: 当前轮次 ID（仅用于日志，不参与节奏判断）

    返回:
        ``"[约束钉] c1；c2…"`` 或空串。任何存储异常都降级为空串，不阻断流程。
    """
    try:
        key = session_id or "_global"
        count = _TURN_COUNTS.get(key, 0) + 1
        _TURN_COUNTS[key] = count
        if count % _PIN_EVERY != 0:
            return ""
        if session_id:
            active = list(SessionPolicyStore.get_instance().get_active_constraints(session_id))
        else:
            active = list(_current_hard_constraints)
        pinned = [str(c).strip()[:_PIN_MAX_CHARS] for c in active if str(c).strip()][:_PIN_MAX]
        if not pinned:
            return ""
        return "[约束钉] " + "；".join(pinned)
    except Exception:
        logger.debug("[gate] constraint pin skipped (turn_id=%s)", turn_id, exc_info=True)
        return ""


def _core_reminders() -> list[str]:
    return [
        "[L1] 荣辱观：以知道自己的不足为荣、以提升认知为荣、以告诉实情为荣。"
        "回应前请用工具验证每个论断，不确定就说不确定。",
        "[L2] 思维方式：第一性原理、Step by Step、假设先行、找盲区、科研严谨。"
        "拆解到最小任务，假设先行验证。",
    ]


def _map_navigation() -> str:
    """读取 MAP.md 导航内容。

    路径在调用时解析：Windows 的 Hermes home 是 ``%LOCALAPPDATA%\\hermes``，
    写死 ``~/.hermes`` 会读到不存在的文件，于是 MAP 事实被静默跳过
    （"跳过了本该看到的记录"）。文件缺失或读取失败都不阻断流程。
    """
    try:
        navigation_path = map_path()
        if navigation_path.exists():
            content = navigation_path.read_text(encoding="utf-8").strip()
            if content:
                return f"[MAP]\n{content}"
    except Exception:
        # 任何文件读取失败都不阻断流程，静默降级
        pass
    return ""


def on_pre_llm_call(**kwargs) -> Optional[Dict[str, Any]]:
    """注入认知提醒、双向原语和范围控制到 user message。

    首轮注入完整 MAP.md + L1/L2 + I-02 完整 + I-08 完整。
    后续轮仅注入 L1/L2 + 简短 I-02 + 简短 I-08（避免上下文膨胀）。

    Args:
        **kwargs: 包含 is_first_turn, session_id 等上下文。
            kwargs 上下文（来自 turn_context.py:468-479）:
                session_id: str — 当前会话 ID
                task_id: str — 当前任务 ID
                turn_id: str — 当前轮次 ID
                user_message: str — 用户消息原文
                conversation_history: list — 对话历史列表
                is_first_turn: bool — 是否为首轮调用
                model: str — 当前使用的模型名称
                platform: str — 对话平台标识
                sender_id: str — 发送者 ID

    Returns:
        dict with 'context' key containing the injection text。
        至少包含 L1/L2 + 简短 I-02/I-08，首轮额外包含 MAP.md 导航内容。
        所有文件读取异常都会被静默捕获，不会阻断流程。
        不会返回 None — 至少返回纯 L1/L2 的 dict。

    Raises:
        不显式抛异常。所有 IO 操作在 try/except 内。
    """
    is_first = kwargs.get("is_first_turn", False)

    # ── I-01: 从用户消息中提取硬约束，存入 SessionPolicyStore ──
    session_id = kwargs.get("session_id", "")
    user_message = kwargs.get("user_message", "")
    if isinstance(user_message, str) and user_message.strip():
        matches = _HARD_CONSTRAINT_PATTERN.findall(user_message)
        if matches:
            extracted = {c.strip() for c in matches if c.strip()}
            if extracted and session_id:
                store = SessionPolicyStore.get_instance()
                store.add_constraints(session_id, extracted, turn_id=kwargs.get("turn_id", ""))
                # 同步兼容层（供旧测试读取当前 session 约束）
                _current_hard_constraints.clear()
                _current_hard_constraints.update(
                    store.get_active_constraints(session_id)
                )
            elif extracted:
                _current_hard_constraints.clear()
                _current_hard_constraints.update(extracted)

    parts: list[str] = _core_reminders()

    if is_first:
        # 首轮：完整注入
        map_content = _map_navigation()
        if map_content:
            parts.append(map_content)
        parts.append(_I02_FULL)
        parts.append(_I08_FULL)
    else:
        # 后续轮：简短提醒，避免上下文膨胀
        parts.append(_I02_BRIEF)
        parts.append(_I08_BRIEF)
        # 约束钉（ConstraintRot 修复）：硬约束被提取后若不再出现，模型会在几轮后
        # 违反它（实测违规率 0% → 30-59%）。每 _PIN_EVERY 轮把最多 7 条原样钉回，
        # 约 47 tokens（<0.5% 上下文），不改动缓存前缀。
        pin = _pinned_constraints(session_id, turn_id=kwargs.get("turn_id", ""))
        if pin:
            parts.append(pin)

    return {"context": "\n".join(parts)}
