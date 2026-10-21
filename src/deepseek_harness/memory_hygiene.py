"""记忆写入端的卫生闸门（纯函数，无 I/O、无网络、无时钟）。

为什么需要这个模块
------------------
HF 讨论 ``deepseek-ai/DeepSeek-V4.1-Flash#46`` 实测：同一段会话，V4 Flash
写了 13 KB / 9 个记忆文件，V4.1 Flash 写了 161 KB / 11 个文件——12 倍。
多出来的字节不是事实：

* "Decisions log" 把每一步复述成段落；
* "open questions" 与写给自己的建议；
* 109 个 "TBC" 占位符，对应的是用户从未提供过的事实；
* "why this file exists" / "how to keep it" 这类文件前言。

同一讨论里，系统提示词层面的规则对这份膨胀**零效果**，所以约束必须放在
写入端：每条候选记忆先过 :func:`classify`，只有 ``kind == "fact"`` 且未触碰
预算的才由 :func:`admit` 放行。调用方（``HarnessDomain.memory_tag`` 的集成点、
Hermes 记忆钩子）负责传 ``used_chars``；本模块自身不读文件、不看时间。

设计取舍
--------
分类是**保守**的：拒绝一条本来可留的叙述，代价远小于继续写 12 倍体积。
但被拒绝的 chronicle 不必直接丢弃——:func:`compress_chronicle` 会保留其中
带数值 / 日期 / 名字 / 状态变化的句子，拼成
``值A; 值B; 值C`` 的一行"当前值表"，这正是同一基准里 V4 Flash 自然写出的
形态（"table of current values, edit history one line in the cell"）。
"""

from __future__ import annotations

import re

__all__ = ["classify", "admit", "compress_chronicle", "KINDS"]

#: 允许出现在 classify() 结果里的种类。
KINDS = ("fact", "chronicle", "placeholder", "self_advice", "preamble")

#: FACT 的"典型"上限；超过它的文本必须携带具体值，否则按叙述处理。
FACT_TYPICAL_MAX_CHARS = 120

# ── 规则词表 ──────────────────────────────────────────────────

_PLACEHOLDER_MARKERS = (
    "tbc",
    "tbd",
    "to be confirmed",
    "to be decided",
    "unknown yet",
    "to be determined",
)

_SELF_ADVICE_PATTERNS = (
    r"\byou should\b",
    r"\byou must\b",
    r"\byou need to\b",
    r"\bremember to\b",
    r"\bconsider whether\b",
    r"\bbe careful\b",
    r"\bmake sure to\b",
    r"\bdon'?t forget\b",
)

_PREAMBLE_PREFIXES = (
    "why this file exists",
    "why this file",
    "how to keep",
    "how to use this file",
    "purpose of this file",
)

#: 叙述过程的关键词；至少命中 2 个**不同**词才算 chronicle。
_CHRONICLE_KEYWORDS = ("decision", "considered", "decided", "verified", "re-ran")
_CHRONICLE_KEYWORD_RES = {
    "decision": re.compile(r"\bdecisions?\b", re.IGNORECASE),
    "considered": re.compile(r"\bconsider(?:ed|ing)\b", re.IGNORECASE),
    "decided": re.compile(r"\bdecid(?:ed|ing)\b", re.IGNORECASE),
    "verified": re.compile(r"\bverif(?:ied|ying)\b", re.IGNORECASE),
    "re-ran": re.compile(r"\bre-?ran\b", re.IGNORECASE),
}

#: 明确的"我在讲我做过什么"的标记。
_PROCESS_MARKERS = (
    re.compile(r"\bi then\b", re.IGNORECASE),
    re.compile(r"\bthen i\b", re.IGNORECASE),
    re.compile(r"\bafter that we\b", re.IGNORECASE),
    re.compile(r"\bnext we\b", re.IGNORECASE),
    re.compile(r"\bwe then\b", re.IGNORECASE),
)

_PAST_VERB = re.compile(
    r"\b(?:ran|added|wrote|created|checked|updated|fixed|decided|verified|changed|"
    r"moved|tested|reviewed|refactored|implemented|tried|found|saw|opened|read|"
    r"removed|replaced|committed|merged|installed|built|deleted|considered)\b",
    re.IGNORECASE,
)

_STEP_MARKER = re.compile(r"\bstep\s+\d+\s*[:.)]", re.IGNORECASE)

# ── "携带具体值"的判定 ────────────────────────────────────────

_VALUE_PATTERNS = (
    re.compile(r"\d"),                                  # 数值
    re.compile(r"https?://"),                           # URL
    re.compile(r"[`\"'][^`\"\s][^`\"']*[`\"']"),        # 引号/反引号里的名字
    re.compile(r"\b[\w-]+\.(?:py|ts|tsx|js|jsx|json|md|ya?ml|toml|db|sqlite|sh|txt)\b"),
    re.compile(r"\b\w+\s*=\s*\S+"),                     # key=value
    re.compile(r"\b[A-Z]{2,}\b"),                        # 缩写：API / TBC / SQL
    re.compile(r"\b[a-z]+[A-Z]\w*"),                     # CamelCase：HarnessStorage / SQLite
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),               # 日期
)

#: compress_chronicle() 里视为"状态变化"的词。
_STATE_CHANGE = re.compile(
    r"\b(?:completed|changed to|moved to|set to|now|migrated|upgraded|bumped|"
    r"passing|failed|status|switched to|replaced by|deprecated)\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


# ── 内部工具 ──────────────────────────────────────────────────


def _normalize(text: str) -> str:
    """压缩空白并剥离首尾空格；判定一律在归一化文本上做。"""
    return " ".join(text.split())


def _has_value(text: str) -> bool:
    """文本是否携带具体值（数字 / 日期 / 名字 / 路径 / 键值）。"""
    return any(pattern.search(text) for pattern in _VALUE_PATTERNS)


def _placeholder_marker(text: str) -> str | None:
    """返回命中的占位符标记；``?`` 单独处理。"""
    lowered = text.lower()
    for marker in _PLACEHOLDER_MARKERS:
        # "tbc"/"tbd" 需要词边界，避免误伤普通单词内部。
        if marker in {"tbc", "tbd"}:
            if re.search(rf"\b{marker}\b", lowered):
                return marker
        elif marker in lowered:
            return marker
    if "?" in text:
        return "?"
    return None


def _is_preamble(text: str) -> bool:
    lowered = text.lower().lstrip("*-_# ").strip()
    if lowered.startswith(_PREAMBLE_PREFIXES):
        return True
    return "this file" in lowered and "purpose" in lowered


def _chronicle_reason(text: str) -> str | None:
    """叙述判定的理由；不是 chronicle 返回 None。"""
    hits = [name for name, pattern in _CHRONICLE_KEYWORD_RES.items() if pattern.search(text)]
    if len(hits) >= 2:
        return "chronicle: narrates process via keywords " + ", ".join(hits)

    for marker in _PROCESS_MARKERS:
        if marker.search(text):
            return f"chronicle: process narration marker {marker.pattern!r}"

    if _STEP_MARKER.search(text) and _PAST_VERB.search(text):
        return "chronicle: numbered steps retold in past tense"

    return None


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(text.strip()) if part.strip()]


# ── 公共 API ─────────────────────────────────────────────────


def classify(text: str) -> dict:
    """判定一条候选记忆的种类。

    Args:
        text: 候选记忆原文（可以是一段）。

    Returns:
        ``{"keep": bool, "kind": str, "reasons": list[str]}``，kind 取自
        :data:`KINDS`。``keep`` 只有在 ``kind == "fact"`` 时为 True。

    判定顺序（先命中先返回）：placeholder → self_advice → preamble →
    chronicle → fact。占位符优先，因为一条 ``TBC`` 无论讲的是过程还是建议，
    它承载的事实都是零。
    """
    reasons: list[str] = []
    normalized = _normalize(text)

    if not normalized:
        return {
            "keep": False,
            "kind": "chronicle",
            "reasons": ["empty text: no state to record"],
        }

    marker = _placeholder_marker(normalized)
    if marker is not None:
        reasons.append(f"placeholder marker {marker!r}: value was never supplied")
        return {"keep": False, "kind": "placeholder", "reasons": reasons}

    for pattern in _SELF_ADVICE_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            reasons.append(f"self_advice: addresses the model ({pattern})")
            return {"keep": False, "kind": "self_advice", "reasons": reasons}

    if _is_preamble(normalized):
        reasons.append("preamble: explains the file itself, not the project state")
        return {"keep": False, "kind": "preamble", "reasons": reasons}

    chronicle_reason = _chronicle_reason(normalized)
    if chronicle_reason is not None:
        reasons.append(chronicle_reason)
        return {"keep": False, "kind": "chronicle", "reasons": reasons}

    if len(normalized) > FACT_TYPICAL_MAX_CHARS and not _has_value(normalized):
        reasons.append(
            "chronicle: long narrative "
            f"({len(normalized)} chars > {FACT_TYPICAL_MAX_CHARS}) without a value"
        )
        return {"keep": False, "kind": "chronicle", "reasons": reasons}

    reasons.append("fact: current state with a concrete value")
    if not _has_value(normalized):
        reasons.append("fact: terse statement without value (kept)")
    return {"keep": True, "kind": "fact", "reasons": reasons}


def admit(
    text: str,
    *,
    budget_chars: int = 4000,
    used_chars: int = 0,
) -> dict:
    """写入闸门：只有 fact 且不撑爆预算的条目才放行。

    Args:
        text: 候选记忆原文。
        budget_chars: 单次会话的记忆写入预算（字符）。
        used_chars: 本次会话已写入的字符数（由调用方维护）。

    Returns:
        ``{"admit", "kind", "reason", "would_overflow"}``。``reason`` 为空串
        表示放行；否则是一句可审计的拒绝理由。
    """
    verdict = classify(text)
    kind = verdict["kind"]
    would_overflow = used_chars + len(text) > budget_chars

    if kind != "fact":
        return {
            "admit": False,
            "kind": kind,
            "reason": f"refused: kind={kind}; " + "; ".join(verdict["reasons"]),
            "would_overflow": would_overflow,
        }

    if would_overflow:
        return {
            "admit": False,
            "kind": kind,
            "reason": (
                "refused: budget overflow "
                f"({used_chars} used + {len(text)} new > {budget_chars})"
            ),
            "would_overflow": True,
        }

    return {
        "admit": True,
        "kind": kind,
        "reason": "",
        "would_overflow": False,
    }


def compress_chronicle(text: str) -> str:
    """把 chronicle 压成"当前值表"：只留带值/状态变化的句子。

    每一句只有在包含数字、日期、名字、键值，或状态变化词时才保留；保留下来的
    句子去掉句末句号后用 ``"; "`` 连接，整体以一个句号收尾。没有任何一句合格
    时返回空串（此时调用方应当整条丢弃，而不是写入空壳）。

    Args:
        text: 叙述性文本。

    Returns:
        压缩结果，例如 ``"Build time is 12.5s now; Tests: 308 passed."``。
    """
    kept: list[str] = []
    for sentence in _sentences(text):
        if _has_value(sentence) or _STATE_CHANGE.search(sentence):
            kept.append(sentence.rstrip(".").strip())

    if not kept:
        return ""
    return "; ".join(kept) + "."
