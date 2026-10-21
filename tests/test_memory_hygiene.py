"""MEM-HYG-001: 记忆写入端的 fact/chronicle 闸门测试。

前置研究（HF discussion deepseek-ai/DeepSeek-V4.1-Flash#46）：同一段会话在
V4 Flash 上写了 13 KB / 9 个文件，在 V4.1 Flash 上写了 161 KB / 11 个文件
（12x）。多出来的不是事实：每一步都要复述成段落的 "Decisions log"、
"open questions"、给自己写的建议、109 个用户从未提供的 "TBC" 占位符、
以及 "why this file exists" / "how to keep it" 这类前言。系统提示词规则
对它的效果为零，所以必须在写入端用代码闸门，而不是求模型守规矩。

本测试只覆盖纯函数（无 I/O、无网络、无时钟），样本均取自上述讨论中
被点名的那几类文本。
"""

from __future__ import annotations

from deepseek_harness.memory_hygiene import (
    admit,
    classify,
    compress_chronicle,
)

# ── 真实样本 ────────────────────────────────────────────────────

#: 一条真正值得写入的记忆：带具体数值的当前状态。
FACT_LINE = "pytest suite: 1544 passed, 0 failed in 3.20s on Python 3.12.10."

#: 讨论中点名的 "Decisions log" 段落：每一步都复述成叙述。
CHRONICLE_SAMPLE = (
    "I then opened storage.py to read the schema. After that we considered whether the "
    "migration path was safe. We decided to keep the SQLite backend and verified the "
    "round-trip. Re-ran the suite afterwards."
)

#: 用户从未提供过的事实，被模型写成占位符。
TBC_LINE = "Deployment target: TBC"

#: 模型写给自己的建议，不是关于项目的事实。
SELF_ADVICE_LINE = "Remember to check the lambda value before storing memories."

#: 典型的文件前言，占用预算但不携带任何事实。
PREAMBLE_LINE = "Why this file exists: the harness keeps a decisions log here."

#: 五句叙述，只有第四句带值。
NARRATIVE = (
    "I then opened the storage layer and read through it. "
    "After that we considered whether the migration path was safe. "
    "We decided to keep using the current backend for now. "
    "Verified that the test suite passed 1544 tests in 3.20s. "
    "Re-ran everything afterwards and it looked fine."
)


# ── classify(): 每一类独立可测 ─────────────────────────────────


def test_fact_line_is_kept_and_carries_reasons() -> None:
    result = classify(FACT_LINE)

    assert result["kind"] == "fact"
    assert result["keep"] is True
    assert result["reasons"], "每条判定都要给出可审计的理由"


def test_decisions_log_paragraph_is_chronicle_and_refused() -> None:
    result = classify(CHRONICLE_SAMPLE)

    assert result["kind"] == "chronicle"
    assert result["keep"] is False
    assert any("chronicle" in reason for reason in result["reasons"])


def test_tbc_placeholder_is_not_a_fact() -> None:
    result = classify(TBC_LINE)

    assert result["kind"] == "placeholder"
    assert result["keep"] is False


def test_bare_question_is_placeholder() -> None:
    result = classify("Which database should the harness use?")

    assert result["kind"] == "placeholder"
    assert result["keep"] is False


def test_self_advice_is_not_a_fact() -> None:
    result = classify(SELF_ADVICE_LINE)

    assert result["kind"] == "self_advice"
    assert result["keep"] is False


def test_file_preamble_is_not_a_fact() -> None:
    result = classify(PREAMBLE_LINE)

    assert result["kind"] == "preamble"
    assert result["keep"] is False


def test_preamble_wins_over_chronicle_keywords() -> None:
    # 前言里常带 "decisions log" 之类的词，但它的功能是解释文件自身，
    # 必须先被判成 preamble 再被拒绝。
    assert classify(PREAMBLE_LINE)["kind"] == "preamble"


def test_long_narrative_without_any_value_is_chronicle() -> None:
    text = (
        "The team discussed the overall approach at length and agreed that the current "
        "shape of the module feels about right for now and that further changes can wait "
        "until there is more information available about the real usage patterns in "
        "production."
    )

    result = classify(text)

    assert len(text) > 120
    assert result["kind"] == "chronicle"
    assert result["keep"] is False


def test_short_terse_statement_without_number_is_still_a_fact() -> None:
    result = classify("Docs are written in Chinese per user preference.")

    assert result["kind"] == "fact"
    assert result["keep"] is True


def test_one_decision_keyword_alone_is_not_enough_for_chronicle() -> None:
    # 只有 1 个过程词、且是一个状态事实 → 仍是 fact（阈值 >= 2）。
    result = classify("Memory layer for decisions: decision, lambda 0.5.")

    assert result["kind"] == "fact"


def test_empty_text_is_not_admissible() -> None:
    result = classify("   ")

    assert result["keep"] is False
    assert result["kind"] in {"chronicle", "placeholder", "self_advice", "preamble"}


# ── admit(): 闸门行为 ─────────────────────────────────────────


def test_admit_allows_a_fact() -> None:
    result = admit(FACT_LINE)

    assert result["admit"] is True
    assert result["kind"] == "fact"
    assert result["would_overflow"] is False


def test_admit_refuses_chronicle_with_a_reason() -> None:
    result = admit(CHRONICLE_SAMPLE)

    assert result["admit"] is False
    assert result["kind"] == "chronicle"
    assert result["reason"]


def test_admit_refuses_when_budget_would_overflow() -> None:
    result = admit(FACT_LINE, budget_chars=100, used_chars=90)

    assert result["admit"] is False
    assert result["would_overflow"] is True
    assert result["reason"]


def test_admit_budget_boundary_is_inclusive() -> None:
    # used + len == budget 不算溢出（严格大于才算）。
    exact = len(FACT_LINE) + 10
    result = admit(FACT_LINE, budget_chars=exact, used_chars=10)

    assert result["would_overflow"] is False
    assert result["admit"] is True


# ── compress_chronicle(): 表格化，而不是删掉 ──────────────────


def test_compress_collapses_narrative_to_the_value_bearing_sentence() -> None:
    compressed = compress_chronicle(NARRATIVE)

    assert "1544" in compressed
    assert "considered" not in compressed
    assert "After that we" not in compressed
    assert len(compressed) < len(NARRATIVE)


def test_compress_joins_multiple_value_sentences_with_semicolons() -> None:
    text = "We tried the old path. Build time is 12.5s now. Tests: 308 passed."

    compressed = compress_chronicle(text)

    assert compressed == "Build time is 12.5s now; Tests: 308 passed."


def test_compress_returns_empty_string_when_nothing_has_a_value() -> None:
    text = "We discussed it. We thought about it. We moved on."

    assert compress_chronicle(text) == ""
