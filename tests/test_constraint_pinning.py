"""约束钉（ConstraintRot 修复）的行为契约。

证据：约束在上下文中消失几轮后，模型违规率从 0% 上升到 30-59%；每 ~6 轮把
3-7 条非协商约束原样钉回（约 47 tokens）可把违规率恢复到 0%，且因为注入在
user message 侧，不会破坏 system prompt 的缓存前缀。

每个测试用独立 session_id：SessionPolicyStore 是进程级单例，复用同一个
session_id 会让上一个测试的约束泄漏进来。
"""

import uuid

import pytest

from deepseek_harness import gate


@pytest.fixture(autouse=True)
def _reset_state():
    gate._TURN_COUNTS.clear()
    gate._current_hard_constraints.clear()
    yield
    gate._TURN_COUNTS.clear()
    gate._current_hard_constraints.clear()


@pytest.fixture
def sid():
    return f"pin-{uuid.uuid4().hex[:8]}"


def _turn(sid, message="继续", first=False):
    return gate.on_pre_llm_call(
        is_first_turn=first, session_id=sid, user_message=message
    )["context"]


def test_constraint_is_extracted_on_the_turn_it_appears(sid):
    _turn(sid, message="不要改动数据库，只做只读查询")
    assert any("不要改动数据库" in c for c in gate._current_hard_constraints)


def test_no_pin_before_the_cadence_is_due(sid):
    _turn(sid, message="不要改动数据库")
    for _ in range(gate._PIN_EVERY - 2):
        assert "[约束钉]" not in _turn(sid)


def test_pin_fires_on_the_cadence_turn_and_carries_the_constraint(sid):
    _turn(sid, message="不要改动数据库")
    texts = [_turn(sid) for _ in range(gate._PIN_EVERY - 1)]
    assert "[约束钉]" not in texts[0]
    pinned = [t for t in texts if "[约束钉]" in t]
    assert len(pinned) == 1
    assert "不要改动数据库" in pinned[0]
    # window restarts: the next turn is quiet again
    assert "[约束钉]" not in _turn(sid)


def test_pin_is_capped_at_seven_entries(sid):
    message = "；".join(f"不要做第{i}件事" for i in range(1, 12))
    _turn(sid, message=message)
    texts = [_turn(sid) for _ in range(gate._PIN_EVERY - 1)]
    pinned = [line for t in texts for line in t.split("\n") if line.startswith("[约束钉]")]
    assert len(pinned) == 1
    entries = pinned[0].removeprefix("[约束钉] ").split("；")
    assert len(entries) <= gate._PIN_MAX


def test_no_pin_when_no_constraints_were_ever_stated(sid):
    for _ in range(gate._PIN_EVERY):
        assert "[约束钉]" not in _turn(sid, message="继续实现")


def test_pin_does_not_touch_the_first_turn(sid):
    text = _turn(sid, message="不要改动数据库", first=True)
    assert "[约束钉]" not in text
