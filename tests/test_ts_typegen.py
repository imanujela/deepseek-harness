"""TSGEN-001: JSON Schema → TypeScript .d.ts 生成的确定性测试。

工具载荷的 JSON Schema（schemas/harness_tools.schema.json）是唯一真源；
TypeScript 侧的类型由脚本生成，生成结果必须可重复（两次运行字节相同），
并且 ``--check`` 能在 checked-in 产物过期时以退出码 1 失败——否则 .d.ts
会悄悄漂移成第二套手工维护的“事实”。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "harness_tools.schema.json"
GEN_PATH = ROOT / "scripts" / "gen_ts_types.py"
CHECKED_IN = ROOT / "types" / "harness_types.d.ts"


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gen_ts_types", GEN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen() -> ModuleType:
    return _load_generator()


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _generated(tmp_path: Path, gen: ModuleType) -> str:
    out = tmp_path / "harness_types.d.ts"
    assert gen.main(["--schema", str(SCHEMA_PATH), "--out", str(out)]) == 0
    return out.read_text(encoding="utf-8")


def _interface_body(dts: str, name: str) -> str:
    start = dts.index(f"export interface {name} {{")
    end = dts.index("\n}", start)
    return dts[start:end]


# ── schema 形状 ───────────────────────────────────────────────


def test_schema_is_json_schema_2020_12(schema: dict) -> None:
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert {"Plan", "PlanStep", "PlanStatus", "MemoryEntry", "Checkpoint"} <= set(
        schema["$defs"]
    )


def test_schema_field_names_match_real_tool_payloads(schema: dict) -> None:
    # 这些名字来自 src/deepseek_harness/tools.py 的 _TOOL_SCHEMAS，
    # 不是凭空发明的。
    step = schema["$defs"]["PlanStep"]
    assert {"step_id", "text", "status"} <= set(step["properties"])
    assert "task_description" in schema["$defs"]["Plan"]["properties"]
    assert "lambda" in schema["$defs"]["MemoryEntry"]["properties"]
    assert "completed_step_ids" in schema["$defs"]["Checkpoint"]["properties"]
    assert step["properties"]["status"]["$ref"] == "#/$defs/StepStatus"


# ── 生成内容 ─────────────────────────────────────────────────


def test_generates_the_expected_interfaces(tmp_path: Path, gen: ModuleType) -> None:
    dts = _generated(tmp_path, gen)

    for name in ("Plan", "PlanStep", "PlanStatus", "MemoryEntry", "Checkpoint"):
        assert f"export interface {name} {{" in dts


def test_required_and_optional_fields_are_marked(tmp_path: Path, gen: ModuleType) -> None:
    body = _interface_body(_generated(tmp_path, gen), "PlanStep")

    assert "step_id: string;" in body
    assert "text: string;" in body
    assert "description?: string;" in body
    assert "key?: string;" in body


def test_enum_becomes_a_string_literal_union(tmp_path: Path, gen: ModuleType) -> None:
    dts = _generated(tmp_path, gen)

    assert "export type StepStatus =" in dts
    for value in ("pending", "in_progress", "completed", "blocked", "pending_review", "cancelled"):
        assert f'"{value}"' in dts
    assert "status: StepStatus;" in _interface_body(dts, "PlanStep")


def test_array_of_ref_becomes_typed_array(tmp_path: Path, gen: ModuleType) -> None:
    assert "steps: PlanStep[];" in _interface_body(_generated(tmp_path, gen), "Plan")


def test_date_time_format_becomes_string(tmp_path: Path, gen: ModuleType) -> None:
    body = _interface_body(_generated(tmp_path, gen), "PlanStatus")

    assert "created_at?: string;" in body
    assert "updated_at?: string;" in body
    assert "adjacency_list: Record<string, string[]>;" in body


def test_output_is_deterministic_across_two_runs(tmp_path: Path, gen: ModuleType) -> None:
    first = _generated(tmp_path, gen)
    second = _generated(tmp_path, gen)

    assert first == second
    assert first.endswith("\n")


def test_output_bytes_are_stable_on_rewrite(tmp_path: Path, gen: ModuleType) -> None:
    out = tmp_path / "harness_types.d.ts"
    assert gen.main(["--schema", str(SCHEMA_PATH), "--out", str(out)]) == 0
    before = out.read_bytes()
    assert gen.main(["--schema", str(SCHEMA_PATH), "--out", str(out)]) == 0
    assert out.read_bytes() == before


# ── --check 闸门 ─────────────────────────────────────────────


def test_check_passes_on_fresh_output(tmp_path: Path, gen: ModuleType) -> None:
    out = tmp_path / "harness_types.d.ts"

    assert gen.main(["--schema", str(SCHEMA_PATH), "--out", str(out)]) == 0
    assert gen.main(["--schema", str(SCHEMA_PATH), "--out", str(out), "--check"]) == 0


def test_check_fails_on_stale_output_and_passes_after_regen(
    tmp_path: Path, gen: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "harness_types.d.ts"
    assert gen.main(["--schema", str(SCHEMA_PATH), "--out", str(out)]) == 0
    out.write_text(out.read_text(encoding="utf-8") + "\n// hand-edited\n", encoding="utf-8")

    assert gen.main(["--schema", str(SCHEMA_PATH), "--out", str(out), "--check"]) == 1
    assert "stale" in capsys.readouterr().err.lower()

    assert gen.main(["--schema", str(SCHEMA_PATH), "--out", str(out)]) == 0
    assert gen.main(["--schema", str(SCHEMA_PATH), "--out", str(out), "--check"]) == 0


def test_check_fails_when_artifact_is_missing(tmp_path: Path, gen: ModuleType) -> None:
    out = tmp_path / "missing.d.ts"

    assert gen.main(["--schema", str(SCHEMA_PATH), "--out", str(out), "--check"]) == 1


# ── checked-in 产物与 schema 同步 ────────────────────────────


def test_checked_in_artifact_is_current(gen: ModuleType) -> None:
    assert CHECKED_IN.is_file(), "types/harness_types.d.ts 必须作为生成产物提交"
    assert gen.main(["--check"]) == 0
