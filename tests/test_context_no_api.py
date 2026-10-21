"""Graceful degradation when no provider API key is configured.

Layer-2 contract: with neither DEEPSEEK_API_KEY nor OPENROUTER_API_KEY set,
- importing ``deepseek_context`` never raises,
- the public engine reports UNAVAILABLE via ``is_configured()``,
- every public operation returns a well-formed no-op/degraded result,
- ``plugin.register(ctx)`` loads cleanly and registers NO context-engine hooks,
- no log output ever contains the API-key value or raw message content.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import deepseek_context
from deepseek_context import DeepSeekContextEngine
from deepseek_context import plugin as context_plugin
from deepseek_context.compressor import DeepSeekCompressor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

PROVIDER_ENV_KEYS = ("DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "DEEPSEEK_BASE_URL")

FAKE_DEEPSEEK_KEY = "sk-fake-d…0123"
FAKE_OPENROUTER_KEY = "sk-or-fake-00000000000000000000"
CONTENT_MARKER = "PRIV-TOKEN-9f3c"


# ── helpers / fixtures ───────────────────────────────────


@pytest.fixture
def no_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in PROVIDER_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def make_engine(**overrides) -> DeepSeekContextEngine:
    defaults = dict(
        model="deepseek-chat",
        context_length=10_000,
        threshold_percent=0.5,
        protect_first_n=1,
        protect_last_n=1,
        quiet_mode=True,
    )
    defaults.update(overrides)
    return DeepSeekContextEngine(**defaults)


def make_conv(n: int = 12, marker: str = "conv") -> list[dict]:
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(1, n + 1):
        role = "user" if i % 2 else "assistant"
        messages.append({"role": role, "content": f"{marker}-{i}: " + "lorem ipsum dolor " * 80})
    return messages


class RecordingContext:
    """Test double for the Hermes plugin ``ctx`` object."""

    def __init__(self) -> None:
        self.engines: list = []
        self.hooks: list = []

    def register_context_engine(self, engine) -> None:
        self.engines.append(engine)

    def register_hook(self, name, fn) -> None:
        self.hooks.append(name)


def install_openai_boom(monkeypatch: pytest.MonkeyPatch) -> list:
    """Replace openai.OpenAI with a recorder; any construction is a failure."""
    attempts: list = []

    def boom(*args, **kwargs):
        attempts.append(kwargs)
        raise AssertionError("outbound provider call attempted without API key")

    monkeypatch.setattr("openai.OpenAI", boom)
    return attempts


# ── UNAVAILABLE predicate ────────────────────────────────


class TestIsConfiguredPredicate:
    def test_false_without_any_keys(self, no_provider_keys):
        engine = make_engine()
        assert engine.api_key == ""
        assert engine.is_configured() is False

    def test_true_with_deepseek_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_DEEPSEEK_KEY)
        engine = make_engine()
        assert engine.is_configured() is True
        assert engine.base_url == "https://api.deepseek.com"
        assert engine.model == "deepseek-chat"

    def test_true_with_openrouter_key_only(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_OPENROUTER_KEY)
        engine = make_engine()
        assert engine.is_configured() is True
        assert engine.base_url == "https://openrouter.ai/api/v1"
        assert engine.model == "deepseek/deepseek-chat"

    def test_true_with_explicit_constructor_key(self, no_provider_keys):
        engine = make_engine(api_key=FAKE_DEEPSEEK_KEY)
        assert engine.is_configured() is True
        assert engine._compressor.api_key == FAKE_DEEPSEEK_KEY


# ── degraded public operations ───────────────────────────


class TestDegradedPublicOperations:
    def test_compress_is_wellformed_noop(self, no_provider_keys):
        engine = make_engine()
        messages = make_conv()
        result = engine.compress(messages, current_tokens=8_000)
        assert isinstance(result, list)
        assert result is messages
        assert all(isinstance(m, dict) and "role" in m and "content" in m for m in result)
        assert engine.compression_count == 0

    def test_compress_never_contacts_provider_without_keys(
        self, no_provider_keys, monkeypatch
    ):
        attempts = install_openai_boom(monkeypatch)
        engine = make_engine()
        messages = make_conv()
        assert engine.compress(messages, current_tokens=8_000) is messages
        assert attempts == []

    def test_compressor_generate_summary_returns_none_without_key(
        self, no_provider_keys, monkeypatch
    ):
        attempts = install_openai_boom(monkeypatch)
        compressor = DeepSeekCompressor(model="deepseek-chat", api_key="", base_url="")
        assert compressor.generate_summary(make_conv(n=2)) is None
        assert attempts == []

    def test_all_public_operations_degrade_without_keys(self, no_provider_keys):
        engine = make_engine()
        messages = make_conv()
        assert engine.name == "deepseek-context"
        assert engine.is_configured() is False

        assert engine.compress(messages, current_tokens=8_000) is messages
        assert isinstance(engine.should_compress(8_000), bool)

        engine.set_intent("convergent")
        assert engine.get_intent() == "convergent"
        assert isinstance(engine.protect_hard_constraints(messages), list)
        assert isinstance(engine.extract_hard_constraints(messages), list)
        assert isinstance(engine.inject_constraint_prefix(["不能删除文件"]), str)
        assert isinstance(
            DeepSeekContextEngine.compose_turn_tail("body"), str
        )

        engine.freeze_prefix()
        assert engine.is_prefix_frozen() is True
        assert isinstance(engine.is_prefix_stable(), bool)

        engine.on_session_start("s-degraded")
        assert isinstance(engine.get_session_summary_state("s-degraded"), dict)
        engine.on_session_end("s-degraded")

        assert engine.update_model("deepseek-chat", 64_000) is None
        engine.set_plan_complexity(4.0)
        assert engine.get_review_depth(9_000) in {
            "shallow", "medium", "deep", "deepest", "reject",
        }
        state = engine.get_context_state()
        assert isinstance(state, dict) and "review_depth" in state
        assert isinstance(engine.get_review_depth_annotation(), str)
        assert engine.update_from_response(
            {"prompt_tokens": 100, "completion_tokens": 50}
        ) is None


# ── update_model key handling (provider switch) ──────────


class TestUpdateModelKeyHandling:
    def test_update_model_adopts_and_propagates_api_key(self, no_provider_keys):
        engine = make_engine()
        assert engine.is_configured() is False
        engine.update_model(
            "deepseek-chat",
            64_000,
            base_url="https://api.deepseek.com",
            api_key=FAKE_DEEPSEEK_KEY,
        )
        assert engine.is_configured() is True
        assert engine.api_key == FAKE_DEEPSEEK_KEY
        assert engine._compressor.api_key == FAKE_DEEPSEEK_KEY
        assert engine._compressor.base_url == "https://api.deepseek.com"

    def test_update_model_to_openrouter_normalizes_model(self, no_provider_keys):
        engine = make_engine()
        engine.update_model(
            "deepseek-chat",
            32_000,
            base_url="https://openrouter.ai/api/v1",
            api_key="or-fake-key",
        )
        assert engine.model == "deepseek/deepseek-chat"
        assert engine._compressor.model == "deepseek/deepseek-chat"

    def test_update_model_without_key_stays_unconfigured(self, no_provider_keys):
        engine = make_engine()
        engine.update_model("deepseek-chat", 64_000)
        assert engine.is_configured() is False


# ── plugin.register degradation ──────────────────────────


class TestPluginRegisterDegradation:
    def test_register_without_keys_loads_and_registers_nothing(
        self, no_provider_keys, caplog
    ):
        ctx = RecordingContext()
        with caplog.at_level(logging.DEBUG, logger="deepseek_context"):
            context_plugin.register(ctx)
        assert ctx.engines == []
        assert ctx.hooks == []
        assert "UNAVAILABLE" in caplog.text

    def test_register_with_deepseek_key_registers_engine(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_DEEPSEEK_KEY)
        ctx = RecordingContext()
        context_plugin.register(ctx)
        assert len(ctx.engines) == 1
        assert ctx.engines[0].is_configured() is True

    def test_register_with_openrouter_key_registers_engine(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_OPENROUTER_KEY)
        ctx = RecordingContext()
        context_plugin.register(ctx)
        assert len(ctx.engines) == 1
        assert ctx.engines[0].is_configured() is True


# ── privacy of degraded-mode logs ────────────────────────


class TestDegradedLogsContainNoSecrets:
    def test_logs_never_contain_api_key_value(self, monkeypatch, caplog):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_DEEPSEEK_KEY)
        engine = make_engine(quiet_mode=False)

        def provider_down(prompt, max_tokens, model=None):
            raise RuntimeError("provider unavailable (simulated outage)")

        monkeypatch.setattr(engine._compressor, "_call_deepseek_llm", provider_down)
        messages = make_conv(marker=CONTENT_MARKER)
        with caplog.at_level(logging.DEBUG, logger="deepseek_context"):
            engine.compress(messages, current_tokens=8_000)
            context_plugin.register(RecordingContext())
            engine.update_model(
                "deepseek-chat", 64_000, base_url="https://api.deepseek.com"
            )
        assert FAKE_DEEPSEEK_KEY not in caplog.text
        assert CONTENT_MARKER not in caplog.text

    def test_degraded_register_hint_names_vars_not_values(
        self, no_provider_keys, caplog
    ):
        ctx = RecordingContext()
        with caplog.at_level(logging.WARNING, logger="deepseek_context"):
            context_plugin.register(ctx)
        assert "DEEPSEEK_API_KEY" in caplog.text
        assert ctx.engines == []


# ── clean-environment smoke (no Hermes host, no keys) ────


def test_import_and_degrade_in_clean_subprocess_without_keys():
    env = {k: v for k, v in os.environ.items() if k not in PROVIDER_ENV_KEYS}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    code = textwrap.dedent(
        """
        import os
        assert not os.environ.get("DEEPSEEK_API_KEY")
        assert not os.environ.get("OPENROUTER_API_KEY")

        import deepseek_context
        assert deepseek_context.HERMES_CONTEXT_AVAILABLE is False

        import deepseek_context.plugin as plugin
        from deepseek_context import DeepSeekContextEngine

        engine = DeepSeekContextEngine(quiet_mode=True, context_length=10_000,
                                       threshold_percent=0.5,
                                       protect_first_n=1, protect_last_n=1)
        assert engine.is_configured() is False
        msgs = [{"role": "system", "content": "sys"}] + [
            {"role": ("user" if i % 2 else "assistant"), "content": "x" * 1200}
            for i in range(1, 12)
        ]
        assert engine.compress(msgs, current_tokens=8_000) is msgs
        engine.on_session_start("probe")
        engine.update_from_response({"prompt_tokens": 5})
        engine.on_session_end("probe")
        print("DEGRADED-OK")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout
    assert "DEGRADED-OK" in completed.stdout
