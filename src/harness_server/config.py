"""Single runtime environment contract for Harness Server."""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .paths import resolve_hermes_home

ENV_HOST = "HARNESS_HOST"
ENV_PORT = "HARNESS_PORT"
ENV_DB_PATH = "HARNESS_DB_PATH"
ENV_IMPORT_MEMORIES = "HARNESS_IMPORT_MEMORIES"
ENV_MEMORIES_DIR = "HARNESS_MEMORIES_DIR"
ENV_SUMMARY_ENABLED = "HARNESS_SUMMARY_ENABLED"
ENV_SUMMARY_OUTBOUND_POLICY = "HARNESS_SUMMARY_OUTBOUND_POLICY"
ENV_SUMMARY_ALLOW_TOOL_ARGS = "HARNESS_SUMMARY_ALLOW_TOOL_ARGUMENTS"
ENV_LOG_INCLUDE_CONTENT = "HARNESS_LOG_INCLUDE_CONTENT"
RUNTIME_ENV_VARS = frozenset(
    {
        ENV_HOST,
        ENV_PORT,
        ENV_DB_PATH,
        ENV_IMPORT_MEMORIES,
        ENV_MEMORIES_DIR,
        ENV_SUMMARY_ENABLED,
        ENV_SUMMARY_OUTBOUND_POLICY,
        ENV_SUMMARY_ALLOW_TOOL_ARGS,
        ENV_LOG_INCLUDE_CONTENT,
    }
)


def _parse_bool(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _validate_loopback(host: str) -> str:
    normalized = host.strip()
    if normalized.lower() == "localhost":
        return "localhost"
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError as exc:
        raise ValueError(f"{ENV_HOST} must be a loopback address") from exc
    if not address.is_loopback:
        raise ValueError(f"{ENV_HOST} must be a loopback address")
    return normalized


def default_db_path() -> str:
    """Return the SQLite path under the platform Hermes home.

    Resolved at call time so a profile switch is honoured; a module constant
    would keep serving the launching environment's home.
    """
    return str(resolve_hermes_home() / "mcp" / "harness.db")


def default_memories_dir() -> str:
    """Return the Hermes memories directory under the platform Hermes home."""
    return str(resolve_hermes_home() / "memories")


@dataclass(frozen=True)
class RuntimeConfig:
    """Validated local-runtime settings.

    Remote binding is deliberately out of scope for the Public Beta baseline.
    """

    host: str = "127.0.0.1"
    port: int = 8200
    db_path: str = ""
    import_memories: bool = False
    memories_dir: str = ""
    summary_enabled: bool = False
    summary_allow_tool_args: bool = False
    log_include_content: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "host", _validate_loopback(self.host))
        if not 1 <= int(self.port) <= 65535:
            raise ValueError(f"{ENV_PORT} must be between 1 and 65535")
        object.__setattr__(self, "port", int(self.port))
        object.__setattr__(
            self,
            "db_path",
            str(Path(self.db_path or default_db_path()).expanduser()),
        )
        object.__setattr__(
            self,
            "memories_dir",
            str(Path(self.memories_dir or default_memories_dir()).expanduser()),
        )

    @property
    def url_host(self) -> str:
        return f"[{self.host}]" if ":" in self.host else self.host

    @property
    def server_url(self) -> str:
        return f"http://{self.url_host}:{self.port}"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "RuntimeConfig":
        env = os.environ if environ is None else environ
        port_text = env.get(ENV_PORT, "8200")
        try:
            port = int(port_text)
        except ValueError as exc:
            raise ValueError(f"{ENV_PORT} must be an integer") from exc
        return cls(
            host=env.get(ENV_HOST, "127.0.0.1"),
            port=port,
            db_path=env.get(ENV_DB_PATH, ""),
            import_memories=_parse_bool(
                env.get(ENV_IMPORT_MEMORIES, "0"),
                name=ENV_IMPORT_MEMORIES,
            ),
            memories_dir=env.get(ENV_MEMORIES_DIR, ""),
            summary_enabled=_parse_bool(
                env.get(ENV_SUMMARY_ENABLED, "0"),
                name=ENV_SUMMARY_ENABLED,
            ),
            summary_allow_tool_args=_parse_bool(
                env.get(ENV_SUMMARY_ALLOW_TOOL_ARGS, "0"),
                name=ENV_SUMMARY_ALLOW_TOOL_ARGS,
            ),
            log_include_content=_parse_bool(
                env.get(ENV_LOG_INCLUDE_CONTENT, "0"),
                name=ENV_LOG_INCLUDE_CONTENT,
            ),
        )
