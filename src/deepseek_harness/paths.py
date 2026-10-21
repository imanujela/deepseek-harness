"""Hermes-home paths for the plugin layer.

The plugins run inside the Hermes Agent process, so they must read the SAME
home the host uses. On Windows that is ``%LOCALAPPDATA%\\hermes``; a hardcoded
``~/.hermes`` writes the audit trail and the learning log somewhere the host
never reads.

Resolution happens at CALL time, never at import time: a module-level constant
captures the launching environment's answer, and a later profile switch then
silently keeps using the wrong home.
"""

from __future__ import annotations

from pathlib import Path

from harness_server.paths import hermes_home_from_environ, resolve_hermes_home


def hermes_home() -> Path:
    """Return the Hermes home of the running process."""
    return resolve_hermes_home()


def memories_path(*parts: str) -> Path:
    """Return a path under the Hermes memories directory."""
    return hermes_home().joinpath("memories", *parts)


def reflections_path(*parts: str) -> Path:
    """Return a path under the Hermes reflections directory."""
    return hermes_home().joinpath("reflections", *parts)


def map_path() -> Path:
    """Return the MAP.md navigation file path."""
    return hermes_home() / "MAP.md"


__all__ = [
    "hermes_home",
    "hermes_home_from_environ",
    "map_path",
    "memories_path",
    "reflections_path",
    "resolve_hermes_home",
]
