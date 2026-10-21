"""Platform-aware path resolution for the Harness product.

The product baseline is Linux/macOS, where the Hermes home is ``~/.hermes``.
On Windows the Hermes home is ``%LOCALAPPDATA%\\hermes`` — the same rule the
Hermes Agent core applies (``hermes_constants._get_platform_default_hermes_home``).
Hardcoding ``~/.hermes`` therefore deploys the plugins somewhere the host never
reads, so every path in the product resolves through this module.

Two rules govern this module:

1. Resolve at CALL time, never at import time. A module-level constant captures
   the launching environment's answer and a later profile switch silently keeps
   using it.
2. Prefer an explicit ``HERMES_HOME`` over the platform default, so a test or a
   second profile can point the product somewhere else without touching the
   host's environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping

HERMES_HOME_ENV = "HERMES_HOME"
DATA_DIR_SUFFIX_ENV = "HERMES_DATA_DIR_SUFFIX"
LOCAL_APPDATA_ENV = "LOCALAPPDATA"

#: Directory name of the Hermes home. The same on both platforms; only the
#: parent differs.
HERMES_DIR_NAME = "hermes"

#: Directory name of the Hermes home on POSIX.
POSIX_HERMES_DIR_NAME = ".hermes"


def is_windows() -> bool:
    """True on Windows, regardless of how the interpreter was launched."""
    return os.name == "nt" or sys.platform == "win32"


def _expand(path: str) -> Path:
    """Expand ``~`` and environment variables in *path* without resolving symlinks."""
    return Path(os.path.expanduser(os.path.expandvars(path)))


def platform_default_hermes_home(environ: Mapping[str, str] | None = None) -> Path:
    """Return the platform's default Hermes home directory.

    Windows: ``%LOCALAPPDATA%\\hermes`` (or ``~\\AppData\\Local\\hermes``).
    POSIX: ``~/.hermes``.
    """
    env = os.environ if environ is None else environ
    suffix = env.get(DATA_DIR_SUFFIX_ENV, "").strip()
    if is_windows():
        local_appdata = env.get(LOCAL_APPDATA_ENV, "").strip()
        base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
        return base / (HERMES_DIR_NAME + suffix)
    return Path.home() / (POSIX_HERMES_DIR_NAME + suffix)


def resolve_hermes_home(environ: Mapping[str, str] | None = None) -> Path:
    """Return the Hermes home this product must use.

    ``HERMES_HOME`` wins when set; otherwise the platform default applies.
    """
    env = os.environ if environ is None else environ
    explicit = env.get(HERMES_HOME_ENV, "").strip()
    if explicit:
        return _expand(explicit)
    return platform_default_hermes_home(env)


def hermes_home_from_environ(environ: Mapping[str, str] | None = None) -> Path:
    """Alias of :func:`resolve_hermes_home` for call sites that read an environ map."""
    return resolve_hermes_home(environ)


def hermes_subpath(*parts: str) -> Path:
    """Return a path under the resolved Hermes home.

    Call at use time, never at import time: the answer changes when the
    process is re-homed to a different profile.
    """
    return resolve_hermes_home().joinpath(*parts)


def memories_dir() -> Path:
    """Return the Hermes memories directory for the current home."""
    return hermes_subpath("memories")


def plugin_dir(name: str) -> Path:
    """Return the directory a Hermes plugin is deployed into."""
    return hermes_subpath("plugins", name)


__all__ = [
    "DATA_DIR_SUFFIX_ENV",
    "HERMES_DIR_NAME",
    "HERMES_HOME_ENV",
    "LOCAL_APPDATA_ENV",
    "POSIX_HERMES_DIR_NAME",
    "hermes_home_from_environ",
    "hermes_subpath",
    "is_windows",
    "memories_dir",
    "platform_default_hermes_home",
    "plugin_dir",
    "resolve_hermes_home",
]
