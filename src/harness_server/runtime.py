"""Runtime state, ownership and internal child-process entry point."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import (
    ENV_DB_PATH,
    ENV_HOST,
    ENV_IMPORT_MEMORIES,
    ENV_MEMORIES_DIR,
    ENV_PORT,
)
from .paths import resolve_hermes_home

STATE_SCHEMA_VERSION = 2
DATA_ROOT_DIR_NAME = "oh-my-deepseek-harness"

#: Win32 access right that permits querying a process's identity without
#: granting control over it.
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


class RuntimeStateError(RuntimeError):
    """Runtime state is missing required fields or contains invalid data."""


def default_data_root() -> str:
    """Return the product's default data root under the platform Hermes home.

    Resolved at call time: a module constant would capture the launching
    environment's answer and keep serving it after a profile switch.
    """
    return str(resolve_hermes_home() / DATA_ROOT_DIR_NAME)


def _enforce_mode(path: Path, mode: int) -> None:
    """Apply and verify private POSIX permissions.

    Windows has no POSIX mode bits: ``os.chmod`` only toggles the read-only
    attribute, and ``st_mode`` always reports 0o777 for a writable directory.
    The Windows contract is enforced by the directory's inherited ACL, which
    the platform already restricts to the owner by default, so there is
    nothing to verify here.
    """
    if os.name == "nt":
        return
    path.chmod(mode)
    actual = stat.S_IMODE(path.stat().st_mode)
    if actual != mode:
        raise PermissionError(
            f"failed to apply private permissions to {path}: {oct(actual)} != {oct(mode)}"
        )


def _fchmod_descriptor(descriptor: int, mode: int) -> None:
    """Set the mode of an open descriptor.

    ``os.fchmod`` is POSIX-only; on Windows the descriptor's ACL comes from the
    directory it was created in, so there is nothing to set.
    """
    fchmod = getattr(os, "fchmod", None)
    if fchmod is None:
        return
    fchmod(descriptor, mode)


@dataclass(frozen=True)
class RuntimePaths:
    data_root: Path
    runtime_dir: Path
    logs_dir: Path
    state_file: Path
    lock_file: Path
    log_file: Path

    @classmethod
    def from_root(cls, data_root: str | os.PathLike[str] | None = None) -> "RuntimePaths":
        root = Path(data_root or default_data_root()).expanduser().resolve()
        runtime_dir = root / "runtime"
        logs_dir = root / "logs"
        return cls(
            data_root=root,
            runtime_dir=runtime_dir,
            logs_dir=logs_dir,
            state_file=runtime_dir / "harness-server.json",
            lock_file=runtime_dir / "harness-server.lock",
            log_file=logs_dir / "harness-server.log",
        )

    def ensure(self) -> None:
        for directory in (self.data_root, self.runtime_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)
            _enforce_mode(directory, 0o700)


@dataclass(frozen=True)
class RuntimeState:
    schema_version: int
    instance_id: str
    pid: int
    process_start_token: str
    host: str
    port: int
    version: str
    started_at: str
    log_path: str
    db_path: str

    @classmethod
    def create(
        cls,
        *,
        instance_id: str,
        pid: int,
        host: str,
        port: int,
        version: str,
        log_path: str,
        db_path: str,
        start_token: str | None = None,
    ) -> "RuntimeState":
        identity = start_token or process_start_token(pid)
        if not identity:
            raise RuntimeStateError("could not determine process start identity")
        return cls(
            schema_version=STATE_SCHEMA_VERSION,
            instance_id=instance_id,
            pid=int(pid),
            process_start_token=identity,
            host=host,
            port=int(port),
            version=version,
            started_at=datetime.now(timezone.utc).isoformat(),
            log_path=log_path,
            db_path=db_path,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RuntimeState":
        try:
            state = cls(
                schema_version=int(payload["schema_version"]),
                instance_id=str(payload["instance_id"]),
                pid=int(payload["pid"]),
                process_start_token=str(payload["process_start_token"]),
                host=str(payload["host"]),
                port=int(payload["port"]),
                version=str(payload["version"]),
                started_at=str(payload["started_at"]),
                log_path=str(payload["log_path"]),
                db_path=str(payload["db_path"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeStateError("runtime state has invalid fields") from exc
        if state.schema_version != STATE_SCHEMA_VERSION:
            raise RuntimeStateError(
                f"unsupported runtime state schema: {state.schema_version}"
            )
        if (
            not state.instance_id
            or state.pid <= 0
            or not state.process_start_token
            or not 1 <= state.port <= 65535
        ):
            raise RuntimeStateError("runtime state values are invalid")
        return state

    def public_dict(self) -> dict[str, Any]:
        """Return non-sensitive status fields; DB path and identity are excluded."""
        return {
            "schema_version": self.schema_version,
            "instance_id": self.instance_id,
            "pid": self.pid,
            "host": self.host,
            "port": self.port,
            "version": self.version,
            "started_at": self.started_at,
            "log_path": self.log_path,
        }


def read_state(paths: RuntimePaths) -> RuntimeState | None:
    if not paths.state_file.exists():
        return None
    try:
        payload = json.loads(paths.state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeStateError("runtime state is unreadable") from exc
    if not isinstance(payload, dict):
        raise RuntimeStateError("runtime state must be a JSON object")
    return RuntimeState.from_dict(payload)


def write_state(paths: RuntimePaths, state: RuntimeState) -> None:
    paths.ensure()
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".harness-server-",
        suffix=".tmp",
        dir=paths.runtime_dir,
    )
    temporary_path = Path(temporary_name)
    try:
        _fchmod_descriptor(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(asdict(state), handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, paths.state_file)
        _enforce_mode(paths.state_file, 0o600)
    except Exception:
        try:
            os.close(file_descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


def remove_state(paths: RuntimePaths) -> None:
    paths.state_file.unlink(missing_ok=True)


class ProcessLock:
    """Cross-process exclusive lock for start/status/stop state transitions."""

    #: How long to wait for a contended lock before giving up.
    LOCK_TIMEOUT_SECONDS = 20.0

    def __init__(self, paths: RuntimePaths):
        self.paths = paths
        self._handle = None

    def _lock_windows(self) -> None:
        """Take an exclusive byte-range lock on the lock file.

        ``msvcrt.locking`` locks a byte range starting at the current file
        position, so the handle must be positioned at 0 and the file must hold
        at least one byte. The handle is opened in append mode, so the guard
        byte is written only when the file is empty — otherwise every lock
        would grow the file.
        """
        import msvcrt

        handle = self._handle
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        deadline = time.monotonic() + self.LOCK_TIMEOUT_SECONDS
        while True:
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeStateError(
                        f"could not acquire the Harness Server lock within "
                        f"{self.LOCK_TIMEOUT_SECONDS:.0f}s: {self.paths.lock_file}"
                    )
                time.sleep(0.05)

    def __enter__(self) -> "ProcessLock":
        self.paths.ensure()
        self._handle = self.paths.lock_file.open("a+b")
        _enforce_mode(self.paths.lock_file, 0o600)
        if os.name == "nt":
            self._lock_windows()
        else:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def _linux_process_state(pid: int) -> str | None:
    """Return the one-letter Linux process state without parsing the comm field."""
    proc_stat = Path(f"/proc/{pid}/stat")
    if not proc_stat.is_file():
        return None
    try:
        fields_after_comm = proc_stat.read_text(encoding="utf-8").rsplit(")", 1)[1]
        fields = fields_after_comm.strip().split()
        return fields[0]
    except (IndexError, OSError):
        return None


def _windows_open_process(pid: int):
    """Open a process handle for querying, or None when the PID is gone.

    ``os.kill(pid, 0)`` raises ``WinError 87`` on Windows, so liveness and
    identity both go through ``OpenProcess``.
    """
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return None
    return handle


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if _linux_process_state(pid) == "Z":
        return False
    if os.name == "nt":
        handle = _windows_open_process(pid)
        if handle is None:
            return False
        import ctypes

        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return _linux_process_state(pid) != "Z"


def process_start_token(pid: int) -> str | None:
    """Return an OS process-start identity that changes when a PID is reused."""
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.is_file():
        try:
            fields_after_comm = proc_stat.read_text(encoding="utf-8").rsplit(")", 1)[1]
            fields = fields_after_comm.strip().split()
            return f"linux:{fields[19]}"
        except (IndexError, OSError):
            return None

    if os.name == "nt":
        # Windows: OpenProcess + GetProcessTimes -> creation FILETIME changes
        # on PID reuse, same identity semantics as /proc/<pid>/stat field 22.
        try:
            import ctypes
            from ctypes import wintypes

            handle = _windows_open_process(pid)
            if handle is None:
                return None
            try:
                creation = wintypes.FILETIME()
                exit_time = wintypes.FILETIME()
                kernel_time = wintypes.FILETIME()
                user_time = wintypes.FILETIME()
                if not ctypes.windll.kernel32.GetProcessTimes(
                    handle,
                    ctypes.byref(creation),
                    ctypes.byref(exit_time),
                    ctypes.byref(kernel_time),
                    ctypes.byref(user_time),
                ):
                    return None
                return f"win:{creation.dwHighDateTime:08x}{creation.dwLowDateTime:08x}"
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            return None

    if os.name != "nt":
        try:
            completed = subprocess.run(
                ["ps", "-ww", "-p", str(pid), "-o", "lstart="],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode == 0:
            rendered = completed.stdout.strip()
            return f"ps:{rendered}" if rendered else None
    return None


def process_arguments(pid: int) -> tuple[str, ...] | None:
    """Read process argv without relying on substring matching."""
    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    if proc_cmdline.is_file():
        try:
            arguments = [
                item.decode("utf-8", errors="replace")
                for item in proc_cmdline.read_bytes().split(b"\0")
                if item
            ]
            return tuple(arguments) or None
        except OSError:
            return None

    if os.name == "nt":
        # Windows: read the full command line via CIM (wmic is removed on
        # current Win11). Returns argv-equivalent tokens or None.
        try:
            completed = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_Process -Filter "
                    f"'ProcessId={int(pid)}').CommandLine",
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        command_line = completed.stdout.strip()
        if completed.returncode == 0 and command_line:
            # Windows argv[0] embeds the interpreter path; tokenize with
            # shlex in POSIX mode after normalizing backslashes.
            normalized = command_line.replace("\\", "/")
            try:
                return tuple(shlex.split(normalized, posix=True))
            except ValueError:
                return None
        return None

    if os.name != "nt":
        try:
            completed = subprocess.run(
                ["ps", "-ww", "-p", str(pid), "-o", "command="],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode == 0 and completed.stdout.strip():
            try:
                return tuple(shlex.split(completed.stdout.strip()))
            except ValueError:
                return None
    return None


def process_is_owned(state: RuntimeState) -> bool:
    """Verify PID reuse identity and exact child argv, allowing only exec transition."""
    if not pid_is_alive(state.pid):
        return False
    expected = (
        "-m",
        "harness_server.runtime",
        "child",
        "--instance-id",
        state.instance_id,
    )
    width = len(expected)
    for attempt in range(6):
        if process_start_token(state.pid) != state.process_start_token:
            return False
        arguments = process_arguments(state.pid)
        if arguments and any(
            arguments[index : index + width] == expected
            for index in range(len(arguments))
        ):
            return True
        if attempt < 5:
            time.sleep(0.05)
    return False


def file_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _child_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    child = subparsers.add_parser("child", add_help=False)
    child.add_argument("--instance-id", required=True)
    child.add_argument("--host", required=True)
    child.add_argument("--port", required=True, type=int)
    child.add_argument("--db-path", required=True)
    child.add_argument("--memories-dir", required=True)
    child.add_argument("--import-memories", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Internal child process; users invoke ``deepseek-harness server`` instead."""
    args = _child_parser().parse_args(argv)
    if args.command != "child":
        return 2

    os.environ[ENV_HOST] = args.host
    os.environ[ENV_PORT] = str(args.port)
    os.environ[ENV_DB_PATH] = args.db_path
    os.environ[ENV_MEMORIES_DIR] = args.memories_dir
    os.environ[ENV_IMPORT_MEMORIES] = "1" if args.import_memories else "0"

    from .server import main as server_main

    server_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
