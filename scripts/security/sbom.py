"""SEC-001 SBOM generation: build a CycloneDX-light JSON SBOM from pyproject.toml.

Portable across POSIX and Windows: the shell wrapper (``sbom.sh``) delegates
here, and CI on Windows calls this module directly. The previous
implementation embedded the Python in a bash heredoc, so it could not run on a
host without bash or a ``python3`` on PATH.

Usage:
    python scripts/security/sbom.py [OUT]
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


def repo_root() -> Path:
    """Return the repository root, preferring git's answer."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return Path(completed.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass
    return Path(__file__).resolve().parents[2]


def _split_requirement(req: str) -> tuple[str, str]:
    """Split a requirement string into (name, version-spec)."""
    name = req.split(";")[0].split("[")[0].strip()
    if " " in name or ">" in name or "<" in name or "=" in name:
        match = re.match(r"([A-Za-z0-9_.-]+)\s*(.*)", name)
        if match:
            return match.group(1), match.group(2).strip()
    return name, ""


def build_sbom(root: Path) -> dict:
    """Return the CycloneDX SBOM payload for the project at *root*."""
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    deps = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        for dep in extra:
            if dep not in deps:
                deps.append(dep)

    components = []
    for req in deps:
        name, spec = _split_requirement(req)
        components.append(
            {
                "name": name,
                "version": spec or "any",
                "purl": f"pkg:pypi/{name.lower()}",
            }
        )

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "component": {
            "name": project["name"],
            "version": project["version"],
            "type": "library",
        },
        "serialNumber": "urn:uuid:00000000-0000-0000-0000-000000000000",
        "dependencies": components,
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    root = repo_root()
    out = Path(args[0]) if args else root / "dist" / "sbom.json"
    sbom = build_sbom(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(sbom, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"SBOM written to {out} ({len(sbom['dependencies'])} components)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
