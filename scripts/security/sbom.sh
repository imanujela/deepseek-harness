#!/usr/bin/env bash
# SEC-001 SBOM generation: thin POSIX wrapper around the portable Python
# implementation. The logic lives in scripts/security/sbom.py so a host
# without bash (Windows CI) can run it directly.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
OUT="${1:-${ROOT}/dist/sbom.json}"
PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python
exec "$PYTHON" "${ROOT}/scripts/security/sbom.py" "$OUT"
