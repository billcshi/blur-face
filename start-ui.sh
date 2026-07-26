#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x ".venv/bin/python" ]]; then
    echo "[ERROR] .venv not found. Run ./init.sh first." >&2
    exit 1
fi
exec ".venv/bin/python" -m blurface.webui "$@"
