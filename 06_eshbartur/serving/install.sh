#!/usr/bin/env bash
# Install the eshbartur engine's environment with uv and check it on the GPU.
#
#   1. uv sync --frozen                     uv fetches Python 3.12 and creates .venv here, exactly as uv.lock records
#   2. uv run --frozen python selfcheck.py  the GPU self-check; `bash install.sh --no-check` skips it
#
# Needs uv (https://docs.astral.sh/uv/).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
command -v uv > /dev/null || { echo "install.sh: uv is not installed (https://docs.astral.sh/uv/getting-started/installation/)" >&2; exit 1; }
uv sync --frozen
if [ "${1:-}" = "--no-check" ]; then echo "install.sh: installed (self-check skipped)"; exit 0; fi
uv run --frozen python selfcheck.py
