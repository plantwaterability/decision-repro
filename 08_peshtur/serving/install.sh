#!/usr/bin/env bash
# Install the peshtur engine's environment with uv and check it on the GPU.
#
#   1. PaddlePaddle's GPU wheel (not on PyPI) is fetched into wheels/: from PaddlePaddle's own storage first and, if that fails,
#      from a mirror that holds an unmodified copy. The file's sha256 is checked whichever source it came from; a file with the
#      right sha256 that is already in wheels/ is used as it is.
#   2. uv sync --frozen                         uv fetches Python 3.12 and creates .venv here, exactly as uv.lock records
#   3. uv run --frozen python selfcheck.py      the GPU self-check; `bash install.sh --no-check` skips it
#
# Needs uv (https://docs.astral.sh/uv/), curl, and the system packages libgl1 and libglib2.0-0 (for OpenCV).
# PADDLE_WHEEL_URLS, a space-separated list of URLs, replaces the two sources below.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
WHEEL=paddlepaddle_gpu-3.2.0-cp312-cp312-linux_x86_64.whl
WHEEL_SHA256=127f7f1afbc7e268080f733718e3c667ebb06d1152dd32861a6201836c8ddba0
URLS=${PADDLE_WHEEL_URLS:-"https://paddle-whl.bj.bcebos.com/stable/cu129/paddlepaddle-gpu/$WHEEL https://huggingface.co/plantwaterer/paddlepaddle-gpu-mirror/resolve/main/$WHEEL"}
command -v uv > /dev/null || { echo "install.sh: uv is not installed (https://docs.astral.sh/uv/getting-started/installation/)" >&2; exit 1; }
sha_ok() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$WHEEL_SHA256" ]; }
mkdir -p wheels
if [ -f "wheels/$WHEEL" ] && sha_ok "wheels/$WHEEL"; then
  echo "install.sh: using wheels/$WHEEL (sha256 ok)"
else
  command -v curl > /dev/null || { echo "install.sh: curl is not installed; install it, or put $WHEEL in wheels/ by hand" >&2; exit 1; }
  [ -f "wheels/$WHEEL" ] && echo "install.sh: wheels/$WHEEL does not have the expected sha256; fetching it again" >&2
  got=""
  for u in $URLS; do
    echo "install.sh: downloading $u"
    if curl -fL --retry 1 --connect-timeout 20 --speed-limit 50000 --speed-time 60 -o "wheels/$WHEEL.part" "$u"; then
      if sha_ok "wheels/$WHEEL.part"; then mv "wheels/$WHEEL.part" "wheels/$WHEEL"; got=$u; break; fi
      echo "install.sh: the file from $u does not have sha256 $WHEEL_SHA256; trying the next source" >&2
    else
      echo "install.sh: could not download from $u; trying the next source" >&2
    fi
    rm -f "wheels/$WHEEL.part"
  done
  [ -n "$got" ] || { echo "install.sh: FAILED: no source gave $WHEEL with sha256 $WHEEL_SHA256. Put the file in wheels/ by hand and run this again." >&2; exit 1; }
  echo "install.sh: wheel from $got (sha256 ok)"
fi
uv sync --frozen
if [ "${1:-}" = "--no-check" ]; then echo "install.sh: installed (self-check skipped)"; exit 0; fi
uv run --frozen python selfcheck.py
