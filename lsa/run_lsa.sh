#!/usr/bin/env bash

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="${1:-}"
PYTHON="${2:-python3}"
LSA_NVZ="${3:-200}"
LSA_NUM="${4:-20}"
LSA_K_MIN="${5:--1.5}"
LSA_K_MAX="${6:-1.5}"


if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
    echo "Example: $0 runs/laptop-20260903-191021 [PYTHON] [LSA_NVZ] [LSA_NUM] [LSA_K_MIN] [LSA_K_MAX]" >&2
    exit 1
fi

MPLCONFIGDIR="$SCRIPT_DIR/../build/matplotlib-cache"
mkdir -p "$MPLCONFIGDIR"
export MPLCONFIGDIR

exec "$PYTHON" "$SCRIPT_DIR/linear_stability.py" \
    "$RUN_DIR" --nvz "$LSA_NVZ" --num "$LSA_NUM" \
    --kmin "$LSA_K_MIN" --kmax "$LSA_K_MAX"
