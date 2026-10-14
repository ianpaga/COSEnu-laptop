#!/usr/bin/env bash

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="${1:-}"
PYTHON="${2:-python3}"

if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
    echo "Example: $0 runs/laptop-20260903-191021 [PYTHON]" >&2
    exit 1
fi

MPLCONFIGDIR="$SCRIPT_DIR/../build/matplotlib-cache"
mkdir -p "$MPLCONFIGDIR"
export MPLCONFIGDIR

exec "$PYTHON" "$SCRIPT_DIR/plotting_cosenu.py" "$RUN_DIR"
