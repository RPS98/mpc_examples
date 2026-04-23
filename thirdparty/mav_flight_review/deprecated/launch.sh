#!/usr/bin/env bash
# Launch the rerun viewer for the sample flight logs.
#
# Usage:
#   ./launch.sh           <- opens flight_py.mcap (Python logger)
#   ./launch.sh cpp       <- opens flight_cpp.mcap (C++ logger)
#   ./launch.sh <path>    <- opens any .mcap file
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_PKG="$SCRIPT_DIR/python"

# Install the package if mav-view is not yet on PATH.
if ! command -v mav-view &>/dev/null; then
  echo "mav-view not found — installing package..."
  pip install -e "$PYTHON_PKG"
fi

# Resolve target MCAP file.
ARG="${1:-}"
case "$ARG" in
  ""|py)   MCAP="$SCRIPT_DIR/samples/flight_py.mcap"  ;;
  cpp)     MCAP="$SCRIPT_DIR/samples/flight_cpp.mcap" ;;
  *)       MCAP="$ARG" ;;
esac

if [[ ! -f "$MCAP" ]]; then
  echo "Error: file not found: $MCAP" >&2
  exit 1
fi

echo "Opening $MCAP with rerun..."
mav-view "$MCAP"
