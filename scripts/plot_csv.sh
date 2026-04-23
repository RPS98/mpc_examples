#!/usr/bin/env bash
# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Open an interactive plot for a single log CSV using plot_results.
# By default the figures are shown on screen; pass --save for PNGs.
#
# Usage: scripts/plot_csv.sh <csv_path> [extra args forwarded to plot_results]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/_lib/env.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <csv_path> [extra args]" >&2
  exit 2
fi
CSV_PATH="$1"; shift

exec python3 -m mav_flight_logger.plot_results -f "${CSV_PATH}" "$@"
