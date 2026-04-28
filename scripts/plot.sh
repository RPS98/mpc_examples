#!/usr/bin/env bash
# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Unified visualisation entry point. Opens every relevant matplotlib window
# for either a full run directory or an explicit list of log files:
#
#   1. Main time-series dashboard (pose, velocity, thrust, references, error).
#   2. 3D trajectory.
#   3. Extras (topics outside the well-known roles), if any.
#   4. Aggregated metrics (bars + boxplots, run-dir mode only).
#
# Usage:
#   scripts/plot.sh                         (newest populated run, all windows)
#   scripts/plot.sh <run_dir>               (explicit run, all windows)
#   scripts/plot.sh <log1> [log2 ...]       (single or overlay, no metrics)
#   scripts/plot.sh [... --no-show --no-save --save PATH ...]
#
# Run mode automatically generates <run>/metrics/summary.csv via
# mav_flight_review.compute_metrics when it is missing, so the metrics
# window always has data when the aggregate logs are available.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/_lib/env.sh"
source "${SCRIPT_DIR}/_lib/discover.sh"

resolve_plot_target "$@" || exit 1

if [[ "${MODE}" == "files" ]]; then
  exec python3 -m mav_flight_review.cli \
    "${LOG_FILES[@]}" "${EXTRA_ARGS[@]}"
fi

# Run mode: ensure summary.csv exists before invoking the plotter.
summary_path="${RUN_DIR%/}/metrics/summary.csv"
if [[ ! -f "${summary_path}" ]]; then
  echo "[info] ${summary_path} missing; running compute_metrics..." >&2
  python3 -m mav_flight_review.compute_metrics --run-dir "${RUN_DIR}" \
    || echo "[warn] compute_metrics failed; aggregated window will be partial" >&2
fi

exec python3 -m mav_flight_review.cli \
  --run-dir "${RUN_DIR}" "${EXTRA_ARGS[@]}"
