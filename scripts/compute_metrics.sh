#!/usr/bin/env bash
# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Compute and print aggregate + per-segment metrics for a run directory
# (simulator_logs/<run_id>). Also writes metrics/summary.csv next to the
# CSV logs.
#
# Usage: scripts/compute_metrics.sh <run_dir> [extra args]
#        scripts/compute_metrics.sh [extra args]      (newest populated run)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/_lib/env.sh"
source "${SCRIPT_DIR}/_lib/newest_run.sh"

RUN_DIR=""
if [[ $# -ge 1 && "${1:-}" != --* && "${1:-}" != -* ]]; then
  RUN_DIR="$1"; shift
fi
if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(find_newest_run || true)"
  if [[ -z "${RUN_DIR}" ]]; then
    echo "[err] no run_dir given and no populated directory found under simulator_logs/." >&2
    exit 1
  fi
  echo "[info] no run_dir given, using newest populated: ${RUN_DIR}"
fi

exec python3 -m mav_flight_logger.compute_metrics --run-dir "${RUN_DIR}" "$@"
