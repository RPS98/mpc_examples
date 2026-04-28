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
source "${SCRIPT_DIR}/_lib/discover.sh"

resolve_run_dir "$@" || exit 1
[[ "${CONSUMED_ARGS}" -eq 1 ]] && shift

exec python3 -m mav_flight_review.compute_metrics --run-dir "${RUN_DIR}" "$@"
