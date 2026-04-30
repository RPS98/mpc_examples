#!/usr/bin/env bash
# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Single shell entry-point for the post-simulation phase. Two modes:
#
#   1. Run mode  — operates on a full run directory:
#        scripts/plot.sh                       (newest populated run)
#        scripts/plot.sh <run_dir>             (explicit run)
#        scripts/plot.sh <run_dir> [flags]     (forwarded to review)
#      Defers to ``mav_flight_review.review``: decodes every MCAP once,
#      computes metrics, writes <run>/metrics/summary.csv, prints the
#      summary table on stdout and renders the per-axis plots under
#      <run>/plots/. Forwarded flags include --no-show, --no-save,
#      --no-overlay, --dpi N, --workers N.
#
#   2. File mode — overlays a hand-picked list of MCAPs without a run
#      directory (no metrics, no aggregated panel):
#        scripts/plot.sh <log1> [log2 ...] [flags]
#      Defers to ``mav_flight_review.cli``.
#
# This script is the canonical place where shell code talks to the
# Python review modules. ``run_all.sh`` and the per-case wrappers
# (``scripts/_lib/env.sh::post_run_review``, used by every
# ``scripts/single/*.sh``) invoke this script — there is no other shell
# path to the post-sim pipeline.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/_lib/env.sh"
source "${SCRIPT_DIR}/_lib/discover.sh"

resolve_plot_target "$@" || exit 1

if [[ "${MODE}" == "files" ]]; then
  exec python3 -m mav_flight_review.cli \
    "${LOG_FILES[@]}" "${EXTRA_ARGS[@]}"
fi

# Run mode: defer to the unified review entry-point. Decodes each MCAP
# once, computes metrics, writes summary.csv, prints the summary table
# and renders the per-axis plots — all in the same process.
exec python3 -m mav_flight_review.review \
  --run-dir "${RUN_DIR}" "${EXTRA_ARGS[@]}"
