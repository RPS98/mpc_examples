#!/usr/bin/env bash
# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Open the per-variable plots for EVERY CSV in a run directory. Each CSV
# goes through ``mav_flight_logger.plot_results``, which produces 4 figures
# (3D trajectory, state tracking with position/orientation/velocity, control
# + angular velocity + speed-magnitude-vs-max_speed + motors, smoothness).
#
# Usage: scripts/plot_run.sh [run_dir] [--lang=cpp|py|both] [extra args]
#        scripts/plot_run.sh            (newest populated run, both languages)
#
# Extra args are forwarded to plot_results. Pass --save --no-show to batch
# export PNGs non-interactively; otherwise each CSV opens its figures and
# waits for Enter before moving on to the next.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/_lib/env.sh"
source "${SCRIPT_DIR}/_lib/newest_run.sh"

RUN_DIR=""
LANG_SEL="both"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lang=*) LANG_SEL="${1#--lang=}" ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \?//' | head -16
      exit 0
      ;;
    *)
      if [[ -z "${RUN_DIR}" && -d "$1" ]]; then
        RUN_DIR="$1"
      else
        EXTRA+=("$1")
      fi
      ;;
  esac
  shift
done

case "${LANG_SEL}" in cpp|py|both) ;; *)
  echo "[err] --lang must be cpp|py|both (got: ${LANG_SEL})" >&2; exit 2 ;;
esac

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(find_newest_run || true)"
  if [[ -z "${RUN_DIR}" ]]; then
    echo "[err] no run_dir given and no populated directory found under simulator_logs/." >&2
    exit 1
  fi
  echo "[info] using newest populated run: ${RUN_DIR}"
fi

declare -a LANG_DIRS=()
[[ "${LANG_SEL}" == cpp || "${LANG_SEL}" == both ]] && LANG_DIRS+=("${RUN_DIR}/cpp")
[[ "${LANG_SEL}" == py  || "${LANG_SEL}" == both ]] && LANG_DIRS+=("${RUN_DIR}/py")

total=0
for d in "${LANG_DIRS[@]}"; do
  [[ -d "${d}" ]] || continue
  # Skip *_metrics.csv and *_segments.csv companions written by compute_metrics.
  while IFS= read -r -d '' csv; do
    case "${csv}" in
      *_metrics.csv|*_segments.csv) continue ;;
    esac
    total=$((total + 1))
    echo "[plot ${total}] ${csv}"
    python3 -m mav_flight_logger.plot_results -f "${csv}" "${EXTRA[@]}" || true
  done < <(find "${d}" -maxdepth 1 -type f -name '*.csv' -print0 | sort -z)
done

if [[ "${total}" -eq 0 ]]; then
  echo "[warn] no CSVs found under ${RUN_DIR}/{cpp,py}/" >&2
  exit 1
fi
echo "[done] plotted ${total} CSV(s)"
