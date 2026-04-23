#!/usr/bin/env bash
# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Open ALL interactive figures for a run: per-case per-variable plots (4
# figures per CSV, via plot_run.sh) plus the comparative dashboard at the
# end (via dashboard.sh --show).
#
# Usage: scripts/show_all.sh [run_dir] [--lang=cpp|py|both]
#        scripts/show_all.sh                     (newest populated run, both)
#
# Per-variable figures come first, one case at a time: close the 4 windows
# (or hit the X on each) to advance to the next CSV. Once every CSV has
# been visited, the comparative dashboard is shown last.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/_lib/env.sh"
source "${SCRIPT_DIR}/_lib/newest_run.sh"

RUN_DIR=""
LANG_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lang=*) LANG_ARGS=("$1") ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \?//' | head -13
      exit 0
      ;;
    *)
      if [[ -z "${RUN_DIR}" && -d "$1" ]]; then
        RUN_DIR="$1"
      else
        echo "[err] unexpected argument: $1" >&2; exit 2
      fi
      ;;
  esac
  shift
done

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(find_newest_run || true)"
  if [[ -z "${RUN_DIR}" ]]; then
    echo "[err] no run_dir given and no populated directory found under simulator_logs/." >&2
    exit 1
  fi
  echo "[info] using newest populated run: ${RUN_DIR}"
fi

echo "[show_all] per-variable plots"
bash "${SCRIPT_DIR}/plot_run.sh" "${RUN_DIR}" "${LANG_ARGS[@]}"

echo "[show_all] comparative dashboard"
bash "${SCRIPT_DIR}/dashboard.sh" "${RUN_DIR}" --show
