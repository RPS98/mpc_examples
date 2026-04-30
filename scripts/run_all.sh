#!/usr/bin/env bash
# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Run the mpc_examples showcase. Dispatches through the two C++ binaries
# (``position_examples``, ``trajectory_examples``) and/or their Python
# counterparts (``python3 -m examples_py.runs.run_*``) and then runs the
# metrics + dashboard tools shipped inside ``mav_flight_review``
# (thirdparty/mav_flight_mcap/viewer).
#
# Usage:
#   scripts/run_all.sh [--lang=cpp|py|both] [--run-id=<id>]
#                      [--no-show] [--no-save]
#
# Defaults:
#   --lang=both
#   --run-id auto-generated as YYYYmmdd_HHMMSS (shared by both langs so the
#   CSVs land under simulator_logs/<run_id>/{cpp,py}/).
#   The dashboard is opened interactively AND saved to <run>/plots/*.png.
#     --no-show: skip the matplotlib windows (e.g. on CI / headless runs).
#     --no-save: skip writing PNGs to disk.
#
# The ``py`` backend requires the CMake build to have run at least once so
# the pure-Python mirrors (examples_py, mav_flight_mcap, mav_flight_review)
# are exposed under build/python/.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}/build/python:${PYTHONPATH:-}"

# Defensive LD_LIBRARY_PATH so the pybind .so files under build/python/mavpy/
# can resolve their NEEDED native libraries regardless of any conflicting
# paths the user may already have exported.
export LD_LIBRARY_PATH="${REPO_ROOT}/build/mav_model/mav_model:\
${REPO_ROOT}/build/mav_controllers/libs/pid_controller:\
${LD_LIBRARY_PATH:-}"

LANG_SEL="both"
RUN_ID=""
SHOW=1
SAVE=1
for arg in "$@"; do
  case "$arg" in
    --lang=*)   LANG_SEL="${arg#--lang=}" ;;
    --run-id=*) RUN_ID="${arg#--run-id=}" ;;
    --show)     SHOW=1 ;;
    --no-show)  SHOW=0 ;;
    --save)     SAVE=1 ;;
    --no-save)  SAVE=0 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \?//' | head -25
      exit 0
      ;;
    *) echo "[err ] unknown argument: $arg" >&2; exit 2 ;;
  esac
done

case "$LANG_SEL" in
  cpp|py|both) ;;
  *) echo "[err ] --lang must be cpp|py|both (got: $LANG_SEL)" >&2; exit 2 ;;
esac

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi
OUT_DIR="simulator_logs/${RUN_ID}"
mkdir -p "$OUT_DIR"

echo "[run_all] run_id=${RUN_ID} · output_dir=${OUT_DIR} · lang=${LANG_SEL}"

EXAMPLE_CFG="configs/simulation/config_example.yaml"
SIM_CFG="configs/simulation/config_simulator.yaml"

run_cpp() {
  local pos_exe="./build/examples_cpp/position_examples"
  local traj_exe="./build/examples_cpp/trajectory_examples"
  for exe in "$pos_exe" "$traj_exe"; do
    if [[ ! -x "$exe" ]]; then
      echo "[skip cpp] $exe not built"
      continue
    fi
    echo "[run  cpp] $(basename "$exe")"
    "$exe" \
      -c "$EXAMPLE_CFG" \
      -s "$SIM_CFG" \
      --output-dir "$OUT_DIR"
  done
}

run_py() {
  if ! python3 -c 'import examples_py' 2>/dev/null; then
    echo "[skip py ] examples_py not available on PYTHONPATH"
    return 0
  fi
  for runner in run_position_examples run_trajectory_examples; do
    echo "[run  py ] examples_py.runs.${runner}"
    python3 -m "examples_py.runs.${runner}" \
      -c "$EXAMPLE_CFG" \
      -s "$SIM_CFG" \
      --output-dir "$OUT_DIR"
  done
}

case "$LANG_SEL" in
  cpp)  run_cpp ;;
  py)   run_py ;;
  both) run_cpp; run_py ;;
esac

echo ""
echo "[run_all] reviewing run (decode + metrics + summary + dashboard)"
REVIEW_FLAGS=()
[[ "$SHOW" -eq 0 ]] && REVIEW_FLAGS+=(--no-show)
[[ "$SAVE" -eq 0 ]] && REVIEW_FLAGS+=(--no-save)
python3 -m mav_flight_review.review --run-dir "$OUT_DIR" "${REVIEW_FLAGS[@]}" || true

echo ""
if [[ "$SAVE" -eq 1 ]]; then
  echo "[run_all] dashboard:    $OUT_DIR/plots/run.png"
  echo "[run_all] metrics fig:  $OUT_DIR/plots/run_metrics.png"
fi
echo "[run_all] summary csv:  $OUT_DIR/metrics/summary.csv"
echo "[run_all] done · $OUT_DIR"
