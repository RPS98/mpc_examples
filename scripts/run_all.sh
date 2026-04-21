#!/usr/bin/env bash
# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Run the 12 unified-framework examples and leave one CSV per run under
# simulator_logs/. Configurations used by each example are the defaults baked
# into its CLI (configs/controllers/config_<ctrl>.yaml + configs/generators/config_<traj>.yaml).
#
# Usage:
#   scripts/run_all.sh [--lang=cpp|py|both]     (default: cpp)
#
# The ``py`` backend requires the CMake build to have run at least once so the
# pure-Python adapters and framework are symlinked under build/python/.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}/build/python:${PYTHONPATH:-}"

mkdir -p simulator_logs

LANG_SEL="cpp"
for arg in "$@"; do
  case "$arg" in
    --lang=*) LANG_SEL="${arg#--lang=}" ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \?//' | head -20
      exit 0
      ;;
    *) echo "[err ] unknown argument: $arg" >&2; exit 2 ;;
  esac
done

case "$LANG_SEL" in
  cpp|py|both) ;;
  *) echo "[err ] --lang must be cpp|py|both (got: $LANG_SEL)" >&2; exit 2 ;;
esac

CONTROLLERS=("pid" "mpc_position" "mpc_trajectory")
GENERATORS=("waypoints" "jerk_limited" "gcopter" "dynamic")

controller_cfg() {
  case "$1" in
    pid)             echo "configs/controllers/config_pid.yaml" ;;
    mpc_position)    echo "configs/controllers/config_mpc.yaml" ;;
    mpc_trajectory)  echo "configs/controllers/config_mpc_trajectory.yaml" ;;
  esac
}

generator_cfg() {
  case "$1" in
    waypoints)     echo "configs/generators/config_waypoints.yaml" ;;
    jerk_limited)  echo "configs/generators/config_jerk_limited.yaml" ;;
    gcopter)       echo "configs/generators/config_gcopter.yaml" ;;
    dynamic)       echo "configs/generators/config_dynamic.yaml" ;;
  esac
}

run_cpp() {
  local ctrl="$1" gen="$2"
  local exe="./build/examples/mpc_examples_run_${ctrl}_${gen}"
  local out="simulator_logs/${ctrl}_${gen}_log.csv"
  if [[ ! -x "$exe" ]]; then
    echo "[skip cpp] $exe not built"
    return 0
  fi
  echo "[run  cpp] ${ctrl}_${gen}"
  rm -f "$out"
  local rc=0
  if [[ "$ctrl" == "pid" && "$gen" == "waypoints" ]]; then
    "$exe" \
      -c configs/simulation/config_example.yaml \
      -s configs/simulation/config_simulator.yaml \
      -p configs/controllers/config_pid.yaml \
      -f "$out" >/dev/null || rc=$?
  else
    "$exe" \
      -c configs/simulation/config_example.yaml \
      -s configs/simulation/config_simulator.yaml \
      -k "$(controller_cfg "$ctrl")" \
      -t "$(generator_cfg "$gen")" \
      -f "$out" >/dev/null || rc=$?
  fi
  if [[ $rc -ne 0 ]]; then
    echo "          [fail cpp ${ctrl}_${gen}] exit=$rc"
  else
    echo "          -> $out"
  fi
}

run_py() {
  local ctrl="$1" gen="$2"
  local script="examples/examples/${ctrl}_${gen}/run_example.py"
  local out="simulator_logs/${ctrl}_${gen}_py_log.csv"
  if [[ ! -f "$script" ]]; then
    echo "[skip py ] $script not found"
    return 0
  fi
  echo "[run  py ] ${ctrl}_${gen}"
  rm -f "$out"
  local rc=0
  if [[ "$ctrl" == "pid" && "$gen" == "waypoints" ]]; then
    python3 "$script" \
      -c configs/simulation/config_example.yaml \
      -s configs/simulation/config_simulator.yaml \
      -p configs/controllers/config_pid.yaml \
      -f "$out" >/dev/null || rc=$?
  else
    python3 "$script" \
      -c configs/simulation/config_example.yaml \
      -s configs/simulation/config_simulator.yaml \
      -k "$(controller_cfg "$ctrl")" \
      -t "$(generator_cfg "$gen")" \
      -f "$out" >/dev/null || rc=$?
  fi
  if [[ $rc -ne 0 ]]; then
    echo "          [fail py  ${ctrl}_${gen}] exit=$rc"
  else
    echo "          -> $out"
  fi
}

for ctrl in "${CONTROLLERS[@]}"; do
  for gen in "${GENERATORS[@]}"; do
    case "$LANG_SEL" in
      cpp)  run_cpp "$ctrl" "$gen" ;;
      py)   run_py  "$ctrl" "$gen" ;;
      both) run_cpp "$ctrl" "$gen"; run_py "$ctrl" "$gen" ;;
    esac
  done
done

echo ""
case "$LANG_SEL" in
  cpp)  echo "Done. 12 C++ CSV files in simulator_logs/." ;;
  py)   echo "Done. 12 Python CSV files in simulator_logs/ (suffix: _py_log.csv)." ;;
  both) echo "Done. 12 C++ + 12 Python CSV files in simulator_logs/." ;;
esac
