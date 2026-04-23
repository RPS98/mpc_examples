# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Shared launcher for the examples_py runners.
#
# Usage: run_one_py <run_module> [extra args...]
#   e.g. run_one_py run_position_examples --only-controller pid --only-generator waypoints
#
# Expects env.sh to have been sourced.

run_one_py() {
  local module="$1"; shift
  if ! python3 -c 'import examples_py' 2>/dev/null; then
    echo "[err] examples_py not importable. Run ./build.sh first." >&2
    return 1
  fi
  ensure_output_dir
  echo "[run py ] examples_py.runs.${module}"
  python3 -m "examples_py.runs.${module}" \
    -c "${EXAMPLE_CFG}" \
    -s "${SIM_CFG}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
}
