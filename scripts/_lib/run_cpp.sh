# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Shared launcher for the C++ position/trajectory executables.
#
# Usage: run_one_cpp <binary_name> [extra executable args...]
#
# Expects env.sh to have been sourced (REPO_ROOT, OUTPUT_DIR, EXAMPLE_CFG,
# SIM_CFG already exported).

run_one_cpp() {
  local binary="$1"; shift
  local exe="./build/examples_cpp/${binary}"
  if [[ ! -x "${exe}" ]]; then
    echo "[err] ${exe} not built. Run ./build.sh first." >&2
    return 1
  fi
  ensure_output_dir
  echo "[run cpp] ${binary}"
  "${exe}" \
    -c "${EXAMPLE_CFG}" \
    -s "${SIM_CFG}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
}
