# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Common environment bootstrap sourced by every wrapper script under
# scripts/. Sets REPO_ROOT, CWD, PYTHONPATH and LD_LIBRARY_PATH so all
# binaries and Python modules resolve bindings from build/python/ regardless
# of the user's shell setup.
#
# Usage (from another script):
#   SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
#   source "${SCRIPT_DIR}/_lib/env.sh"

set -euo pipefail

SCRIPT_DIR_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR_ROOT}/.." && pwd )"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/build/python:${PYTHONPATH:-}"

# Help the pybind .so files under build/python/mavpy/ resolve their NEEDED
# native libraries regardless of the user's existing LD_LIBRARY_PATH.
export LD_LIBRARY_PATH="${REPO_ROOT}/build/mav_model/mav_model:\
${REPO_ROOT}/build/mav_controllers/libs/pid_controller:\
${LD_LIBRARY_PATH:-}"

# Default configs shared by every wrapper.
EXAMPLE_CFG="${EXAMPLE_CFG:-configs/simulation/config_example.yaml}"
SIM_CFG="${SIM_CFG:-configs/simulation/config_simulator.yaml}"

# NOTE: OUTPUT_DIR is intentionally NOT created here. Only run helpers that
# actually need to write CSVs (run_cpp.sh, run_py.sh) materialise it via
# ensure_output_dir(). Post-processing wrappers (compute_metrics.sh,
# dashboard.sh, plot_csv.sh) must not spawn an empty simulator_logs/<id>
# directory or "find newest run" lookups pick it up instead of real runs.
ensure_output_dir() {
  if [[ -z "${OUTPUT_DIR:-}" ]]; then
    OUTPUT_DIR="simulator_logs/$(date +%Y%m%d_%H%M%S)"
  fi
  mkdir -p "${OUTPUT_DIR}"
}
