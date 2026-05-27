#!/usr/bin/env bash
# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Regenerate the acados-exported MPC artefacts consumed by the C++ and Python
# examples. The generator runs from REPO_ROOT so that the relative
# ``solver.export_dir`` fields in
# libs/acados_{position,trajectory}_mpc/configs/solver_definition.yaml resolve
# under ``libs/acados_{position,trajectory}_mpc/``. The resulting
# ``acados_ocp.json`` files therefore embed correct absolute paths to the
# sibling ``mpc_generated_code/`` C code.
#
# Usage:
#   ./libs/generate_acados.sh [position|trajectory|both]    (default: both)
#
# Requirements:
#   - A prior ``./build.sh`` so ``build/python/`` exposes the in-repo
#     ``mpc_acados_{position,trajectory}`` packages via symlinks.
#   - acados installed and on PYTHONPATH (acados_template).

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "${REPO_ROOT}"

TARGET="${1:-both}"
case "$TARGET" in
  position|trajectory|both) ;;
  -h|--help)
    grep '^#' "$0" | sed 's/^# \?//' | head -20
    exit 0
    ;;
  *) echo "error: argument must be position|trajectory|both (got: $TARGET)" >&2; exit 2 ;;
esac

PY_ROOT="${REPO_ROOT}/build/python"
if [[ -d "${PY_ROOT}/mpc_acados_position" && -d "${PY_ROOT}/mpc_acados_trajectory" ]]; then
  export PYTHONPATH="${PY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
else
  MPC_SRC="${REPO_ROOT}/thirdparty/mpc"
  for sub in \
    "mpc_acados_core" \
    "controllers/position/mpc_acados_position" \
    "controllers/trajectory/mpc_acados_trajectory"
  do
    if [[ ! -d "${MPC_SRC}/${sub}" ]]; then
      echo "error: missing source ${MPC_SRC}/${sub}; did you run 'git submodule update --init --recursive'?" >&2
      exit 1
    fi
  done
  # The parent of each package is what goes on PYTHONPATH.
  export PYTHONPATH="${MPC_SRC}:${MPC_SRC}/controllers/position:${MPC_SRC}/controllers/trajectory${PYTHONPATH:+:${PYTHONPATH}}"
fi

generate_one() {
  local variant="$1"
  local yaml="${REPO_ROOT}/libs/acados_${variant}_mpc/configs/solver_definition.yaml"
  local export_dir="${REPO_ROOT}/libs/acados_${variant}_mpc"

  if [[ ! -f "${yaml}" ]]; then
    echo "error: solver definition not found: ${yaml}" >&2
    exit 1
  fi

  echo "[generate ${variant}] using ${yaml}"
  echo "[generate ${variant}] export_dir -> ${export_dir}/mpc_generated_code/"

  # Wipe the previous generated code so stale artefacts cannot be picked up.
  rm -rf "${export_dir}/mpc_generated_code"

  # Resolve mpc_acados_<variant> once and surface which copy we will use
  # (the user has a separate pip-installed copy that could otherwise win).
  local resolved
  resolved="$(python3 - "$variant" <<'PY'
import importlib
import os
import sys

variant = sys.argv[1]
module = importlib.import_module(f'mpc_acados_{variant}')
print(os.path.realpath(module.__file__))
PY
  )"
  echo "[generate ${variant}] mpc_acados_${variant} -> ${resolved}"

  python3 - "$yaml" <<PY
import sys
from mpc_acados_${variant} import AcadosMPCSolver

AcadosMPCSolver(
    solver_definition_path=sys.argv[1],
    generate_acados_solver=True,
    generate_acados_simulator=True,
    generate_code=True,
)
PY
}

case "$TARGET" in
  position)   generate_one position ;;
  trajectory) generate_one trajectory ;;
  both)       generate_one position; generate_one trajectory ;;
esac

echo "Done. Regenerated acados artefacts under libs/acados_*_mpc/mpc_generated_code/."
