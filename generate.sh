#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

LOCAL_POSITION_MPC="$(realpath "${SCRIPT_DIR}/thirdparty/position_mpc")"
if [[ ! -d "${LOCAL_POSITION_MPC}" ]]; then
  echo "ERROR: local position_mpc path not found: ${LOCAL_POSITION_MPC}" >&2
  exit 1
fi

export PYTHONPATH="${LOCAL_POSITION_MPC}${PYTHONPATH:+:${PYTHONPATH}}"

MPC_POSITION_FILE="$(python3 - <<'PY'
import os
import mpc_position
print(os.path.realpath(mpc_position.__file__))
PY
)"

echo "Using mpc_position from: ${MPC_POSITION_FILE}"
if [[ "${MPC_POSITION_FILE}" != "${LOCAL_POSITION_MPC}"/* ]]; then
  echo "ERROR: mpc_position resolved outside local repo." >&2
  echo "Expected prefix: ${LOCAL_POSITION_MPC}" >&2
  exit 1
fi

rm -rf examples/acados_position_mpc
python3 -m mpc_position.acados_solver -c solver_definition_mpc_position.yaml
