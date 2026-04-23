#!/usr/bin/env bash
# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Regenerate the acados-exported C code for BOTH the position and trajectory
# MPC variants (libs/acados_{position,trajectory}_mpc/mpc_generated_code/).
# Requires a previous `./build.sh` so the Python mpc_acados_* packages are
# exposed under build/python/.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/../_lib/env.sh"

exec bash "${REPO_ROOT}/libs/generate_acados.sh" both "$@"
