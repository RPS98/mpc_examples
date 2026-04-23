#!/usr/bin/env bash
# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Regenerate only the acados trajectory-MPC C code
# (libs/acados_trajectory_mpc/mpc_generated_code/). Requires a previous
# `./build.sh` so mpc_acados_trajectory is exposed under build/python/.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/../_lib/env.sh"

exec bash "${REPO_ROOT}/libs/generate_acados.sh" trajectory "$@"
