#!/usr/bin/env bash
# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Regenerate only the acados position-MPC C code
# (libs/acados_position_mpc/mpc_generated_code/). Requires a previous
# `./build.sh` so mpc_acados_position is exposed under build/python/.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/../_lib/env.sh"

exec bash "${REPO_ROOT}/libs/generate_acados.sh" position "$@"
