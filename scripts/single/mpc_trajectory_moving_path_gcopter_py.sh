#!/usr/bin/env bash
# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Paper "moving_path" phase, Trajectory-MPC + gcopter (continuous reference).
# Python mirror of mpc_trajectory_moving_path_gcopter_cpp.sh.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/../_lib/env.sh"
source "${SCRIPT_DIR}/../_lib/run_py.sh"

EXAMPLE_CFG="${EXAMPLE_CFG:-configs/simulation/config_example_continuous.yaml}" \
  run_one_py run_trajectory_examples \
    --only-controller mpc_trajectory \
    --only-generator  gcopter \
    "$@"
