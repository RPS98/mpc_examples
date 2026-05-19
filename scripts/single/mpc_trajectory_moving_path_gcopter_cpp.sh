#!/usr/bin/env bash
# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Paper "moving_path" phase, Trajectory-MPC + gcopter (continuous reference).
# Same controller/generator pair as the triangle case, but the simulator
# runs in mission_mode=continuous so the drone flows through every
# waypoint without idle holds.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/../_lib/env.sh"
source "${SCRIPT_DIR}/../_lib/run_cpp.sh"

EXAMPLE_CFG="${EXAMPLE_CFG:-configs/simulation/config_example_continuous.yaml}" \
  run_one_cpp trajectory_examples \
    --only-controller mpc_trajectory \
    --only-generator  gcopter \
    "$@"
