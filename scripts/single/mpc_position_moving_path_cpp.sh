#!/usr/bin/env bash
# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Paper "moving_path" phase, Position-MPC + waypoints (continuous reference).
# Same architecture as `pid_moving_path_cpp.sh`: Pos-MPC consumes the TF
# target directly through the `waypoints` local adapter (no
# smooth-trajectory generator in between), matching aerostack2's
# `follow_reference_plugin_position`.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/../_lib/env.sh"
source "${SCRIPT_DIR}/../_lib/run_cpp.sh"

EXAMPLE_CFG="${EXAMPLE_CFG:-configs/simulation/config_example_continuous.yaml}" \
  run_one_cpp position_examples \
    --only-controller mpc_position \
    --only-generator  waypoints \
    "$@"
