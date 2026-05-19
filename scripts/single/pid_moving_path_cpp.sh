#!/usr/bin/env bash
# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Paper "moving_path" phase, cascade PID + waypoints (continuous reference).
# Mirrors aerostack2's PID + moving_path flow, which uses
# `follow_reference_plugin_position`: the plugin samples the moving TF
# target and publishes it directly as `motion_reference/pose` — no
# smooth-trajectory generator (gcopter / jerk_limited) sits between the
# broadcaster and the controller. The mav equivalent is the local
# `waypoints` adapter (`WaypointReferenceGenerator::evaluate()` returns
# the latched `target_wp_`), advanced by the follow_reference emulator
# inside `WaypointsSimulator` at `target_modify_frequency` (default
# 10 Hz, matches aerostack2's
# `follow_reference_plugin_trajectory.modify_frequency`).

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/../_lib/env.sh"
source "${SCRIPT_DIR}/../_lib/run_cpp.sh"

EXAMPLE_CFG="${EXAMPLE_CFG:-configs/simulation/config_example_continuous.yaml}" \
  run_one_cpp position_examples \
    --only-controller pid \
    --only-generator  waypoints \
    "$@"
