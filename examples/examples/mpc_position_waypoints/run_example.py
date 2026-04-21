#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Unified Python example: MpcPositionController + WaypointReferenceGenerator.

Python mirror of ``examples/examples/mpc_position_waypoints/run_example.cpp``. Consumes the
same YAML configuration files as the C++ binary so that a back-to-back CSV
comparison is possible.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES_ROOT = os.path.dirname(_THIS_DIR)
if _EXAMPLES_ROOT not in sys.path:
    sys.path.insert(0, _EXAMPLES_ROOT)

from _common import parse_full_args  # noqa: E402

from mavpy.simulator import load_simulator_parameters_from_yaml
from mpc_examples_adapters.controllers import MpcPositionController
from mpc_examples_adapters.trajectory_generators import WaypointReferenceGenerator
from mpc_examples_framework import (
    WaypointsSimulator,
    load_example_config,
    normalize_output_path,
)


def main() -> int:
    args = parse_full_args(
        prog='mpc_position_waypoints',
        default_controller_cfg='configs/controllers/config_mpc.yaml',
        default_trajectory_cfg='configs/generators/config_waypoints.yaml',
        default_output_file='simulator_logs/mpc_position_waypoints_log.csv',
    )
    output_file = normalize_output_path(args.output_file)

    example_cfg = load_example_config(args.example_config_path)
    sim_params = load_simulator_parameters_from_yaml(args.simulator_config_path)

    controller_cfg = MpcPositionController.load_config_from_yaml(args.controller_config_path)
    traj_cfg = WaypointReferenceGenerator.load_config_from_yaml(args.trajectory_config_path)

    controller = MpcPositionController(controller_cfg)
    traj_gen = WaypointReferenceGenerator(traj_cfg)

    simulator = WaypointsSimulator(controller, traj_gen, example_cfg,
                                   sim_params, output_file)
    simulator.run()
    simulator.print_benchmark()
    return 0


if __name__ == '__main__':
    sys.exit(main())
