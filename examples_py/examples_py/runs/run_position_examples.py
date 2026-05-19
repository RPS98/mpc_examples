#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Python mirror of ``examples_cpp/src/run_position_examples.cpp``.

Iterates ``sim_config.runs[]`` from ``configs/simulation/config_example.yaml``
and executes only the entries whose generator is ``waypoints`` — the two
position-showcase cases:

    - pid           + waypoints
    - mpc_position  + waypoints

Any enabled entry outside this scope is skipped with a log line.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

import sys

from examples_py.framework import RunSpec
from examples_py.runs._runner import run_with_filter


def _is_position_run(spec: RunSpec) -> bool:
    # Explicit scope wins; otherwise, the legacy generator-based whitelist.
    if spec.scope:
        return spec.scope == 'position'
    # Smooth generators (gcopter / jerk_limited / dynamic) are accepted in
    # the position scope so the paper's moving_path phase can feed the
    # cascade PID and Pos-MPC controllers with a continuously-moving
    # target (aerostack2 follow_reference parity). mav_traj_gen stays
    # trajectory-binary-only.
    return spec.generator in ('waypoints', 'gcopter', 'jerk_limited', 'dynamic')


def main() -> int:
    return run_with_filter(
        prog='run_position_examples',
        label='position',
        generator_whitelist=_is_position_run,
    )


if __name__ == '__main__':
    sys.exit(main())
