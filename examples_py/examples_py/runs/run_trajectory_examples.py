#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Python mirror of ``examples_cpp/src/run_trajectory_examples.cpp``.

Iterates ``sim_config.runs[]`` from ``configs/simulation/config_example.yaml``
and executes only the entries whose generator produces a full trajectory
(``gcopter``, ``jerk_limited``, ``dynamic`` or ``mav_traj_gen``) — the
eight trajectory-showcase cases:

    - pid            + gcopter
    - pid            + jerk_limited
    - pid            + dynamic
    - pid            + mav_traj_gen
    - mpc_trajectory + gcopter
    - mpc_trajectory + jerk_limited
    - mpc_trajectory + dynamic
    - mpc_trajectory + mav_traj_gen

Any enabled entry outside this scope is skipped with a log line.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

import sys

from examples_py.framework import RunSpec
from examples_py.runs._runner import run_with_filter


_TRAJECTORY_GENERATORS = frozenset(
    {'gcopter', 'jerk_limited', 'dynamic', 'mav_traj_gen'}
)


def _is_trajectory_run(spec: RunSpec) -> bool:
    # Explicit scope wins; otherwise, the legacy generator-based whitelist.
    if spec.scope:
        return spec.scope == 'trajectory'
    return spec.generator in _TRAJECTORY_GENERATORS


def main() -> int:
    return run_with_filter(
        prog='run_trajectory_examples',
        label='trajectory',
        generator_whitelist=_is_trajectory_run,
    )


if __name__ == '__main__':
    sys.exit(main())
