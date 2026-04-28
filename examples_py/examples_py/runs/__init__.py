#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Entry-point runners for the Python half of the mpc_examples showcase.

Each module mirrors one of the two C++ executables under ``examples_cpp``:
    - :mod:`examples_py.runs.run_position_examples`   (waypoints generator)
    - :mod:`examples_py.runs.run_trajectory_examples`
      (gcopter / jerk_limited / dynamic / mav_traj_gen)
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'
