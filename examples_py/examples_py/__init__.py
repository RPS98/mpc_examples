#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Python half of the mpc_examples showcase.

Mirrors ``examples_cpp`` in pure Python on top of the pybind11 bindings
published under ``build/python/`` by the thirdparty submodules.

Top-level layout:
    - :mod:`examples_py.framework`   — IController / ITrajectoryGenerator /
                                        WaypointsSimulator and shared helpers.
    - :mod:`examples_py.controllers` — concrete controller adapters.
    - :mod:`examples_py.generators`  — concrete trajectory-generator adapters.
    - :mod:`examples_py.runs`        — two entry-point scripts matching the
                                        C++ executables (position, trajectory).
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'
__version__ = '0.1.0'
