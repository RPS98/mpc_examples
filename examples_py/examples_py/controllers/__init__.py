#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Controller adapters exposing the :class:`IController` interface."""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

from .pid_geometric_controller import PidGeometricController
from .pid_position_geometric_controller import PidPositionGeometricController
from .pid_trajectory_geometric_controller import PidTrajectoryGeometricController
from .mpc_position_controller import MpcPositionController
from .mpc_trajectory_controller import MpcTrajectoryController

__all__ = [
    'MpcPositionController',
    'MpcTrajectoryController',
    'PidGeometricController',
    'PidPositionGeometricController',
    'PidTrajectoryGeometricController',
]
