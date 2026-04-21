#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory generator adapters exposing :class:`ITrajectoryGenerator`."""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

from .waypoint_reference_generator import WaypointReferenceGenerator
from .jerk_limited_generator import JerkLimitedGenerator
from .gcopter_generator import GcopterGenerator
from .dynamic_trajectory_generator import DynamicTrajectoryGenerator

__all__ = [
    'DynamicTrajectoryGenerator',
    'GcopterGenerator',
    'JerkLimitedGenerator',
    'WaypointReferenceGenerator',
]
