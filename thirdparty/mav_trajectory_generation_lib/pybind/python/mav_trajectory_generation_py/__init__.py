# Copyright 2025 mav_trajectory_generation_lib contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""mav_trajectory_generation_py — Python bindings for mav_trajectory_generation_lib."""

__license__ = "Apache-2.0"
__version__ = "0.1.0"

from ._mav_trajectory_generation_bindings import (
    EndWaypoint,
    OptimizationConfig,
    SegmentPolynomial,
    Solver,
    Spline,
    TrajectoryGenerator,
    TrajectorySample,
    Waypoint,
)
from .trajectory import (
    GeneratorConfig,
    Trajectory,
    TrajectoryPoint,
    load_generator_config,
)

__all__ = [
    "EndWaypoint",
    "GeneratorConfig",
    "OptimizationConfig",
    "SegmentPolynomial",
    "Solver",
    "Spline",
    "Trajectory",
    "TrajectoryGenerator",
    "TrajectoryPoint",
    "TrajectorySample",
    "Waypoint",
    "load_generator_config",
]
