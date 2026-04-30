#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Time-based waypoint advancement for fair controller × generator comparison.

Python mirror of
``examples/framework/include/framework/waypoint_scheduler.hpp`` and
``examples/framework/src/waypoint_scheduler.cpp``.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

from dataclasses import dataclass
from typing import List

import numpy as np


_EPS = 1e-12


@dataclass
class TickResult:
    waypoint_changed: bool = False
    active_index: int = 0
    finished: bool = False


class WaypointScheduler:
    """Advances the active waypoint on a time basis.

    Given a list of waypoints and the maximum allowed speed, precomputes a
    switch time for each waypoint as::

        t_switch[i] = t_switch[i-1]
                    + ||wp[i] - wp[i-1]|| / (max_speed * scheduler_speed_factor)
                    + settle_margin_s

    where the hop from the initial position to ``wp[0]`` follows the same
    rule. ``scheduler_speed_factor`` lies in (0, 1] and models the fact
    that smooth generators (bell-shaped or trapezoidal) never sustain
    ``max_speed`` during the whole segment; lowering the factor allocates
    more wall-clock time per hop so the drone can settle before the next
    waypoint switch. A factor of 1.0 recovers the legacy heuristic.
    """

    def __init__(self) -> None:
        self._waypoints: List[np.ndarray] = []
        self._switch_times: List[float] = []
        self._active_index: int = 0

    def initialize(
        self,
        waypoints: List[np.ndarray],
        initial_position: np.ndarray,
        max_speed: float,
        settle_margin_s: float,
        scheduler_speed_factor: float = 1.0,
    ) -> None:
        if not waypoints:
            raise ValueError('WaypointScheduler: waypoints must not be empty.')
        if max_speed <= 0.0:
            raise ValueError('WaypointScheduler: max_speed must be > 0.')
        if settle_margin_s < 0.0:
            raise ValueError('WaypointScheduler: settle_margin_s must be >= 0.')
        if scheduler_speed_factor <= 0.0 or scheduler_speed_factor > 1.0:
            raise ValueError(
                'WaypointScheduler: scheduler_speed_factor must lie in (0, 1].')

        self._waypoints = [np.asarray(wp, dtype=float).copy() for wp in waypoints]
        self._active_index = 0
        self._switch_times = []

        effective_speed = max_speed * scheduler_speed_factor
        previous = np.asarray(initial_position, dtype=float).copy()
        t_cumulative = 0.0
        for wp in self._waypoints:
            distance = float(np.linalg.norm(wp - previous))
            t_cumulative += distance / effective_speed + settle_margin_s
            self._switch_times.append(t_cumulative)
            previous = wp

    def tick(self, t: float) -> TickResult:
        result = TickResult()
        if not self._waypoints:
            return result

        previous_index = self._active_index
        while (self._active_index + 1 < len(self._waypoints)
               and t + _EPS >= self._switch_times[self._active_index]):
            self._active_index += 1

        result.active_index = self._active_index
        result.waypoint_changed = (self._active_index != previous_index)
        result.finished = (
            self._active_index + 1 == len(self._waypoints)
            and t + _EPS >= self._switch_times[-1]
        )
        return result

    def switch_time(self, i: int) -> float:
        return self._switch_times[i]

    def final_time(self) -> float:
        return self._switch_times[-1] if self._switch_times else 0.0

    def active_index(self) -> int:
        return self._active_index

    def size(self) -> int:
        return len(self._waypoints)

    def waypoint(self, i: int) -> np.ndarray:
        return self._waypoints[i]

    def switch_times(self) -> List[float]:
        return list(self._switch_times)
