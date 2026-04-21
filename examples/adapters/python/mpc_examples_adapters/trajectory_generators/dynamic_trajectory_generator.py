#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Dynamic (replan-capable) trajectory adapter.

Python mirror of
``examples/adapters/trajectory_generators/dynamic/src/dynamic_trajectory_generator.cpp``.

The pybind11 binding exposes a narrower API than the C++ library. Only
:func:`DynamicTrajectory.generate_trajectory` and
:func:`DynamicTrajectory.evaluate_trajectory` are available from Python, so
this adapter:

* builds the trajectory once in :meth:`initialize` via ``generate_trajectory``;
* evaluates it in :meth:`evaluate` using ``evaluate_trajectory(t)``;
* tracks the current-waypoint index in Python via nearest-neighbour to the
  mission waypoints (purely informational, matches the C++ adapter);
* applies path-facing yaw with the same slew-rate policy as the C++ adapter.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

import math
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from dynamic_trajectory_generator_py import DynamicTrajectory
from mavpy.model import State

from mpc_examples_framework import (
    ExampleConfig,
    ITrajectoryGenerator,
    ReferenceField,
    ReferenceSample,
    make_mask,
)


_MIN_HORIZONTAL_SPEED_FOR_YAW = 0.5   # [m/s]
_MAX_YAW_RATE_REF_RAD_PER_SEC = 1.0   # [rad/s]


@dataclass
class DynamicTrajectoryConfig:
    """Parsed configuration for :class:`DynamicTrajectoryGenerator`.

    The adapter has no tunables: travel speed is sourced from
    :attr:`ExampleConfig.max_speed` at ``initialize()`` time. The class is kept
    for symmetry with the C++ interface.
    """


def _quat_to_yaw(q: np.ndarray) -> float:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_to_pi(x: float) -> float:
    while x > math.pi:
        x -= 2.0 * math.pi
    while x < -math.pi:
        x += 2.0 * math.pi
    return x


def _nearest_waypoint_index(waypoints: List[np.ndarray],
                            position: np.ndarray) -> int:
    best_index = 0
    best_dist_sq = math.inf
    for i, wp in enumerate(waypoints):
        d2 = float(np.sum(np.square(wp - position)))
        if d2 < best_dist_sq:
            best_dist_sq = d2
            best_index = i
    return best_index


class DynamicTrajectoryGenerator(ITrajectoryGenerator):
    """Thin wrapper around ``dynamic_trajectory_generator_py.DynamicTrajectory``.

    Plan is generated once in :meth:`initialize`; no asynchronous replanning is
    exposed by the Python binding. ``t_min``/``t_max`` bound the evaluation
    window; past ``t_max`` the adapter holds the last position with zero
    velocity/acceleration.
    """

    def __init__(self, cfg: DynamicTrajectoryConfig) -> None:
        self._cfg = cfg
        self._traj: Optional[DynamicTrajectory] = None
        self._waypoints: List[np.ndarray] = []
        self._wp_index = 0
        self._path_facing = False
        self._has_prev_t = False
        self._prev_t = 0.0
        self._yaw_ref_hold = 0.0
        self._t_min = 0.0
        self._t_max = 0.0
        self._last_sample = ReferenceSample()
        self._name = 'DynamicTrajectoryGenerator'

    @staticmethod
    def load_config_from_yaml(path: str) -> DynamicTrajectoryConfig:
        """Return an empty :class:`DynamicTrajectoryConfig`.

        The adapter has no tunables; existence of *path* is checked only so the
        CLI contract (``-t <yaml>``) stays uniform across generators. Travel
        speed is taken from :attr:`ExampleConfig.max_speed` at initialisation.
        """
        if not os.path.isfile(path):
            raise ValueError(f'Config file not found at {os.path.abspath(path)}.')
        return DynamicTrajectoryConfig()

    def initialize(self,
                   waypoints: List[np.ndarray],
                   initial_state: State,
                   example_cfg: ExampleConfig) -> None:
        if not waypoints:
            raise ValueError('DynamicTrajectoryGenerator: waypoints must not be empty.')
        self._waypoints = [np.asarray(w, dtype=float) for w in waypoints]
        self._wp_index = 0
        self._path_facing = bool(example_cfg.path_facing)
        self._has_prev_t = False
        self._prev_t = 0.0
        self._yaw_ref_hold = _quat_to_yaw(np.asarray(initial_state.orientation, dtype=float))

        if example_cfg.max_speed <= 0.0:
            raise ValueError(
                'DynamicTrajectoryGenerator: ExampleConfig.max_speed must be > 0 '
                '(set in config_example.yaml).')
        speed = example_cfg.max_speed

        self._traj = DynamicTrajectory()
        # The Python binding applies its own yaw internally; disable it and
        # reuse the same rate-limited path-facing policy as the other adapters.
        self._traj.set_path_facing(False)
        initial_position = np.asarray(initial_state.position, dtype=float)
        self._traj.generate_trajectory(initial_position,
                                       self._yaw_ref_hold,
                                       list(self._waypoints),
                                       float(speed))
        self._t_min = float(self._traj.get_min_time())
        self._t_max = float(self._traj.get_max_time())

        self._last_sample = ReferenceSample(
            position=self._waypoints[-1].copy(),
            velocity=np.zeros(3, dtype=float),
            acceleration=np.zeros(3, dtype=float),
            yaw=self._yaw_ref_hold,
        )

    def _evaluate_sample(self, t: float) -> Optional[ReferenceSample]:
        if self._traj is None:
            return None
        t_eval = min(max(t, self._t_min), self._t_max)
        try:
            pos, vel, acc, _ = self._traj.evaluate_trajectory(float(t_eval))
        except Exception:  # noqa: BLE001 — binding raises on invalid state
            return None
        pos = np.asarray(pos, dtype=float)
        vel = np.asarray(vel, dtype=float)
        acc = np.asarray(acc, dtype=float)
        if not (np.all(np.isfinite(pos))
                and np.all(np.isfinite(vel))
                and np.all(np.isfinite(acc))):
            return None
        return ReferenceSample(position=pos, velocity=vel, acceleration=acc)

    def update(self, t: float, state: State) -> None:
        if self._traj is None:
            raise RuntimeError(
                'DynamicTrajectoryGenerator: initialize() must be called before update().')

        dt = max(t - self._prev_t, 0.0) if self._has_prev_t else 0.0
        self._prev_t = t
        self._has_prev_t = True

        position = np.asarray(state.position, dtype=float)

        if t <= self._t_max:
            sample = self._evaluate_sample(t)
            if sample is not None:
                self._last_sample.position = sample.position
                self._last_sample.velocity = sample.velocity
                self._last_sample.acceleration = sample.acceleration
        else:
            self._last_sample.velocity = np.zeros(3, dtype=float)
            self._last_sample.acceleration = np.zeros(3, dtype=float)

        if self._path_facing:
            horiz_speed = float(np.linalg.norm(self._last_sample.velocity[:2]))
            if horiz_speed > _MIN_HORIZONTAL_SPEED_FOR_YAW:
                yaw_target = math.atan2(float(self._last_sample.velocity[1]),
                                        float(self._last_sample.velocity[0]))
                yaw_delta = _wrap_to_pi(yaw_target - self._yaw_ref_hold)
                step = _MAX_YAW_RATE_REF_RAD_PER_SEC * dt
                yaw_delta = max(-step, min(step, yaw_delta))
                self._yaw_ref_hold += yaw_delta
        else:
            self._yaw_ref_hold = 0.0
        self._last_sample.yaw = self._yaw_ref_hold

        self._wp_index = _nearest_waypoint_index(self._waypoints, position)

    def evaluate(self, t: float) -> ReferenceSample:
        out = ReferenceSample(
            position=self._last_sample.position.copy(),
            velocity=self._last_sample.velocity.copy(),
            acceleration=self._last_sample.acceleration.copy(),
            yaw=self._last_sample.yaw,
        )
        if self._traj is None:
            return out
        if t > self._t_max:
            out.velocity = np.zeros(3, dtype=float)
            out.acceleration = np.zeros(3, dtype=float)
            return out
        sample = self._evaluate_sample(t)
        if sample is not None:
            out.position = sample.position
            out.velocity = sample.velocity
            out.acceleration = sample.acceleration
        out.yaw = self._yaw_ref_hold
        return out

    def is_finished(self, t: float) -> bool:
        return t >= self._t_max

    def current_waypoint_index(self) -> int:
        return int(self._wp_index)

    def provided_reference_fields(self) -> ReferenceField:
        return make_mask([
            ReferenceField.POSITION,
            ReferenceField.VELOCITY,
            ReferenceField.ACCELERATION,
        ])

    def name(self) -> str:
        return self._name
