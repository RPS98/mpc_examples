#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Dynamic (async restitching) trajectory adapter (point-to-point API).

Python mirror of
``examples/adapters/trajectory_generators/dynamic/src/dynamic_trajectory_generator.cpp``.

The pybind11 binding exposes a narrower API than the C++ library — only
``generate_trajectory``/``evaluate_trajectory``/``get_min_time``/
``get_max_time``/``set_path_facing``/``get_was_trajectory_regenerated``. In
particular, ``updateVehiclePosition`` and ``setWaypoints`` are **not**
exposed, so each call to :meth:`on_waypoint_changed` reseeds the trajectory
via :func:`DynamicTrajectory.generate_trajectory` from the current vehicle
state to the new waypoint.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from dynamic_trajectory_generator_py import DynamicTrajectory
from mavpy.model import State

from examples_py.framework import (
    ExampleConfig,
    ITrajectoryGenerator,
    ReferenceField,
    ReferenceSample,
    make_mask,
)


_MIN_HORIZONTAL_SPEED_FOR_YAW = 0.5   # [m/s]
_MAX_YAW_RATE_REF_RAD_PER_SEC = 1.0   # [rad/s]
_MIN_SEGMENT_LENGTH = 1e-6            # [m]


@dataclass
class DynamicTrajectoryConfig:
    """Placeholder configuration for :class:`DynamicTrajectoryGenerator`.

    The adapter has no tunables: travel speed is sourced from
    :attr:`ExampleConfig.max_speed` at ``initialize()`` time. The class is
    kept for symmetry with the C++ interface.
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


def _finite_vec3(v: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(v)))


class DynamicTrajectoryGenerator(ITrajectoryGenerator):
    """Replan-on-waypoint-change wrapper around ``DynamicTrajectory`` (p2p).

    Each :meth:`on_waypoint_changed` call invokes the binding's one-shot
    :func:`generate_trajectory` to build a fresh single-segment plan from the
    current vehicle position to the new waypoint. While evaluating within
    the planned window we use the polynomial sample; once past ``t_max`` the
    reference freezes at the target waypoint with zero velocity/acceleration
    to avoid asynchronous restitching from perturbing the hover state.
    """

    def __init__(self, cfg: DynamicTrajectoryConfig) -> None:
        self._cfg = cfg
        self._traj: Optional[DynamicTrajectory] = None
        self._path_facing = False
        self._max_speed = 0.0
        self._has_prev_t = False
        self._prev_t = 0.0
        self._yaw_ref_hold = 0.0
        self._t_min = 0.0
        self._t_max = 0.0
        self._has_plan = False
        self._segment_completed = False
        self._hold_pos = np.zeros(3, dtype=float)
        self._target_wp = np.zeros(3, dtype=float)
        self._last_sample = ReferenceSample()
        self._name = 'DynamicTrajectoryGenerator'

    @staticmethod
    def load_config_from_yaml(path: str) -> DynamicTrajectoryConfig:
        if not os.path.isfile(path):
            raise ValueError(f'Config file not found at {os.path.abspath(path)}.')
        return DynamicTrajectoryConfig()

    def initialize(self, initial_state: State, example_cfg: ExampleConfig) -> None:
        if example_cfg.max_speed <= 0.0:
            raise ValueError(
                'DynamicTrajectoryGenerator: ExampleConfig.max_speed must be > 0 '
                '(set in config_example.yaml).')

        self._path_facing = bool(example_cfg.path_facing)
        self._max_speed = float(example_cfg.max_speed)
        self._has_prev_t = False
        self._prev_t = 0.0
        self._yaw_ref_hold = _quat_to_yaw(
            np.asarray(initial_state.orientation, dtype=float))

        self._hold_pos = np.asarray(initial_state.position, dtype=float).copy()
        self._target_wp = self._hold_pos.copy()
        self._t_min = 0.0
        self._t_max = 0.0
        self._has_plan = False
        self._segment_completed = False

        self._traj = DynamicTrajectory()
        # The binding applies its own yaw internally; disable it and reuse the
        # same rate-limited path-facing policy as the other adapters.
        self._traj.set_path_facing(False)

        self._last_sample = ReferenceSample(
            position=self._hold_pos.copy(),
            velocity=np.zeros(3, dtype=float),
            acceleration=np.zeros(3, dtype=float),
            yaw=self._yaw_ref_hold,
        )

    def on_waypoint_changed(
        self, next_waypoint: np.ndarray, state: State, t_start: float,
    ) -> None:
        del t_start
        if self._traj is None:
            raise RuntimeError(
                'DynamicTrajectoryGenerator: initialize() must be called '
                'before on_waypoint_changed().')

        p0 = np.asarray(state.position, dtype=float)
        self._target_wp = np.asarray(next_waypoint, dtype=float).copy()
        self._hold_pos = self._target_wp.copy()
        self._segment_completed = False

        if float(np.linalg.norm(self._target_wp - p0)) < _MIN_SEGMENT_LENGTH:
            self._has_plan = False
            self._t_min = 0.0
            self._t_max = 0.0
            return

        try:
            self._traj.generate_trajectory(
                p0,
                float(self._yaw_ref_hold),
                [p0.copy(), self._target_wp.copy()],
                float(self._max_speed),
            )
            self._t_min = float(self._traj.get_min_time())
            self._t_max = float(self._traj.get_max_time())
            self._has_plan = True
        except Exception:  # noqa: BLE001 — binding raises on optimiser failure
            self._has_plan = False
            self._t_min = 0.0
            self._t_max = 0.0

    def _evaluate_sample(self, t_eval: float) -> Optional[ReferenceSample]:
        if self._traj is None:
            return None
        try:
            pos, vel, acc, _ = self._traj.evaluate_trajectory(float(t_eval))
        except Exception:  # noqa: BLE001
            return None
        pos = np.asarray(pos, dtype=float)
        vel = np.asarray(vel, dtype=float)
        acc = np.asarray(acc, dtype=float)
        if not (_finite_vec3(pos) and _finite_vec3(vel) and _finite_vec3(acc)):
            return None
        return ReferenceSample(position=pos, velocity=vel, acceleration=acc)

    def update(self, t: float, state: State) -> None:
        del state
        if self._traj is None:
            raise RuntimeError(
                'DynamicTrajectoryGenerator: initialize() must be called before update().')

        dt = max(t - self._prev_t, 0.0) if self._has_prev_t else 0.0
        self._prev_t = t
        self._has_prev_t = True

        # Keep internal bounds fresh only while within the planned window;
        # past t_max switch to a static setpoint and ignore further library
        # state — otherwise asynchronous restitching can push the reference
        # velocity/acceleration out of bounds during hover.
        if self._has_plan and not self._segment_completed:
            try:
                self._t_max = float(self._traj.get_max_time())
            except Exception:  # noqa: BLE001
                pass

        if (self._has_plan and not self._segment_completed
                and self._t_min <= t <= self._t_max):
            t_eval = min(max(t, self._t_min), self._t_max)
            sample = self._evaluate_sample(t_eval)
            if sample is not None:
                self._last_sample.position = sample.position
                self._last_sample.velocity = sample.velocity
                self._last_sample.acceleration = sample.acceleration
        elif self._has_plan and t > self._t_max:
            self._segment_completed = True
            self._last_sample.position = self._target_wp.copy()
            self._last_sample.velocity = np.zeros(3, dtype=float)
            self._last_sample.acceleration = np.zeros(3, dtype=float)
        elif not self._has_plan:
            self._last_sample.position = self._hold_pos.copy()
            self._last_sample.velocity = np.zeros(3, dtype=float)
            self._last_sample.acceleration = np.zeros(3, dtype=float)

        if self._path_facing:
            horiz_speed = float(np.linalg.norm(self._last_sample.velocity[:2]))
            if horiz_speed > _MIN_HORIZONTAL_SPEED_FOR_YAW:
                yaw_target = math.atan2(
                    float(self._last_sample.velocity[1]),
                    float(self._last_sample.velocity[0]))
                yaw_delta = _wrap_to_pi(yaw_target - self._yaw_ref_hold)
                step = _MAX_YAW_RATE_REF_RAD_PER_SEC * dt
                yaw_delta = max(-step, min(step, yaw_delta))
                self._yaw_ref_hold += yaw_delta
        else:
            self._yaw_ref_hold = 0.0
        self._last_sample.yaw = self._yaw_ref_hold

    def evaluate(self, t: float) -> ReferenceSample:
        out = ReferenceSample(
            position=self._last_sample.position.copy(),
            velocity=self._last_sample.velocity.copy(),
            acceleration=self._last_sample.acceleration.copy(),
            yaw=self._yaw_ref_hold,
        )
        if self._traj is None or not self._has_plan:
            out.position = self._hold_pos.copy()
            out.velocity = np.zeros(3, dtype=float)
            out.acceleration = np.zeros(3, dtype=float)
            return out
        if t > self._t_max:
            out.position = self._target_wp.copy()
            out.velocity = np.zeros(3, dtype=float)
            out.acceleration = np.zeros(3, dtype=float)
            return out
        if t < self._t_min:
            return out
        t_eval = min(max(t, self._t_min), self._t_max)
        sample = self._evaluate_sample(t_eval)
        if sample is not None:
            out.position = sample.position
            out.velocity = sample.velocity
            out.acceleration = sample.acceleration
        return out

    def provided_reference_fields(self) -> ReferenceField:
        return make_mask([
            ReferenceField.POSITION,
            ReferenceField.VELOCITY,
            ReferenceField.ACCELERATION,
        ])

    def name(self) -> str:
        return self._name
