#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Jerk-limited trajectory adapter (point-to-point API).

Python mirror of
``examples/adapters/trajectory_generators/jerk_limited/src/jerk_limited_generator.cpp``.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml
from mavpy.model import State
from trajectory_generator_jerk_limited import (
    TrajectoryParameters,
    WaypointTrajectoryController,
)

from examples_py.framework import (
    ExampleConfig,
    ITrajectoryGenerator,
    ReferenceField,
    ReferenceSample,
    make_mask,
)


_MIN_HORIZONTAL_SPEED_FOR_YAW = 0.5   # [m/s]
_MAX_YAW_RATE_REF_RAD_PER_SEC = 1.0   # [rad/s]


@dataclass
class JerkLimitedConfig:
    max_acceleration: float = 0.0
    max_jerk: float = 0.0
    max_tracking_error: float = 0.0


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


class JerkLimitedGenerator(ITrajectoryGenerator):
    """Jerk-limited smoother toward the current target waypoint (p2p)."""

    def __init__(self, cfg: JerkLimitedConfig) -> None:
        self._cfg = cfg
        self._ctrl: Optional[WaypointTrajectoryController] = None
        self._target_wp = np.zeros(3, dtype=float)
        self._path_facing = False
        self._has_prev_t = False
        self._prev_t = 0.0
        self._last_sample_t = 0.0
        self._yaw_ref_hold = 0.0
        self._last_sample = ReferenceSample()
        self._name = 'JerkLimitedGenerator'

    @staticmethod
    def load_config_from_yaml(path: str) -> JerkLimitedConfig:
        if not os.path.isfile(path):
            raise ValueError(f'Config file not found at {os.path.abspath(path)}.')
        with open(path, 'r') as f:
            root = yaml.safe_load(f)
        if root is not None and not isinstance(root, dict):
            raise ValueError('jerk_limited config: root must be a mapping.')
        cfg = JerkLimitedConfig()
        if isinstance(root, dict):
            for key in ('max_acceleration', 'max_jerk', 'max_tracking_error'):
                if key in root:
                    setattr(cfg, key, float(root[key]))
        return cfg

    def initialize(self, initial_state: State, example_cfg: ExampleConfig) -> None:
        if example_cfg.max_speed <= 0.0:
            raise ValueError(
                'JerkLimitedGenerator: ExampleConfig.max_speed must be > 0 '
                '(set in config_example.yaml).')
        self._path_facing = bool(example_cfg.path_facing)
        self._has_prev_t = False
        self._prev_t = 0.0
        self._last_sample_t = 0.0
        self._yaw_ref_hold = _quat_to_yaw(
            np.asarray(initial_state.orientation, dtype=float))

        params = TrajectoryParameters()
        params.max_speed = example_cfg.max_speed
        params.max_acceleration = self._cfg.max_acceleration
        params.max_jerk = self._cfg.max_jerk
        params.max_tracking_error = self._cfg.max_tracking_error

        self._ctrl = WaypointTrajectoryController(params)
        p0 = np.asarray(initial_state.position, dtype=float)
        self._ctrl.reset(p0)
        self._target_wp = p0.copy()

        self._last_sample = ReferenceSample(
            position=p0.copy(),
            velocity=np.zeros(3, dtype=float),
            acceleration=np.zeros(3, dtype=float),
            yaw=self._yaw_ref_hold,
        )

    def on_waypoint_changed(
        self, next_waypoint: np.ndarray, state: State, t_start: float,
    ) -> None:
        del state, t_start
        self._target_wp = np.asarray(next_waypoint, dtype=float).copy()
        # The jerk-limited controller integrates continuously; no explicit
        # replan call is needed — subsequent update() steps drive to target.

    def update(self, t: float, state: State) -> None:
        if self._ctrl is None:
            raise RuntimeError(
                'JerkLimitedGenerator: initialize() must be called before update().')

        dt = max(t - self._prev_t, 0.0) if self._has_prev_t else 0.0
        self._prev_t = t
        self._has_prev_t = True

        position = np.asarray(state.position, dtype=float)
        if dt > 0.0:
            setpoints = self._ctrl.update(dt, position, self._target_wp)
            self._last_sample.position = np.asarray(
                setpoints.position, dtype=float).copy()
            self._last_sample.velocity = np.asarray(
                setpoints.velocity, dtype=float).copy()
            self._last_sample.acceleration = np.asarray(
                setpoints.acceleration, dtype=float).copy()
        self._last_sample_t = t

        if self._path_facing:
            horiz_speed = float(np.linalg.norm(self._last_sample.velocity[:2]))
            if horiz_speed > _MIN_HORIZONTAL_SPEED_FOR_YAW:
                yaw_target = math.atan2(
                    float(self._last_sample.velocity[1]),
                    float(self._last_sample.velocity[0]))
                yaw_delta = _wrap_to_pi(yaw_target - self._yaw_ref_hold)
                step = _MAX_YAW_RATE_REF_RAD_PER_SEC * (dt if dt > 0.0 else 0.0)
                yaw_delta = max(-step, min(step, yaw_delta))
                self._yaw_ref_hold += yaw_delta
            self._last_sample.yaw = self._yaw_ref_hold
        else:
            self._last_sample.yaw = 0.0

    def evaluate(self, t: float) -> ReferenceSample:
        # Forward-project the last stage-0 sample kinematically so the MPC
        # horizon stages are consistent (pos[k+1] ≈ pos[k] + vel*dt). The
        # internal S-curve is stateful and would be mutated by a proper
        # look-ahead, so we use constant-velocity extrapolation clamped at the
        # current target waypoint.
        dt = t - self._last_sample_t
        if dt <= 0.0:
            return ReferenceSample(
                position=self._last_sample.position.copy(),
                velocity=self._last_sample.velocity.copy(),
                acceleration=self._last_sample.acceleration.copy(),
                yaw=self._last_sample.yaw,
            )
        predicted = self._last_sample.position + self._last_sample.velocity * dt
        to_target = self._target_wp - self._last_sample.position
        remaining = float(np.linalg.norm(to_target))
        travel = float(np.linalg.norm(self._last_sample.velocity)) * dt
        if remaining <= 1e-9 or travel >= remaining:
            pos = self._target_wp.copy()
            vel = np.zeros(3, dtype=float)
        else:
            pos = predicted
            vel = self._last_sample.velocity.copy()
        return ReferenceSample(
            position=pos,
            velocity=vel,
            acceleration=np.zeros(3, dtype=float),
            yaw=self._last_sample.yaw,
        )

    def provided_reference_fields(self) -> ReferenceField:
        return make_mask([
            ReferenceField.POSITION,
            ReferenceField.VELOCITY,
            ReferenceField.ACCELERATION,
        ])

    def name(self) -> str:
        return self._name
