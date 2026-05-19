#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Jerk-limited trajectory adapter (point-to-point API).

Python mirror of
``examples_cpp/src/generators/jerk_limited_generator.cpp``.

At each transition, ``on_waypoint_changed()`` runs the offline S-curve
simulator with ``[current_pos, next_waypoint]``; ``evaluate()`` then
samples the resulting trajectory using a segment-local time
``t - t_start``.
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
    EndWaypoint,
    TrajectoryGenerator,
    Waypoint,
    _NativeGeneratorConfig,
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
    #: Forwarded to TrajectoryParameters but unused by the one-shot API
    #: (the streaming time-stretch gate does not apply to offline planning).
    #: Kept for schema compatibility; <= 0 disables the bound.
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
    """Offline S-curve replanned per waypoint transition (p2p)."""

    def __init__(self, cfg: JerkLimitedConfig) -> None:
        self._cfg = cfg
        self._ctrl: Optional[TrajectoryGenerator] = None
        self._target_wp = np.zeros(3, dtype=float)
        self._hold_pos = np.zeros(3, dtype=float)
        self._duration = 0.0
        self._t_segment_start = 0.0
        self._has_plan = False
        self._max_speed = 0.0
        self._path_facing = False
        self._has_prev_t = False
        self._prev_t = 0.0
        self._yaw_ref_hold = 0.0
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
        self._yaw_ref_hold = _quat_to_yaw(
            np.asarray(initial_state.orientation, dtype=float))
        self._max_speed = float(example_cfg.max_speed)

        p0 = np.asarray(initial_state.position, dtype=float)
        self._hold_pos = p0.copy()
        self._target_wp = p0.copy()
        self._duration = 0.0
        self._t_segment_start = 0.0
        self._has_plan = False

        # Construct the offline solver once; reused across segments via generate().
        cfg = _NativeGeneratorConfig()
        cfg.params.max_speed = self._max_speed
        cfg.params.max_acceleration = self._cfg.max_acceleration
        cfg.params.max_jerk = self._cfg.max_jerk
        cfg.params.max_tracking_error = self._cfg.max_tracking_error
        self._ctrl = TrajectoryGenerator(cfg)

    def on_waypoint_changed(
        self, next_waypoint: np.ndarray, state: State, t_start: float,
    ) -> None:
        if self._ctrl is None:
            raise RuntimeError(
                'JerkLimitedGenerator: initialize() must be called before '
                'on_waypoint_changed().')
        p0 = np.asarray(state.position, dtype=float)
        next_wp = np.asarray(next_waypoint, dtype=float).copy()
        self._target_wp = next_wp
        self._hold_pos = next_wp.copy()
        self._t_segment_start = float(t_start)

        if float(np.linalg.norm(next_wp - p0)) < 1e-6:
            self._has_plan = False
            self._duration = 0.0
            return

        # Two-waypoint hop. Pin the C1 initial conditions of the segment
        # to the live drone state (v0 = state.linear_velocity, a0 = 0) so
        # the jerk-limited integrator stitches continuously across replans
        # — matching the gcopter adapter's `wp[0].velocity = ...` pattern
        # and the aerostack2 plugin behaviour. Without this, every replan
        # reactive to a follow_reference modify injects a step in the
        # commanded velocity.
        start = Waypoint(p0)
        start.velocity = np.asarray(state.linear_velocity, dtype=float).copy()
        wps = [start, EndWaypoint(next_wp)]
        self._has_plan = bool(self._ctrl.generate(wps, self._max_speed))
        if not self._has_plan:
            self._duration = 0.0
            return
        self._duration = float(self._ctrl.duration())

    def update(self, t: float, state: State) -> None:
        del state
        if self._ctrl is None:
            raise RuntimeError(
                'JerkLimitedGenerator: initialize() must be called before update().')

        dt = max(t - self._prev_t, 0.0) if self._has_prev_t else 0.0
        self._prev_t = t
        self._has_prev_t = True

        if not self._has_plan:
            return
        t_local = max(0.0, min(t - self._t_segment_start, self._duration))

        if self._path_facing:
            vel = np.asarray(self._ctrl.velocity(t_local), dtype=float)
            horiz_speed = float(np.linalg.norm(vel[:2]))
            if horiz_speed > _MIN_HORIZONTAL_SPEED_FOR_YAW:
                yaw_target = math.atan2(float(vel[1]), float(vel[0]))
                yaw_delta = _wrap_to_pi(yaw_target - self._yaw_ref_hold)
                step = _MAX_YAW_RATE_REF_RAD_PER_SEC * dt
                yaw_delta = max(-step, min(step, yaw_delta))
                self._yaw_ref_hold += yaw_delta
        else:
            self._yaw_ref_hold = 0.0

    def evaluate(self, t: float) -> ReferenceSample:
        sample = ReferenceSample(yaw=self._yaw_ref_hold)
        if not self._has_plan or self._ctrl is None:
            sample.position = self._hold_pos.copy()
            return sample
        t_rel = t - self._t_segment_start
        t_eval = max(0.0, min(t_rel, self._duration))
        sample.position = np.asarray(self._ctrl.position(t_eval), dtype=float).copy()
        sample.velocity = np.asarray(self._ctrl.velocity(t_eval), dtype=float).copy()
        sample.acceleration = np.asarray(
            self._ctrl.acceleration(t_eval), dtype=float).copy()

        # Past the segment end, freeze on the exact target with zero motion.
        # The integrator stops once ‖v‖ ≤ settle_velocity, which can leave
        # the last grabbed sample a few centimetres short of the waypoint;
        # pin the outgoing reference to the requested target so the
        # controller does not see a residual offset during the hover plateau.
        if t_rel >= self._duration:
            sample.position = self._target_wp.copy()
            sample.velocity = np.zeros(3, dtype=float)
            sample.acceleration = np.zeros(3, dtype=float)
        return sample

    def provided_reference_fields(self) -> ReferenceField:
        return make_mask([
            ReferenceField.POSITION,
            ReferenceField.VELOCITY,
            ReferenceField.ACCELERATION,
        ])

    def name(self) -> str:
        return self._name
