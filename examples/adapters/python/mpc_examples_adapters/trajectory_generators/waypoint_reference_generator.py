#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Waypoint-only reference generator (no smoothing).

Python mirror of
``examples/adapters/trajectory_generators/waypoint_reference/src/waypoint_reference_generator.cpp``.
The adapter snapshots the currently active waypoint on every ``update`` and
returns it (optionally clamped to ``d_max``) in ``evaluate``; path-facing yaw
uses ``atan2`` in the horizontal plane.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

import math
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import yaml
from mavpy.model import State

from mpc_examples_framework import (
    ExampleConfig,
    ITrajectoryGenerator,
    ReferenceField,
    ReferenceSample,
    make_mask,
)


@dataclass
class WaypointReferenceConfig:
    """Parsed configuration for :class:`WaypointReferenceGenerator`."""

    d_max: float = 1.0
    reach_threshold: float = 0.2


def _quat_to_yaw(q: np.ndarray) -> float:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _clamp_to_distance(current: np.ndarray, target: np.ndarray, d_max: float) -> np.ndarray:
    delta = target - current
    dist = float(np.linalg.norm(delta))
    if dist < 1e-9 or dist <= d_max:
        return target.copy()
    return current + (delta / dist) * d_max


def _path_facing_yaw(frm: np.ndarray,
                     to: np.ndarray,
                     fallback_yaw: float,
                     reach_threshold: float) -> float:
    diff = to - frm
    if float(np.linalg.norm(diff[:2])) < reach_threshold:
        return fallback_yaw
    return math.atan2(float(diff[1]), float(diff[0]))


class WaypointReferenceGenerator(ITrajectoryGenerator):
    """Step-wise waypoint reference with no smoothing.

    Matches the legacy ``clampReference`` logic used by the original PID
    example: the reference is held at the active waypoint until the drone
    enters the acceptance radius, at which point the adapter advances to the
    next waypoint on the *next* control step.
    """

    def __init__(self, cfg: WaypointReferenceConfig) -> None:
        if cfg.d_max <= 0.0:
            raise ValueError('WaypointReferenceGenerator: d_max must be > 0.')
        if cfg.reach_threshold <= 0.0:
            raise ValueError('WaypointReferenceGenerator: reach_threshold must be > 0.')
        self._cfg = cfg
        self._waypoints: List[np.ndarray] = []
        self._wp_index = 0
        self._finished = False
        self._path_facing = False
        self._last_position = np.zeros(3, dtype=float)
        self._reference_wp = np.zeros(3, dtype=float)
        self._cached_yaw = 0.0
        self._name = 'WaypointReferenceGenerator'

    @staticmethod
    def load_config_from_yaml(path: str) -> WaypointReferenceConfig:
        """Load a :class:`WaypointReferenceConfig` from YAML.

        Expected structure::

            d_max: 1.0
            reach_threshold: 0.2
        """
        if not os.path.isfile(path):
            raise ValueError(f'Config file not found at {os.path.abspath(path)}.')
        with open(path, 'r') as f:
            root = yaml.safe_load(f)
        if root is not None and not isinstance(root, dict):
            raise ValueError('waypoint_reference config: root must be a mapping.')
        cfg = WaypointReferenceConfig()
        if isinstance(root, dict):
            if 'd_max' in root:
                cfg.d_max = float(root['d_max'])
            if 'reach_threshold' in root:
                cfg.reach_threshold = float(root['reach_threshold'])
        return cfg

    def initialize(self,
                   waypoints: List[np.ndarray],
                   initial_state: State,
                   example_cfg: ExampleConfig) -> None:
        if not waypoints:
            raise ValueError(
                'WaypointReferenceGenerator: waypoints must contain at least one element.')
        self._waypoints = [np.asarray(w, dtype=float) for w in waypoints]
        self._wp_index = 0
        self._finished = False
        self._path_facing = bool(example_cfg.path_facing)
        self._last_position = np.asarray(initial_state.position, dtype=float).copy()
        self._reference_wp = self._waypoints[0].copy()
        self._cached_yaw = _quat_to_yaw(np.asarray(initial_state.orientation, dtype=float))

    def update(self, t: float, state: State) -> None:
        del t  # unused
        self._last_position = np.asarray(state.position, dtype=float).copy()
        drone_yaw = _quat_to_yaw(np.asarray(state.orientation, dtype=float))

        # Snapshot the waypoint BEFORE advancing so the current step uses the
        # waypoint that was active at entry (matches legacy ordering).
        self._reference_wp = self._waypoints[self._wp_index].copy()
        if self._path_facing:
            self._cached_yaw = _path_facing_yaw(self._last_position,
                                                self._reference_wp,
                                                drone_yaw,
                                                self._cfg.reach_threshold)
        else:
            self._cached_yaw = 0.0

        error_to_wp = float(np.linalg.norm(self._last_position - self._reference_wp))
        if error_to_wp < self._cfg.reach_threshold:
            if self._wp_index + 1 < len(self._waypoints):
                self._wp_index += 1
            else:
                self._finished = True

    def evaluate(self, t: float) -> ReferenceSample:
        del t  # unused — this generator is stepwise, not time-parametrised
        position = _clamp_to_distance(self._last_position,
                                      self._reference_wp,
                                      self._cfg.d_max)
        return ReferenceSample(position=position, yaw=self._cached_yaw)

    def is_finished(self, t: float) -> bool:
        del t
        return self._finished

    def current_waypoint_index(self) -> int:
        return int(self._wp_index)

    def provided_reference_fields(self) -> ReferenceField:
        return make_mask([ReferenceField.POSITION])

    def name(self) -> str:
        return self._name
