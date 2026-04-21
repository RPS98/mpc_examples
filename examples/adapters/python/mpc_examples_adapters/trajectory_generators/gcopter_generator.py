#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""GCOPTER trajectory adapter (offline polynomial planner).

Python mirror of
``examples/adapters/trajectory_generators/gcopter/src/gcopter_generator.cpp``.
Mirrors the C++ anchor-expansion heuristic in Python because the pybind11
binding exposes :class:`gcopterpy.trajectory.TrajectoryGenerator` but does not
accept :class:`GeneratorConfig` in its constructor.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import yaml
from gcopterpy.trajectory import (
    DroneLimits,
    DroneParameters,
    OptimizationConfig,
    TrajectoryGenerator,
    Waypoint,
)
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
class _DroneParamsSpec:
    mass: float = 1.0
    gravity: float = 9.81
    horizontal_drag: float = 0.0
    vertical_drag: float = 0.0
    parasitic_drag: float = 0.0
    speed_smooth_factor: float = 0.0


@dataclass
class _DroneLimitsSpec:
    max_velocity: float = 3.0
    max_body_rate: float = 3.0
    max_tilt_angle: float = 1.0
    min_thrust: float = 0.0
    max_thrust: float = 30.0


@dataclass
class _OptimizationSpec:
    time_weight: float = 1.0
    position_weight: float = 100.0
    velocity_weight: float = 10.0
    body_rate_weight: float = 1.0
    tilt_weight: float = 1.0
    thrust_weight: float = 1.0
    smoothing_eps: float = 1.0e-2
    integral_resolution: int = 8
    rel_cost_tol: float = 1.0e-3


@dataclass
class GcopterConfig:
    """Parsed configuration for :class:`GcopterGenerator`."""

    corridor_margin: float = 2.0
    waypoint_margin: float = 0.0
    waypoint_anchor_radius: float = 0.0
    drone_params: _DroneParamsSpec = field(default_factory=_DroneParamsSpec)
    drone_limits: _DroneLimitsSpec = field(default_factory=_DroneLimitsSpec)
    optimization: _OptimizationSpec = field(default_factory=_OptimizationSpec)


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


def _require(node: dict, key: str, section: str) -> float:
    if key not in node:
        raise ValueError(f'gcopter config: {section}.{key} is required.')
    return float(node[key])


def _expand_path_with_anchors(
    waypoints: List[np.ndarray],
    loose_margin: float,
    tight_margin: float,
    anchor_radius: float,
) -> Tuple[List[np.ndarray], List[float]]:
    """Mirror of the C++ ``expandPathWithAnchors`` used by gcopter_lib.

    Anchor mode inserts two helper points around every intermediate waypoint
    so the corridor hugs the path tightly there and stays loose in between.
    """
    n = len(waypoints)
    if n < 3 or tight_margin <= 0.0 or anchor_radius <= 0.0:
        positions = [w.copy() for w in waypoints]
        margins = [loose_margin] * (n - 1 if n > 0 else 0)
        return positions, margins

    positions: List[np.ndarray] = [waypoints[0].copy()]
    for i in range(1, n - 1):
        prev = waypoints[i - 1]
        curr = waypoints[i]
        nxt = waypoints[i + 1]
        in_vec = curr - prev
        out_vec = nxt - curr
        in_norm = float(np.linalg.norm(in_vec))
        out_norm = float(np.linalg.norm(out_vec))
        if in_norm <= 0.0 or out_norm <= 0.0:
            positions.append(curr.copy())
            continue
        r_in = min(anchor_radius, 0.45 * in_norm)
        r_out = min(anchor_radius, 0.45 * out_norm)
        positions.append(curr - in_vec * (r_in / in_norm))
        positions.append(curr.copy())
        positions.append(curr + out_vec * (r_out / out_norm))
    positions.append(waypoints[-1].copy())

    m = n - 2
    margins: List[float] = [loose_margin]
    for i in range(m):
        margins.append(tight_margin)
        margins.append(tight_margin)
        if i + 1 < m:
            margins.append(loose_margin)
    margins.append(loose_margin)
    return positions, margins


class GcopterGenerator(ITrajectoryGenerator):
    """Offline polynomial planner built once in :meth:`initialize`.

    The L-BFGS optimiser runs once over the full waypoint list. ``evaluate``
    only samples the resulting polynomial at time ``t`` (clamped to the total
    duration). After the polynomial is exhausted the reference holds the last
    point with zero velocity/acceleration for hover.
    """

    def __init__(self, cfg: GcopterConfig) -> None:
        if cfg.corridor_margin <= 0.0:
            raise ValueError('GcopterGenerator: corridor_margin must be > 0.')
        self._cfg = cfg
        self._ctrl: Optional[TrajectoryGenerator] = None
        self._duration = 0.0
        self._waypoints: List[np.ndarray] = []
        self._wp_index = 0
        self._path_facing = False
        self._has_prev_t = False
        self._prev_t = 0.0
        self._yaw_ref_hold = 0.0
        self._name = 'GcopterGenerator'

    @staticmethod
    def load_config_from_yaml(path: str) -> GcopterConfig:
        """Load a :class:`GcopterConfig` from YAML matching the C++ schema."""
        if not os.path.isfile(path):
            raise ValueError(f'Config file not found at {os.path.abspath(path)}.')
        with open(path, 'r') as f:
            root = yaml.safe_load(f)
        if not isinstance(root, dict):
            raise ValueError('gcopter config: root must be a mapping.')
        for key in ('trajectory_generator', 'drone_params', 'drone_limits', 'optimization'):
            if key not in root or not isinstance(root[key], dict):
                raise ValueError(
                    f"gcopter config: '{key}' section must be present and a mapping.")

        tg = root['trajectory_generator']
        dp = root['drone_params']
        dl = root['drone_limits']
        opt = root['optimization']

        cfg = GcopterConfig()
        cfg.corridor_margin = _require(tg, 'corridor_margin', 'trajectory_generator')
        if 'waypoint_margin' in tg:
            cfg.waypoint_margin = float(tg['waypoint_margin'])
        if 'waypoint_anchor_radius' in tg:
            cfg.waypoint_anchor_radius = float(tg['waypoint_anchor_radius'])

        cfg.drone_params.mass = _require(dp, 'mass', 'drone_params')
        cfg.drone_params.gravity = _require(dp, 'gravity', 'drone_params')
        cfg.drone_params.horizontal_drag = _require(dp, 'horizontal_drag', 'drone_params')
        cfg.drone_params.vertical_drag = _require(dp, 'vertical_drag', 'drone_params')
        cfg.drone_params.parasitic_drag = _require(dp, 'parasitic_drag', 'drone_params')
        cfg.drone_params.speed_smooth_factor = _require(
            dp, 'speed_smooth_factor', 'drone_params')

        # drone_limits.max_velocity is filled in initialize() from ExampleConfig.max_speed.
        cfg.drone_limits.max_body_rate = _require(dl, 'max_body_rate', 'drone_limits')
        cfg.drone_limits.max_tilt_angle = _require(dl, 'max_tilt_angle', 'drone_limits')
        cfg.drone_limits.min_thrust = _require(dl, 'min_thrust', 'drone_limits')
        cfg.drone_limits.max_thrust = _require(dl, 'max_thrust', 'drone_limits')

        cfg.optimization.time_weight = _require(opt, 'time_weight', 'optimization')
        cfg.optimization.position_weight = _require(opt, 'position_weight', 'optimization')
        cfg.optimization.velocity_weight = _require(opt, 'velocity_weight', 'optimization')
        cfg.optimization.body_rate_weight = _require(opt, 'body_rate_weight', 'optimization')
        cfg.optimization.tilt_weight = _require(opt, 'tilt_weight', 'optimization')
        cfg.optimization.thrust_weight = _require(opt, 'thrust_weight', 'optimization')
        if 'smoothing_eps' in opt:
            cfg.optimization.smoothing_eps = float(opt['smoothing_eps'])
        if 'integral_resolution' in opt:
            cfg.optimization.integral_resolution = int(opt['integral_resolution'])
        if 'rel_cost_tol' in opt:
            cfg.optimization.rel_cost_tol = float(opt['rel_cost_tol'])
        return cfg

    def _build_native_drone_params(self) -> DroneParameters:
        dp = DroneParameters()
        src = self._cfg.drone_params
        dp.mass = src.mass
        dp.gravity = src.gravity
        dp.horizontal_drag = src.horizontal_drag
        dp.vertical_drag = src.vertical_drag
        dp.parasitic_drag = src.parasitic_drag
        dp.speed_smooth_factor = src.speed_smooth_factor
        return dp

    def _build_native_drone_limits(self) -> DroneLimits:
        dl = DroneLimits()
        src = self._cfg.drone_limits
        dl.max_velocity = src.max_velocity
        dl.max_body_rate = src.max_body_rate
        dl.max_tilt_angle = src.max_tilt_angle
        dl.min_thrust = src.min_thrust
        dl.max_thrust = src.max_thrust
        return dl

    def _build_native_optimization(self) -> OptimizationConfig:
        oc = OptimizationConfig()
        src = self._cfg.optimization
        oc.time_weight = src.time_weight
        oc.position_weight = src.position_weight
        oc.velocity_weight = src.velocity_weight
        oc.body_rate_weight = src.body_rate_weight
        oc.tilt_weight = src.tilt_weight
        oc.thrust_weight = src.thrust_weight
        oc.smoothing_eps = src.smoothing_eps
        oc.integral_resolution = src.integral_resolution
        oc.rel_cost_tol = src.rel_cost_tol
        return oc

    def initialize(self,
                   waypoints: List[np.ndarray],
                   initial_state: State,
                   example_cfg: ExampleConfig) -> None:
        if len(waypoints) < 2:
            raise ValueError('GcopterGenerator: at least two waypoints are required.')
        if example_cfg.max_speed <= 0.0:
            raise ValueError(
                'GcopterGenerator: ExampleConfig.max_speed must be > 0 '
                '(set in config_example.yaml).')
        self._waypoints = [np.asarray(w, dtype=float) for w in waypoints]
        self._wp_index = 0
        self._path_facing = bool(example_cfg.path_facing)
        self._has_prev_t = False
        self._prev_t = 0.0
        self._yaw_ref_hold = _quat_to_yaw(np.asarray(initial_state.orientation, dtype=float))

        # Pull the velocity bound from the shared simulation config.
        self._cfg.drone_limits.max_velocity = float(example_cfg.max_speed)

        # Prepend the initial position unless colocated with the first waypoint.
        p0 = np.asarray(initial_state.position, dtype=float)
        full_path: List[np.ndarray] = []
        if float(np.linalg.norm(self._waypoints[0] - p0)) > 1e-6:
            full_path.append(p0.copy())
        full_path.extend(w.copy() for w in self._waypoints)

        anchors_enabled = (self._cfg.waypoint_margin > 0.0
                           and self._cfg.waypoint_anchor_radius > 0.0)
        expanded_positions, expanded_margins = _expand_path_with_anchors(
            full_path,
            self._cfg.corridor_margin,
            self._cfg.waypoint_margin,
            self._cfg.waypoint_anchor_radius,
        )

        wps = [Waypoint(p) for p in expanded_positions]
        self._ctrl = TrajectoryGenerator(
            self._build_native_drone_params(),
            self._build_native_drone_limits(),
            self._build_native_optimization(),
        )

        if anchors_enabled:
            ok = self._ctrl.generate(wps, expanded_margins)
        else:
            ok = self._ctrl.generate(wps, self._cfg.corridor_margin)
        if not ok:
            raise RuntimeError(
                'GcopterGenerator: L-BFGS optimisation failed '
                '(check waypoints, limits, margins).')
        self._duration = float(self._ctrl.duration())

    def update(self, t: float, state: State) -> None:
        del state  # GCOPTER is a pure feedforward planner
        if self._ctrl is None:
            raise RuntimeError(
                'GcopterGenerator: initialize() must be called before update().')

        dt = max(t - self._prev_t, 0.0) if self._has_prev_t else 0.0
        self._prev_t = t
        self._has_prev_t = True

        t_eval = min(max(t, 0.0), self._duration)
        pos = np.asarray(self._ctrl.position(t_eval), dtype=float)

        # Update "current waypoint" index for logging only.
        n_wps = len(self._waypoints)
        i = self._wp_index
        while i < n_wps - 1:
            if float(np.linalg.norm(pos - self._waypoints[i])) < 0.1:
                self._wp_index = i + 1
                i += 1
            else:
                break

        if self._path_facing:
            vel = np.asarray(self._ctrl.velocity(t_eval), dtype=float)
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
        sample = ReferenceSample()
        if self._ctrl is None:
            return sample
        t_eval = min(max(t, 0.0), self._duration)
        sample.position = np.asarray(self._ctrl.position(t_eval), dtype=float).copy()
        sample.velocity = np.asarray(self._ctrl.velocity(t_eval), dtype=float).copy()
        sample.acceleration = np.asarray(
            self._ctrl.acceleration(t_eval), dtype=float).copy()
        sample.yaw = self._yaw_ref_hold
        if t >= self._duration:
            sample.velocity.fill(0.0)
            sample.acceleration.fill(0.0)
        return sample

    def is_finished(self, t: float) -> bool:
        return t >= self._duration

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
