#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""mav_trajectory_generation_lib adapter (polynomial trajectory planner).

Python mirror of
``examples_cpp/src/generators/mav_traj_gen_generator.cpp``.

Wraps :class:`mav_trajectory_generation_py.TrajectoryGenerator`, a ROS-free
facade around the ETH-ASL ``mav_trajectory_generation`` core. Point-to-point
contract: :meth:`on_waypoint_changed` rebuilds the spline through three
waypoints ``[current_pos, midpoint, next_waypoint]`` so the solver always sees
≥ 2 segments. :meth:`evaluate` samples the polynomial using a segment-local
time ``t - t_segment_start``.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml
from mav_trajectory_generation_py import (
    GeneratorConfig,
    OptimizationConfig,
    Solver,
    TrajectoryGenerator,
    Waypoint,
)
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


@dataclass
class _OptimizationSpec:
    derivative_to_optimize: int = 4
    solver: str = 'linear'  # 'linear' or 'nonlinear'
    a_max: float = 4.0
    nl_max_iterations: int = 2000
    nl_f_rel: float = 0.05
    nl_x_rel: float = 0.1
    nl_time_penalty: float = 1000.0
    nl_initial_stepsize_rel: float = 0.1
    nl_inequality_constraint_tolerance: float = 0.2


@dataclass
class MavTrajGenConfig:
    optimization: _OptimizationSpec = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.optimization is None:
            self.optimization = _OptimizationSpec()


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


def _parse_solver(value: str) -> Solver:
    if value == 'linear':
        return Solver.linear
    if value == 'nonlinear':
        return Solver.nonlinear
    raise ValueError(
        f"mav_traj_gen config: optimization.solver must be 'linear' or 'nonlinear' "
        f"(got '{value}').")


class MavTrajGenGenerator(ITrajectoryGenerator):
    """Polynomial trajectory generator with replan-on-waypoint-change semantics."""

    def __init__(self, cfg: MavTrajGenConfig) -> None:
        self._cfg = cfg
        self._ctrl: Optional[TrajectoryGenerator] = None
        self._max_speed = 0.0

        self._target_wp = np.zeros(3, dtype=float)
        self._hold_pos = np.zeros(3, dtype=float)
        self._duration = 0.0
        self._t_segment_start = 0.0
        self._has_plan = False
        self._path_facing = False

        self._yaw_ref_hold = 0.0
        self._prev_t = 0.0
        self._has_prev_t = False
        self._name = 'MavTrajGenGenerator'

    @staticmethod
    def load_config_from_yaml(path: str) -> MavTrajGenConfig:
        if not os.path.isfile(path):
            raise ValueError(f'Config file not found at {os.path.abspath(path)}.')
        with open(path, 'r') as f:
            root = yaml.safe_load(f)
        if not isinstance(root, dict):
            raise ValueError('mav_traj_gen config: root must be a mapping.')
        if 'optimization' not in root or not isinstance(root['optimization'], dict):
            raise ValueError("mav_traj_gen config: 'optimization' section is required.")

        opt = root['optimization']
        spec = _OptimizationSpec()
        if 'derivative_to_optimize' in opt:
            spec.derivative_to_optimize = int(opt['derivative_to_optimize'])
        if 'solver' in opt:
            spec.solver = str(opt['solver'])
        if 'a_max' in opt:
            spec.a_max = float(opt['a_max'])
        if 'nl_max_iterations' in opt:
            spec.nl_max_iterations = int(opt['nl_max_iterations'])
        if 'nl_f_rel' in opt:
            spec.nl_f_rel = float(opt['nl_f_rel'])
        if 'nl_x_rel' in opt:
            spec.nl_x_rel = float(opt['nl_x_rel'])
        if 'nl_time_penalty' in opt:
            spec.nl_time_penalty = float(opt['nl_time_penalty'])
        if 'nl_initial_stepsize_rel' in opt:
            spec.nl_initial_stepsize_rel = float(opt['nl_initial_stepsize_rel'])
        if 'nl_inequality_constraint_tolerance' in opt:
            spec.nl_inequality_constraint_tolerance = float(
                opt['nl_inequality_constraint_tolerance'])

        return MavTrajGenConfig(optimization=spec)

    def _build_native_optimization(self) -> OptimizationConfig:
        oc = OptimizationConfig()
        src = self._cfg.optimization
        oc.derivative_to_optimize = int(src.derivative_to_optimize)
        oc.solver = _parse_solver(src.solver)
        oc.a_max = float(src.a_max)
        oc.nl_max_iterations = int(src.nl_max_iterations)
        oc.nl_f_rel = float(src.nl_f_rel)
        oc.nl_x_rel = float(src.nl_x_rel)
        oc.nl_time_penalty = float(src.nl_time_penalty)
        oc.nl_initial_stepsize_rel = float(src.nl_initial_stepsize_rel)
        oc.nl_inequality_constraint_tolerance = float(
            src.nl_inequality_constraint_tolerance)
        return oc

    def _build_native_generator_config(self) -> GeneratorConfig:
        cfg = GeneratorConfig()
        cfg.optimization = self._build_native_optimization()
        return cfg

    def initialize(self, initial_state: State, example_cfg: ExampleConfig) -> None:
        if example_cfg.max_speed <= 0.0:
            raise ValueError(
                'MavTrajGenGenerator: ExampleConfig.max_speed must be > 0 '
                '(set in config_example.yaml).')
        self._path_facing = bool(example_cfg.path_facing)
        self._max_speed = float(example_cfg.max_speed)
        self._has_prev_t = False
        self._prev_t = 0.0
        self._yaw_ref_hold = _quat_to_yaw(
            np.asarray(initial_state.orientation, dtype=float))

        self._hold_pos = np.asarray(initial_state.position, dtype=float).copy()
        self._target_wp = self._hold_pos.copy()
        self._duration = 0.0
        self._t_segment_start = 0.0
        self._has_plan = False

        self._ctrl = TrajectoryGenerator(self._build_native_generator_config())

    def on_waypoint_changed(
        self, next_waypoint: np.ndarray, state: State, t_start: float,
    ) -> None:
        if self._ctrl is None:
            raise RuntimeError(
                'MavTrajGenGenerator: initialize() must be called before on_waypoint_changed().')

        p0 = np.asarray(state.position, dtype=float)
        self._target_wp = np.asarray(next_waypoint, dtype=float).copy()
        self._hold_pos = self._target_wp.copy()
        self._t_segment_start = float(t_start)

        if float(np.linalg.norm(self._target_wp - p0)) < 1e-6:
            self._has_plan = False
            self._duration = 0.0
            return

        # Insert a midpoint to encourage well-conditioned segment-time
        # allocation. Arrange-from-rest at the start waypoint, same as the
        # rest of p2p adapters: the WaypointScheduler settle margin absorbs
        # any residual v0 at segment boundaries.
        midpoint = 0.5 * (p0 + self._target_wp)
        wps = [Waypoint(p0), Waypoint(midpoint), Waypoint(self._target_wp)]

        ok = self._ctrl.generate(wps, self._max_speed)
        if not ok:
            self._has_plan = False
            self._duration = 0.0
            return
        self._has_plan = True
        self._duration = float(self._ctrl.duration())

    def update(self, t: float, state: State) -> None:
        del state
        dt = max(t - self._prev_t, 0.0) if self._has_prev_t else 0.0
        self._prev_t = t
        self._has_prev_t = True

        if not self._has_plan or self._ctrl is None:
            return
        t_local = min(max(t - self._t_segment_start, 0.0), self._duration)

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
        sample = ReferenceSample()
        sample.yaw = self._yaw_ref_hold

        if not self._has_plan or self._ctrl is None:
            sample.position = self._hold_pos.copy()
            return sample

        t_rel = t - self._t_segment_start
        t_eval = min(max(t_rel, 0.0), self._duration)
        sample.position = np.asarray(self._ctrl.position(t_eval), dtype=float).copy()
        vel = np.asarray(self._ctrl.velocity(t_eval), dtype=float)
        acc = np.asarray(self._ctrl.acceleration(t_eval), dtype=float)
        # Guard against non-finite polynomial evaluation (ill-conditioned segments).
        if not np.all(np.isfinite(vel)):
            vel = np.zeros(3)
        if not np.all(np.isfinite(acc)):
            acc = np.zeros(3)
        sample.velocity = vel.copy()
        sample.acceleration = acc.copy()

        if t_rel >= self._duration:
            sample.velocity.fill(0.0)
            sample.acceleration.fill(0.0)
        return sample

    def provided_reference_fields(self) -> ReferenceField:
        return make_mask([
            ReferenceField.POSITION,
            ReferenceField.VELOCITY,
            ReferenceField.ACCELERATION,
        ])

    def name(self) -> str:
        return self._name
