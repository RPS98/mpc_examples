#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Direct trajectory PID + geometric attitude controller adapter.

Python mirror of
``examples_cpp/src/controllers/pid_trajectory_geometric_controller.cpp``.
Wraps :class:`mavpy.controllers.pid_controllers.TrajectoryController` and
:class:`mavpy.controllers.geometric_controller.GeometricController`
behind :class:`IController` for trajectory-based references.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import yaml
from mavpy.controllers.geometric_controller import (
    AttitudeGeometricControllerParameters,
    GeometricController,
    GeometricControllerParameters,
    RatesGeometricControllerParameters,
)
from mavpy.controllers.libs.pid_controller import PIDParameters
from mavpy.controllers.pid_controllers import (
    TrajectoryController,
    TrajectoryControllerParameters,
)
from mavpy.model import State

from examples_py.framework import (
    ControlCommand,
    ExampleConfig,
    IController,
    ReferenceField,
    ReferenceSample,
    make_mask,
)


@dataclass
class PidTrajectoryGeometricConfig:
    """Parsed configuration for :class:`PidTrajectoryGeometricController`."""

    trajectory_pid_params: PIDParameters = field(default_factory=PIDParameters)
    attitude_params: AttitudeGeometricControllerParameters = field(
        default_factory=AttitudeGeometricControllerParameters)
    rates_params: RatesGeometricControllerParameters = field(
        default_factory=RatesGeometricControllerParameters)
    v_max: float = 1.0


def _read_vec3(node, name: str) -> np.ndarray:
    if not isinstance(node, (list, tuple)) or len(node) != 3:
        raise ValueError(f'{name} must be a sequence with 3 elements.')
    return np.array([float(x) for x in node], dtype=float)


def _parse_pid_params(node: dict, section: str) -> PIDParameters:
    params = PIDParameters()
    if 'kp' not in node:
        raise ValueError(f'{section}.kp is required.')
    if 'ki' not in node:
        raise ValueError(f'{section}.ki is required.')
    if 'kd' not in node:
        raise ValueError(f'{section}.kd is required.')
    params.Kp_gains = _read_vec3(node['kp'], f'{section}.kp')
    params.Ki_gains = _read_vec3(node['ki'], f'{section}.ki')
    params.Kd_gains = _read_vec3(node['kd'], f'{section}.kd')
    if 'antiwindup_cte' in node:
        value = float(node['antiwindup_cte'])
        params.antiwindup_cte = np.full(3, value, dtype=float)
    if 'alpha' in node:
        value = float(node['alpha'])
        params.alpha = np.full(3, value, dtype=float)
    if 'a_max' in node:
        value = float(node['a_max'])
        if value > 0.0:
            params.upper_output_saturation = np.full(3, value, dtype=float)
            params.lower_output_saturation = np.full(3, -value, dtype=float)
    return params


class PidTrajectoryGeometricController(IController):
    """Direct trajectory PID for smooth trajectory tracking (horizon size 1).

    Pipeline, executed once per control period:
      1. trajectory PID: (state.pos, state.vel, ref.pos, ref.vel) -> acc_des
         (parallel pos/vel feedback into a single PID; no acceleration feedforward)
      2. geometric:      (acc_des, ref.yaw, state.orientation) -> (thrust, rates)

    Requires ReferenceField.POSITION | VELOCITY.
    Suitable for smooth trajectory generators (jerk_limited, gcopter, dynamic,
    mav_traj_gen) and trajectory MPC.
    """

    def __init__(self, cfg: PidTrajectoryGeometricConfig) -> None:
        if cfg.v_max <= 0.0:
            raise ValueError('PidTrajectoryGeometricController: v_max must be > 0.')
        self._cfg = cfg
        self._traj_ctrl: Optional[TrajectoryController] = None
        self._geo_ctrl: Optional[GeometricController] = None
        self._control_period = 0.01
        self._last_solve_us = 0.0
        self._name = 'PidTrajectoryGeometricController'

    @staticmethod
    def load_config_from_yaml(path: str) -> PidTrajectoryGeometricConfig:
        """Load a :class:`PidTrajectoryGeometricConfig` from a YAML file.

        Expected structure (config_pid_trajectory.yaml)::

            v_max: 3.0
            controller:
              trajectory: {kp, ki, kd, antiwindup_cte, alpha}
              geometric:  {mass, rotation_kp}
        """
        if not os.path.isfile(path):
            raise ValueError(f'Config file not found at {os.path.abspath(path)}.')
        with open(path, 'r') as f:
            root = yaml.safe_load(f)
        if not isinstance(root, dict):
            raise ValueError('pid_trajectory_geometric config: root must be a mapping.')

        cfg = PidTrajectoryGeometricConfig()
        if 'v_max' in root:
            cfg.v_max = float(root['v_max'])

        ctrl = root.get('controller')
        if not isinstance(ctrl, dict):
            raise ValueError("pid_trajectory_geometric config: 'controller' must be a mapping.")
        if 'trajectory' not in ctrl:
            raise ValueError("pid_trajectory_geometric config: 'controller.trajectory' is required.")
        cfg.trajectory_pid_params = _parse_pid_params(ctrl['trajectory'], 'controller.trajectory')

        geo = ctrl.get('geometric')
        if not isinstance(geo, dict):
            raise ValueError("pid_trajectory_geometric config: 'controller.geometric' must be a mapping.")
        if 'mass' not in geo:
            raise ValueError("pid_trajectory_geometric config: 'controller.geometric.mass' is required.")
        if 'rotation_kp' not in geo:
            raise ValueError(
                "pid_trajectory_geometric config: 'controller.geometric.rotation_kp' is required.")
        cfg.attitude_params = AttitudeGeometricControllerParameters()
        cfg.attitude_params.vehicle_mass = float(geo['mass'])
        cfg.rates_params = RatesGeometricControllerParameters()
        cfg.rates_params.kp_rotation = _read_vec3(
            geo['rotation_kp'], 'controller.geometric.rotation_kp')
        return cfg

    def initialize(self, initial_state: State, example_cfg: ExampleConfig) -> None:
        self._control_period = example_cfg.pid_dt
        if self._control_period <= 0.0:
            raise ValueError('PidTrajectoryGeometricController: example_cfg.pid_dt must be > 0.')

        traj_params = TrajectoryControllerParameters()
        traj_params.pid_parameters = self._cfg.trajectory_pid_params
        self._traj_ctrl = TrajectoryController(traj_params)

        geo_params = GeometricControllerParameters(
            attitude_parameters=self._cfg.attitude_params,
            rates_parameters=self._cfg.rates_params,
        )
        self._geo_ctrl = GeometricController(geo_params)

    def reference_horizon_size(self) -> int:
        return 1

    def reference_horizon_dt(self) -> float:
        return self._control_period

    def control_period(self) -> float:
        return self._control_period

    def compute_command(
        self,
        state: State,
        references: List[ReferenceSample],
    ) -> ControlCommand:
        if not references:
            raise ValueError('PidTrajectoryGeometricController: references must not be empty.')
        assert self._traj_ctrl is not None
        assert self._geo_ctrl is not None
        ref = references[0]

        position = np.asarray(state.position, dtype=float)
        velocity = np.asarray(state.linear_velocity, dtype=float)
        orientation = np.asarray(state.orientation, dtype=float)

        t0 = time.perf_counter()
        acc_des = self._traj_ctrl.trajectory_to_linear_acceleration(
            position,
            velocity,
            np.asarray(ref.position, dtype=float),
            np.asarray(ref.velocity, dtype=float),
            np.zeros(3),
            self._control_period)

        thrust, rates = self._geo_ctrl.acceleration_to_rates(
            acc_des, float(ref.yaw), orientation)
        t1 = time.perf_counter()
        self._last_solve_us = (t1 - t0) * 1e6

        return ControlCommand(thrust_n=float(thrust),
                              angular_rate=np.asarray(rates, dtype=float))

    def required_reference_fields(self) -> ReferenceField:
        return make_mask([
            ReferenceField.POSITION,
            ReferenceField.VELOCITY,
        ])

    def name(self) -> str:
        return self._name

    def last_solve_time_micros(self) -> float:
        return self._last_solve_us
