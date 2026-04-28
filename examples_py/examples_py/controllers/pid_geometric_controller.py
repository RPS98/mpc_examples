#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Cascaded PID position + velocity + geometric attitude controller adapter.

Python mirror of
``examples/adapters/controllers/pid_geometric/src/pid_geometric_controller.cpp``.
Wraps :class:`mavpy.controllers.pid_controllers.{PositionController,
VelocityController}` and :class:`mavpy.controllers.geometric_controller.
GeometricController` behind :class:`IController`.
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
    PositionController,
    PositionControllerParameters,
    VelocityController,
    VelocityControllerParameters,
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
class PidGeometricConfig:
    """Parsed configuration for :class:`PidGeometricController`."""

    position_pid_params: PIDParameters = field(default_factory=PIDParameters)
    velocity_pid_params: PIDParameters = field(default_factory=PIDParameters)
    attitude_params: AttitudeGeometricControllerParameters = field(
        default_factory=AttitudeGeometricControllerParameters)
    rates_params: RatesGeometricControllerParameters = field(
        default_factory=RatesGeometricControllerParameters)
    v_max: float = 1.0
    feedforward_velocity: bool = False


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
    if 'saturation_upper' in node and 'saturation_lower' in node:
        params.upper_output_saturation = _read_vec3(
            node['saturation_upper'], f'{section}.saturation_upper')
        params.lower_output_saturation = _read_vec3(
            node['saturation_lower'], f'{section}.saturation_lower')
        params.proportional_saturation_flag = True
    return params


def _saturate_velocity(v: np.ndarray, v_max: float) -> np.ndarray:
    speed = float(np.linalg.norm(v))
    if speed > v_max:
        return (v / speed) * v_max
    return v


class PidGeometricController(IController):
    """Outer-loop PID cascade with horizon size 1.

    Pipeline, executed once per control period:
      1. position PID:   (state.position, ref.position) -> vel_des
      2. saturate to ``v_max`` (direction preserved)
      3. velocity PID:   (state.velocity, vel_des) -> acc_des
      4. geometric:      (acc_des, ref.yaw, state.orientation) -> (thrust, rates)
    """

    def __init__(self, cfg: PidGeometricConfig) -> None:
        if cfg.v_max <= 0.0:
            raise ValueError('PidGeometricController: v_max must be > 0.')
        self._cfg = cfg
        self._pos_ctrl: Optional[PositionController] = None
        self._vel_ctrl: Optional[VelocityController] = None
        self._geo_ctrl: Optional[GeometricController] = None
        self._control_period = 0.01
        self._last_solve_us = 0.0
        self._name = 'PidGeometricController'

    @staticmethod
    def load_config_from_yaml(path: str) -> PidGeometricConfig:
        """Load a :class:`PidGeometricConfig` from a YAML file.

        Expected structure (matches legacy ``config_pid.yaml``)::

            v_max: 3.0
            controller:
              position:  {kp, ki, kd, antiwindup_cte, alpha}
              velocity:  {kp, ki, kd, antiwindup_cte, alpha,
                          saturation_upper, saturation_lower}
              geometric: {mass, rotation_kp}
        """
        if not os.path.isfile(path):
            raise ValueError(f'Config file not found at {os.path.abspath(path)}.')
        with open(path, 'r') as f:
            root = yaml.safe_load(f)
        if not isinstance(root, dict):
            raise ValueError('pid_geometric config: root must be a mapping.')

        cfg = PidGeometricConfig()
        if 'v_max' in root:
            cfg.v_max = float(root['v_max'])
        if 'feedforward_velocity' in root:
            cfg.feedforward_velocity = bool(root['feedforward_velocity'])

        ctrl = root.get('controller')
        if not isinstance(ctrl, dict):
            raise ValueError("pid_geometric config: 'controller' must be a mapping.")
        if 'position' not in ctrl:
            raise ValueError("pid_geometric config: 'controller.position' is required.")
        cfg.position_pid_params = _parse_pid_params(ctrl['position'], 'controller.position')
        if 'velocity' not in ctrl:
            raise ValueError("pid_geometric config: 'controller.velocity' is required.")
        cfg.velocity_pid_params = _parse_pid_params(ctrl['velocity'], 'controller.velocity')

        geo = ctrl.get('geometric')
        if not isinstance(geo, dict):
            raise ValueError("pid_geometric config: 'controller.geometric' must be a mapping.")
        if 'mass' not in geo:
            raise ValueError("pid_geometric config: 'controller.geometric.mass' is required.")
        if 'rotation_kp' not in geo:
            raise ValueError(
                "pid_geometric config: 'controller.geometric.rotation_kp' is required.")
        cfg.attitude_params = AttitudeGeometricControllerParameters()
        cfg.attitude_params.vehicle_mass = float(geo['mass'])
        cfg.rates_params = RatesGeometricControllerParameters()
        cfg.rates_params.kp_rotation = _read_vec3(
            geo['rotation_kp'], 'controller.geometric.rotation_kp')
        return cfg

    def initialize(self, initial_state: State, example_cfg: ExampleConfig) -> None:
        self._control_period = example_cfg.pid_dt
        if self._control_period <= 0.0:
            raise ValueError('PidGeometricController: example_cfg.pid_dt must be > 0.')

        pos_params = PositionControllerParameters()
        pos_params.pid_parameters = self._cfg.position_pid_params
        self._pos_ctrl = PositionController(pos_params)

        vel_params = VelocityControllerParameters()
        vel_params.pid_parameters = self._cfg.velocity_pid_params
        self._vel_ctrl = VelocityController(vel_params)

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
            raise ValueError('PidGeometricController: references must not be empty.')
        assert self._pos_ctrl is not None
        assert self._vel_ctrl is not None
        assert self._geo_ctrl is not None
        ref = references[0]

        position = np.asarray(state.position, dtype=float)
        velocity = np.asarray(state.linear_velocity, dtype=float)
        orientation = np.asarray(state.orientation, dtype=float)

        t0 = time.perf_counter()
        vel_des = self._pos_ctrl.position_to_linear_velocity(
            position, np.asarray(ref.position, dtype=float), self._control_period)
        if self._cfg.feedforward_velocity:
            vel_des = vel_des + np.asarray(ref.velocity, dtype=float)
        vel_des = _saturate_velocity(vel_des, self._cfg.v_max)

        acc_des = self._vel_ctrl.linear_velocity_to_linear_acceleration(
            velocity, vel_des, self._control_period)

        thrust, rates = self._geo_ctrl.acceleration_to_rates(
            acc_des, float(ref.yaw), orientation)
        t1 = time.perf_counter()
        self._last_solve_us = (t1 - t0) * 1e6

        return ControlCommand(thrust_n=float(thrust),
                              angular_rate=np.asarray(rates, dtype=float))

    def required_reference_fields(self) -> ReferenceField:
        if self._cfg.feedforward_velocity:
            return make_mask([ReferenceField.POSITION, ReferenceField.VELOCITY])
        return make_mask([ReferenceField.POSITION])

    def name(self) -> str:
        return self._name

    def last_solve_time_micros(self) -> float:
        return self._last_solve_us
