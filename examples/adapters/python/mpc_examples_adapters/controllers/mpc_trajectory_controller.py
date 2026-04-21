#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Acados trajectory-MPC adapter wrapping :class:`mpc_acados_trajectory.MPC`.

Python mirror of
``examples/adapters/controllers/mpc_trajectory/src/mpc_trajectory_controller.cpp``.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import yaml
from mavpy.model import State
from mpc_acados_trajectory import MPC
from mpc_acados_trajectory.utils.mpc_yaml import configure_mpc_from_yaml

from mpc_examples_framework import (
    ControlCommand,
    ExampleConfig,
    IController,
    ReferenceField,
    ReferenceSample,
    make_mask,
)


@dataclass
class MpcTrajectoryConfig:
    """Parsed configuration for :class:`MpcTrajectoryController`."""

    mpc_yaml_path: str = ''


def _yaw_to_quat(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=float)


def _read_ocp_json_file(path: str) -> str:
    if not os.path.isfile(path):
        raise ValueError(f'Config file not found at {os.path.abspath(path)}.')
    with open(path, 'r') as f:
        root = yaml.safe_load(f)
    if not isinstance(root, dict):
        raise ValueError('mpc config: root must be a mapping.')
    controller = root.get('controller')
    if not isinstance(controller, dict):
        raise ValueError("mpc config: 'controller' must be a mapping.")
    ocp_path = controller.get('ocp_json_file_path')
    if not isinstance(ocp_path, str) or not ocp_path:
        raise ValueError("mpc config: 'controller.ocp_json_file_path' is required.")
    return ocp_path


class MpcTrajectoryController(IController):
    """IController adapter around the trajectory variant of the acados MPC.

    Horizon size: ``N_horizon + 1``. The generator is expected to supply
    position + velocity + acceleration at every stage; the adapter forwards
    each stage reference to the solver verbatim.
    """

    def __init__(self, cfg: MpcTrajectoryConfig) -> None:
        if not cfg.mpc_yaml_path:
            raise ValueError('MpcTrajectoryController: mpc_yaml_path must be provided.')
        self._cfg = cfg
        self._mpc: Optional[MPC] = None
        self._control_period = 0.01
        self._dt_horizon = 0.05
        self._horizon_steps = 0
        self._last_solve_us = 0.0
        self._name = 'MpcTrajectoryController'

    @staticmethod
    def load_config_from_yaml(path: str) -> MpcTrajectoryConfig:
        """Validate the YAML file is readable and store its path.

        The full configuration is applied later by
        :func:`configure_mpc_from_yaml` during :meth:`initialize`.
        """
        if not os.path.isfile(path):
            raise ValueError(f'Config file not found at {os.path.abspath(path)}.')
        with open(path, 'r') as f:
            yaml.safe_load(f)
        return MpcTrajectoryConfig(mpc_yaml_path=path)

    def initialize(self, initial_state: State, example_cfg: ExampleConfig) -> None:
        if example_cfg.mpc_dt <= 0.0:
            raise ValueError('MpcTrajectoryController: example_cfg.mpc_dt must be > 0.')

        ocp_json_file = _read_ocp_json_file(self._cfg.mpc_yaml_path)
        self._mpc = MPC(ocp_json_file)
        configure_mpc_from_yaml(self._mpc, self._cfg.mpc_yaml_path)

        self._control_period = example_cfg.mpc_dt
        self._horizon_steps = int(self._mpc.get_prediction_steps())
        self._dt_horizon = float(self._mpc.get_prediction_time_step())

    def reference_horizon_size(self) -> int:
        return self._horizon_steps + 1

    def reference_horizon_dt(self) -> float:
        return self._dt_horizon

    def control_period(self) -> float:
        return self._control_period

    def compute_command(
        self,
        state: State,
        references: List[ReferenceSample],
    ) -> ControlCommand:
        if self._mpc is None:
            raise RuntimeError(
                'MpcTrajectoryController: initialize() must be called before use.')
        expected = self._horizon_steps + 1
        if len(references) != expected:
            raise ValueError('MpcTrajectoryController: references size mismatch.')

        position = np.asarray(state.position, dtype=float)
        orientation = np.asarray(state.orientation, dtype=float)
        velocity = np.asarray(state.linear_velocity, dtype=float)

        mpc_data = self._mpc.get_data()
        mpc_data.state.position = position
        mpc_data.state.orientation = orientation
        mpc_data.state.linear_velocity = velocity

        desired_quat = _yaw_to_quat(float(references[0].yaw))
        params = mpc_data.parameters

        for k in range(self._horizon_steps + 1):
            r = references[k]
            params.set_desired_position(np.asarray(r.position, dtype=float), k)
            params.set_desired_velocity(np.asarray(r.velocity, dtype=float), k)
            params.set_desired_acceleration(np.asarray(r.acceleration, dtype=float), k)
            params.set_desired_orientation(desired_quat, k)

        t0 = time.perf_counter()
        status = int(self._mpc.solve())
        t1 = time.perf_counter()
        self._last_solve_us = (t1 - t0) * 1e6

        if status != 0:
            raise RuntimeError(
                f'MpcTrajectoryController: solver returned status {status}')

        thrust = float(mpc_data.actuation.thrust)
        rates = np.asarray(mpc_data.actuation.angular_velocity, dtype=float)
        return ControlCommand(thrust_n=thrust, angular_rate=rates)

    def required_reference_fields(self) -> ReferenceField:
        return make_mask([
            ReferenceField.POSITION,
            ReferenceField.VELOCITY,
            ReferenceField.ACCELERATION,
        ])

    def name(self) -> str:
        return self._name

    def last_solve_time_micros(self) -> float:
        return self._last_solve_us
