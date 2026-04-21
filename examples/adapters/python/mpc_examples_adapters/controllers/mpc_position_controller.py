#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Acados position-MPC adapter wrapping :class:`mpc_acados_position.MPC`.

Python mirror of
``examples/adapters/controllers/mpc_position/src/mpc_position_controller.cpp``.
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
from mpc_acados_position import MPC
from mpc_acados_position.utils.mpc_yaml import configure_mpc_from_yaml

from mpc_examples_framework import (
    ControlCommand,
    ExampleConfig,
    IController,
    ReferenceField,
    ReferenceSample,
    make_mask,
)


@dataclass
class MpcPositionConfig:
    """Parsed configuration for :class:`MpcPositionController`."""

    mpc_yaml_path: str = ''
    soft_speed_margin: float = 1.0


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


class MpcPositionController(IController):
    """IController adapter around the position variant of the acados MPC.

    Horizon size: 1 (the generator only needs to provide the instantaneous goal;
    stage references are built internally from ``max_speed``, ``N`` and
    ``dt_horizon``). Required reference fields: ``POSITION``.
    """

    def __init__(self, cfg: MpcPositionConfig) -> None:
        if not cfg.mpc_yaml_path:
            raise ValueError('MpcPositionController: mpc_yaml_path must be provided.')
        if cfg.soft_speed_margin <= 0.0:
            raise ValueError('MpcPositionController: soft_speed_margin must be > 0.')
        self._cfg = cfg
        self._mpc: Optional[MPC] = None
        self._control_period = 0.01
        self._v_ref = 1.0
        self._dt_horizon = 0.05
        self._horizon_steps = 0
        self._last_solve_us = 0.0
        self._name = 'MpcPositionController'

    @staticmethod
    def load_config_from_yaml(path: str) -> MpcPositionConfig:
        """Load an :class:`MpcPositionConfig` from ``configs/controllers/config_mpc.yaml``-
        like YAML. ``mpc.soft_speed_margin`` is optional (default 1.0).
        """
        if not os.path.isfile(path):
            raise ValueError(f'Config file not found at {os.path.abspath(path)}.')
        with open(path, 'r') as f:
            root = yaml.safe_load(f)
        cfg = MpcPositionConfig()
        cfg.mpc_yaml_path = path
        mpc_node = root.get('mpc') if isinstance(root, dict) else None
        if isinstance(mpc_node, dict) and 'soft_speed_margin' in mpc_node:
            cfg.soft_speed_margin = float(mpc_node['soft_speed_margin'])
        return cfg

    def initialize(self, initial_state: State, example_cfg: ExampleConfig) -> None:
        if example_cfg.mpc_dt <= 0.0:
            raise ValueError('MpcPositionController: example_cfg.mpc_dt must be > 0.')
        if example_cfg.max_speed <= 0.0:
            raise ValueError('MpcPositionController: example_cfg.max_speed must be > 0.')

        ocp_json_file = _read_ocp_json_file(self._cfg.mpc_yaml_path)
        self._mpc = MPC(ocp_json_file)
        configure_mpc_from_yaml(self._mpc, self._cfg.mpc_yaml_path)
        self._apply_soft_speed_constraint(example_cfg.max_speed)

        self._control_period = example_cfg.mpc_dt
        self._v_ref = example_cfg.max_speed
        self._horizon_steps = int(self._mpc.get_prediction_steps())
        self._dt_horizon = float(self._mpc.get_prediction_time_step())

    def _apply_soft_speed_constraint(self, max_speed: float) -> None:
        assert self._mpc is not None
        nb = self._mpc.get_nonlinear_constraint_bounds()
        if int(nb.nh_size) <= 0:
            return
        soft_speed = self._cfg.soft_speed_margin * max_speed
        nb.set_uh(np.array([soft_speed * soft_speed], dtype=float))
        self._mpc.update_nonlinear_constraint_bounds()

    def reference_horizon_size(self) -> int:
        return 1

    def reference_horizon_dt(self) -> float:
        return self._control_period

    def control_period(self) -> float:
        return self._control_period

    def _set_progressive_references(
        self,
        mpc_data,
        current_position: np.ndarray,
        goal_position: np.ndarray,
        desired_orientation: np.ndarray,
    ) -> None:
        delta = goal_position - current_position
        distance = float(np.linalg.norm(delta))
        params = mpc_data.parameters
        if distance < 1e-9:
            params.set_desired_position(goal_position)
        else:
            direction = delta / distance
            for k in range(self._horizon_steps + 1):
                s_k = min((k + 1) * self._v_ref * self._dt_horizon, distance)
                stage_position = current_position + s_k * direction
                params.set_desired_position(stage_position, k)
        params.set_desired_orientation(desired_orientation)

    def compute_command(
        self,
        state: State,
        references: List[ReferenceSample],
    ) -> ControlCommand:
        if self._mpc is None:
            raise RuntimeError(
                'MpcPositionController: initialize() must be called before use.')
        if not references:
            raise ValueError('MpcPositionController: references must not be empty.')

        ref = references[0]
        position = np.asarray(state.position, dtype=float)
        orientation = np.asarray(state.orientation, dtype=float)
        velocity = np.asarray(state.linear_velocity, dtype=float)

        mpc_data = self._mpc.get_data()
        mpc_data.state.position = position
        mpc_data.state.orientation = orientation
        mpc_data.state.linear_velocity = velocity

        desired_orientation = _yaw_to_quat(float(ref.yaw))
        self._set_progressive_references(
            mpc_data, position, np.asarray(ref.position, dtype=float),
            desired_orientation)

        t0 = time.perf_counter()
        status = int(self._mpc.solve())
        t1 = time.perf_counter()
        self._last_solve_us = (t1 - t0) * 1e6

        if status != 0:
            raise RuntimeError(
                f'MpcPositionController: solver returned status {status}')

        thrust = float(mpc_data.actuation.thrust)
        rates = np.asarray(mpc_data.actuation.angular_velocity, dtype=float)
        return ControlCommand(thrust_n=thrust, angular_rate=rates)

    def required_reference_fields(self) -> ReferenceField:
        return make_mask([ReferenceField.POSITION])

    def name(self) -> str:
        return self._name

    def last_solve_time_micros(self) -> float:
        return self._last_solve_us
