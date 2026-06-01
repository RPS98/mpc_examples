#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Acados Steady-State-Aware Position-MPC adapter.

SSA-PMPC carries an extra ``artificial_position`` state (free decision
variable, zero dynamics) driven towards the desired set-point by the SSA
offset cost. The adapter feeds the goal as a constant set-point across the
horizon; the admissible (speed-limited) approach emerges from the SSA cost
balance, without a hand-rolled progressive carrot.

Python mirror of
``examples_cpp/src/controllers/ssa_position_mpc_controller.cpp``.
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
from ssa_position_mpc_acados import MPC
from ssa_position_mpc_acados.utils.mpc_yaml import configure_mpc_from_yaml

from examples_py.framework import (
    ControlCommand,
    ExampleConfig,
    IController,
    ReferenceField,
    ReferenceSample,
    make_mask,
)


@dataclass
class SsaPositionMpcConfig:
    """Parsed configuration for :class:`SsaPositionMpcController`."""

    mpc_yaml_path: str = ''
    max_vel_percentage: float = 1.0


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


# Local speed-bound helpers (kept here rather than imported from
# examples_py.controllers.mpc_speed_utils because that module is bound to
# the position-MPC's solver type; the helpers below talk to our SSA solver).

def _read_uh_default(mpc, who: str) -> float:
    bounds = mpc.get_nonlinear_constraint_bounds()
    if bounds is None:
        return 0.0
    uh = bounds.get_uh()
    if uh is None or len(uh) == 0:
        return 0.0
    if float(uh[0]) <= 0.0:
        raise ValueError(
            f"{who}: constraints.uh[0] must be > 0 in the YAML (it encodes "
            f"max_speed²). Got uh={uh[0]}.")
    return float(uh[0])


def _derive_v_ref(uh_default: float, max_vel_percentage: float, who: str) -> float:
    if max_vel_percentage <= 0.0 or max_vel_percentage > 1.0:
        raise ValueError(f'{who}: max_vel_percentage must be in (0, 1].')
    return math.sqrt(uh_default) * max_vel_percentage


def _update_speed_constraint(mpc, v_ref: float) -> None:
    bounds = mpc.get_nonlinear_constraint_bounds()
    if bounds is None:
        return
    uh = bounds.get_uh()
    if uh is None or len(uh) == 0:
        return
    new_uh = np.array(uh, dtype=float)
    new_uh[0] = v_ref * v_ref
    bounds.set_uh(new_uh)
    mpc.update_nonlinear_constraint_bounds()


class SsaPositionMpcController(IController):
    """IController adapter around the SSA Position MPC.

    Horizon size: 1 (the generator only needs to provide the instantaneous
    set-point; the SSA artificial reference handles horizon progression).
    Required reference fields: ``POSITION``.
    """

    def __init__(self, cfg: SsaPositionMpcConfig) -> None:
        if not cfg.mpc_yaml_path:
            raise ValueError('SsaPositionMpcController: mpc_yaml_path must be provided.')
        if cfg.max_vel_percentage <= 0.0 or cfg.max_vel_percentage > 1.0:
            raise ValueError(
                'SsaPositionMpcController: max_vel_percentage must be in (0, 1].')
        self._cfg = cfg
        self._mpc: Optional[MPC] = None
        self._control_period = 0.01
        self._v_ref = 1.0
        self._last_solve_us = 0.0
        self._last_desired_velocity = np.zeros(3, dtype=float)
        self._name = 'SsaPositionMpcController'

    @staticmethod
    def load_config_from_yaml(path: str) -> SsaPositionMpcConfig:
        if not os.path.isfile(path):
            raise ValueError(f'Config file not found at {os.path.abspath(path)}.')
        with open(path, 'r') as f:
            root = yaml.safe_load(f)
        cfg = SsaPositionMpcConfig()
        cfg.mpc_yaml_path = path
        mpc_node = root.get('mpc') if isinstance(root, dict) else None
        if isinstance(mpc_node, dict) and 'max_vel_percentage' in mpc_node:
            cfg.max_vel_percentage = float(mpc_node['max_vel_percentage'])
        return cfg

    def initialize(self, initial_state: State, example_cfg: ExampleConfig) -> None:
        if example_cfg.mpc_dt <= 0.0:
            raise ValueError('SsaPositionMpcController: example_cfg.mpc_dt must be > 0.')

        ocp_json_file = _read_ocp_json_file(self._cfg.mpc_yaml_path)
        self._mpc = MPC(ocp_json_file)
        configure_mpc_from_yaml(self._mpc, self._cfg.mpc_yaml_path)

        # Cap the solver's runtime soft v² bound by sqrt(uh) * max_vel_percentage.
        # The trajectory shape is governed by the SSA artificial reference.
        uh_default = _read_uh_default(self._mpc, 'SsaPositionMpcController')
        self._v_ref = _derive_v_ref(
            uh_default, self._cfg.max_vel_percentage, 'SsaPositionMpcController')
        _update_speed_constraint(self._mpc, self._v_ref)

        self._control_period = example_cfg.mpc_dt

    def reference_horizon_size(self) -> int:
        return 1

    def reference_horizon_dt(self) -> float:
        return self._control_period

    def control_period(self) -> float:
        return self._control_period

    def _set_setpoint_reference(
        self,
        mpc_data,
        goal_position: np.ndarray,
        desired_orientation: np.ndarray,
    ) -> None:
        """Feed the goal as a constant set-point across the horizon (SSA)."""
        params = mpc_data.parameters
        params.set_desired_position(goal_position)
        params.set_desired_orientation(desired_orientation)

    def compute_command(
        self,
        state: State,
        references: List[ReferenceSample],
    ) -> ControlCommand:
        if self._mpc is None:
            raise RuntimeError(
                'SsaPositionMpcController: initialize() must be called before use.')
        if not references:
            raise ValueError('SsaPositionMpcController: references must not be empty.')

        ref = references[0]
        position = np.asarray(state.position, dtype=float)
        orientation = np.asarray(state.orientation, dtype=float)
        velocity = np.asarray(state.linear_velocity, dtype=float)

        mpc_data = self._mpc.get_data()
        mpc_data.state.position = position
        mpc_data.state.orientation = orientation
        mpc_data.state.linear_velocity = velocity

        desired_orientation = _yaw_to_quat(float(ref.yaw))
        self._set_setpoint_reference(
            mpc_data, np.asarray(ref.position, dtype=float), desired_orientation)

        t0 = time.perf_counter()
        status = int(self._mpc.solve())
        t1 = time.perf_counter()
        self._last_solve_us = (t1 - t0) * 1e6

        if status != 0:
            raise RuntimeError(
                f'SsaPositionMpcController: solver returned status {status}')

        # Stage-1 predicted velocity. State layout: position(0..2) + quaternion(3..6)
        # + linear_velocity(7..9) + artificial_position(10..12).
        stage1 = self._mpc.acados_ocp_solver.get(1, 'x')
        self._last_desired_velocity = np.asarray(stage1[7:10], dtype=float).copy()

        thrust = float(mpc_data.actuation.thrust)
        rates = np.asarray(mpc_data.actuation.angular_velocity, dtype=float)
        return ControlCommand(thrust_n=thrust, angular_rate=rates)

    def required_reference_fields(self) -> int:
        return make_mask([ReferenceField.POSITION])

    def name(self) -> str:
        return self._name

    def last_solve_time_micros(self) -> float:
        return self._last_solve_us

    def last_desired_velocity(self) -> np.ndarray:
        return self._last_desired_velocity

    def provides_desired_velocity(self) -> bool:
        return True
