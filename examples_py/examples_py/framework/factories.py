#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""String-keyed factories that build IController / ITrajectoryGenerator.

Python mirror of
``examples/framework/include/framework/factories.hpp`` and
``examples/framework/src/factories.cpp``.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

from .controller_base import IController
from .trajectory_generator_base import ITrajectoryGenerator


# Canonical keys (kept in sync with the C++ factories) ------------------------

class ControllerKeys:
    PID = 'pid'
    MPC_POSITION = 'mpc_position'
    MPC_TRAJECTORY = 'mpc_trajectory'
    MPCC = 'mpcc'


class GeneratorKeys:
    WAYPOINTS = 'waypoints'
    JERK_LIMITED = 'jerk_limited'
    GCOPTER = 'gcopter'
    DYNAMIC = 'dynamic'
    MAV_TRAJ_GEN = 'mav_traj_gen'
    CIRCUIT = 'circuit'


def default_controller_config_path(name: str) -> str:
    if name == ControllerKeys.PID:
        return 'configs/controllers/config_pid.yaml'
    if name == ControllerKeys.MPC_POSITION:
        return 'configs/controllers/config_mpc.yaml'
    if name == ControllerKeys.MPC_TRAJECTORY:
        return 'configs/controllers/config_mpc_trajectory.yaml'
    if name == ControllerKeys.MPCC:
        return 'configs/controllers/config_mpcc.yaml'
    raise ValueError(f"Unknown controller name: '{name}'.")


def default_generator_config_path(name: str) -> str:
    if name == GeneratorKeys.WAYPOINTS:
        return 'configs/generators/config_waypoints.yaml'
    if name == GeneratorKeys.JERK_LIMITED:
        return 'configs/generators/config_jerk_limited.yaml'
    if name == GeneratorKeys.GCOPTER:
        return 'configs/generators/config_gcopter.yaml'
    if name == GeneratorKeys.DYNAMIC:
        return 'configs/generators/config_dynamic.yaml'
    if name == GeneratorKeys.MAV_TRAJ_GEN:
        return 'configs/generators/config_mav_traj_gen.yaml'
    if name == GeneratorKeys.CIRCUIT:
        return 'configs/generators/config_circuit.yaml'
    raise ValueError(f"Unknown generator name: '{name}'.")


def make_controller(name: str, config_path: str = '', is_trajectory_scope: bool = False) -> IController:
    from examples_py.controllers.pid_position_geometric_controller import (
        PidPositionGeometricController,
    )
    from examples_py.controllers.pid_trajectory_geometric_controller import (
        PidTrajectoryGeometricController,
    )
    from examples_py.controllers.mpc_position_controller import (
        MpcPositionController,
    )
    from examples_py.controllers.mpc_trajectory_controller import (
        MpcTrajectoryController,
    )

    path = config_path or default_controller_config_path(name)
    if name == ControllerKeys.PID:
        if is_trajectory_scope:
            cfg = PidTrajectoryGeometricController.load_config_from_yaml(path)
            return PidTrajectoryGeometricController(cfg)
        else:
            cfg = PidPositionGeometricController.load_config_from_yaml(path)
            return PidPositionGeometricController(cfg)
    if name == ControllerKeys.MPC_POSITION:
        cfg = MpcPositionController.load_config_from_yaml(path)
        return MpcPositionController(cfg)
    if name == ControllerKeys.MPC_TRAJECTORY:
        cfg = MpcTrajectoryController.load_config_from_yaml(path)
        return MpcTrajectoryController(cfg)
    if name == ControllerKeys.MPCC:
        from examples_py.controllers.mpcc_controller import MpccController
        cfg = MpccController.load_config_from_yaml(path)
        return MpccController(cfg)
    raise ValueError(f"Unknown controller name: '{name}'.")


def make_generator(name: str, config_path: str = '') -> ITrajectoryGenerator:
    from examples_py.generators.waypoint_reference_generator import (
        WaypointReferenceGenerator,
    )
    from examples_py.generators.jerk_limited_generator import (
        JerkLimitedGenerator,
    )
    from examples_py.generators.gcopter_generator import (
        GcopterGenerator,
    )
    from examples_py.generators.dynamic_trajectory_generator import (
        DynamicTrajectoryGenerator,
    )
    from examples_py.generators.mav_traj_gen_generator import (
        MavTrajGenGenerator,
    )

    path = config_path or default_generator_config_path(name)
    if name == GeneratorKeys.WAYPOINTS:
        cfg = WaypointReferenceGenerator.load_config_from_yaml(path)
        return WaypointReferenceGenerator(cfg)
    if name == GeneratorKeys.JERK_LIMITED:
        cfg = JerkLimitedGenerator.load_config_from_yaml(path)
        return JerkLimitedGenerator(cfg)
    if name == GeneratorKeys.GCOPTER:
        cfg = GcopterGenerator.load_config_from_yaml(path)
        return GcopterGenerator(cfg)
    if name == GeneratorKeys.DYNAMIC:
        cfg = DynamicTrajectoryGenerator.load_config_from_yaml(path)
        return DynamicTrajectoryGenerator(cfg)
    if name == GeneratorKeys.MAV_TRAJ_GEN:
        cfg = MavTrajGenGenerator.load_config_from_yaml(path)
        return MavTrajGenGenerator(cfg)
    if name == GeneratorKeys.CIRCUIT:
        from examples_py.generators.circuit_generator import CircuitGenerator
        cfg = CircuitGenerator.load_config_from_yaml(path)
        return CircuitGenerator(cfg)
    raise ValueError(f"Unknown generator name: '{name}'.")
