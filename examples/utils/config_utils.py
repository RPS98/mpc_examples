#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Configuration loading for the integrated MPC + simulator example."""

from __future__ import annotations

__authors__ = 'Rafael Perez-Segui'
__copyright__ = 'Copyright (c) 2025 Universidad Politécnica de Madrid'
__license__ = 'BSD-3-Clause'

import argparse
from dataclasses import dataclass, field
import os
from typing import Any

import numpy as np
import yaml

from mavpy.simulator import load_simulator_parameters_from_yaml

SIMULATOR_LOGS_DIR = 'simulator_logs'


@dataclass
class ExampleConfig:
    """Example-specific simulation configuration."""

    sim_time: float
    model_dt: float
    controller_dt: float
    mpc_dt: float
    pid_dt: float
    max_speed: float
    hover_time: float
    path_facing: bool
    benchmark: bool
    silent: bool
    waypoints: list[np.ndarray]


@dataclass
class MpcRuntimeConfig:
    """Runtime metadata required to construct the MPC object."""

    ocp_json_file_path: str
    soft_speed_margin: float = 1.0


@dataclass
class ExampleArgs:
    """CLI arguments."""

    example_config_path: str = 'configs/simulation/config_example.yaml'
    simulator_config_path: str = 'configs/simulation/config_simulator.yaml'
    mpc_config_path: str = 'configs/controllers/config_mpc.yaml'
    output_file: str = 'simulator_logs/mpc_log.csv'


@dataclass
class PidParameters:
    """PID controller gains and settings."""

    kp: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ki: np.ndarray = field(default_factory=lambda: np.zeros(3))
    kd: np.ndarray = field(default_factory=lambda: np.zeros(3))
    antiwindup_cte: np.ndarray = field(default_factory=lambda: np.zeros(3))
    alpha: np.ndarray = field(default_factory=lambda: np.ones(3))
    saturation_upper: np.ndarray | None = None
    saturation_lower: np.ndarray | None = None


@dataclass
class PidControllerConfig:
    """Configuration for the geometric position controller example."""

    position: PidParameters = field(default_factory=PidParameters)
    velocity: PidParameters = field(default_factory=PidParameters)
    mass: float = 1.0
    rotation_kp: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v_max: float = 1.0
    d_max: float = 2.0


@dataclass
class PidExampleArgs:
    """CLI arguments for the PID geometric position controller example."""

    example_config_path: str = 'configs/simulation/config_example.yaml'
    simulator_config_path: str = 'configs/simulation/config_simulator.yaml'
    pid_config_path: str = 'configs/controllers/config_pid.yaml'
    output_file: str = 'simulator_logs/pid_log.csv'


@dataclass
class PidTrajectoryControllerConfig:
    """Configuration for the PID trajectory controller example.

    The trajectory generator maximum velocity is read from
    :class:`ExampleConfig` (`max_speed`) and is intentionally not duplicated
    here.
    """

    trajectory: PidParameters = field(default_factory=PidParameters)
    mass: float = 1.0
    rotation_kp: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class PidTrajectoryExampleArgs:
    """CLI arguments for the PID trajectory controller example."""

    example_config_path: str = 'configs/simulation/config_example.yaml'
    simulator_config_path: str = 'configs/simulation/config_simulator.yaml'
    pid_trajectory_config_path: str = 'configs/config_pid_trajectory.yaml'
    output_file: str = 'simulator_logs/pid_trajectory_log.csv'


@dataclass
class MpcTrajectoryExampleArgs:
    """CLI arguments for the trajectory-tracking MPC example."""

    example_config_path: str = 'configs/simulation/config_example.yaml'
    simulator_config_path: str = 'configs/simulation/config_simulator.yaml'
    mpc_config_path: str = 'configs/controllers/config_mpc_trajectory.yaml'
    output_file: str = 'simulator_logs/mpc_trajectory_log.csv'


def _normalize_output_path(output_path: str) -> str:
    """Resolve plain file names into simulator_logs/<name>."""
    if os.path.dirname(output_path):
        return output_path
    return os.path.join(SIMULATOR_LOGS_DIR, output_path)


def _load_yaml_file(path: str, *, required: bool = False) -> dict[str, Any]:
    """Load YAML file and validate mapping root."""
    if not os.path.isfile(path):
        absolute = os.path.abspath(path)
        if required:
            raise FileNotFoundError(f'Config file not found: {absolute}.')
        return {}

    with open(path, 'r', encoding='utf-8') as file:
        loaded = yaml.safe_load(file)

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f'Config file {path} must contain a mapping at root.')
    return loaded


def _vector(values: Any, size: int, name: str) -> np.ndarray:
    """Convert array-like to fixed-size vector."""
    data = np.asarray(values, dtype=float)
    if data.shape != (size,):
        raise ValueError(f'{name} must have shape ({size},), got {data.shape}.')
    return data


def _require_mapping(node: Any, name: str) -> dict[str, Any]:
    """Read required mapping node."""
    if not isinstance(node, dict):
        raise ValueError(f'{name} must be a mapping.')
    return node


def _required_value(config: dict[str, Any], key: str, path: str) -> Any:
    """Read required key from mapping."""
    if key not in config:
        raise ValueError(f'Missing required configuration key: {path}')
    return config[key]


def _required_float(config: dict[str, Any], key: str, path: str) -> float:
    """Read required float key."""
    return float(_required_value(config, key, path))


def _validate_dt_divisibility(cfg: ExampleConfig) -> None:
    """Validate model_dt | controller_dt | mpc_dt and pid_dt divisibility."""
    if cfg.model_dt <= 0.0:
        raise ValueError('sim_config.model_dt must be greater than zero.')
    if cfg.controller_dt <= 0.0:
        raise ValueError('sim_config.controller_dt must be greater than zero.')
    if cfg.mpc_dt <= 0.0:
        raise ValueError('sim_config.mpc_dt must be greater than zero.')
    if cfg.pid_dt <= 0.0:
        raise ValueError('sim_config.pid_dt must be greater than zero.')

    def _check_multiple(larger: float, smaller: float,
                        larger_name: str, smaller_name: str) -> None:
        ratio = larger / smaller
        rounded = round(ratio)
        if ratio < 1.0 or abs(ratio - rounded) > 1e-9:
            raise ValueError(
                f'{smaller_name} must divide {larger_name} exactly. '
                f'Received {larger_name}={larger}, {smaller_name}={smaller}.')

    _check_multiple(cfg.controller_dt, cfg.model_dt, 'controller_dt', 'model_dt')
    _check_multiple(cfg.mpc_dt, cfg.controller_dt, 'mpc_dt', 'controller_dt')
    _check_multiple(cfg.pid_dt, cfg.controller_dt, 'pid_dt', 'controller_dt')


def load_example_config(path: str) -> ExampleConfig:
    """Load example configuration (sim_config)."""
    data = _load_yaml_file(path, required=True)
    sim_node = _require_mapping(_required_value(data, 'sim_config', 'sim_config'), 'sim_config')

    waypoints_raw = _required_value(sim_node, 'waypoints', 'sim_config.waypoints')
    if not isinstance(waypoints_raw, list):
        raise ValueError('sim_config.waypoints must be a list.')
    waypoints = [_vector(wp, 3, f'sim_config.waypoints[{index}]')
                 for index, wp in enumerate(waypoints_raw)]
    if not waypoints:
        raise ValueError('sim_config.waypoints must contain at least one waypoint.')

    path_facing = _required_value(sim_node, 'path_facing', 'sim_config.path_facing')
    if not isinstance(path_facing, bool):
        raise ValueError('sim_config.path_facing must be a boolean.')

    benchmark = sim_node.get('benchmark', False)
    if not isinstance(benchmark, bool):
        raise ValueError('sim_config.benchmark must be a boolean.')

    silent = sim_node.get('silent', False)
    if not isinstance(silent, bool):
        raise ValueError('sim_config.silent must be a boolean.')

    cfg = ExampleConfig(
        sim_time=_required_float(sim_node, 'sim_time', 'sim_config.sim_time'),
        model_dt=_required_float(sim_node, 'model_dt', 'sim_config.model_dt'),
        controller_dt=_required_float(sim_node, 'controller_dt', 'sim_config.controller_dt'),
        mpc_dt=_required_float(sim_node, 'mpc_dt', 'sim_config.mpc_dt'),
        pid_dt=_required_float(sim_node, 'pid_dt', 'sim_config.pid_dt'),
        max_speed=_required_float(sim_node, 'max_speed', 'sim_config.max_speed'),
        hover_time=_required_float(sim_node, 'hover_time', 'sim_config.hover_time'),
        path_facing=path_facing,
        benchmark=benchmark,
        silent=silent,
        waypoints=waypoints,
    )
    _validate_dt_divisibility(cfg)
    return cfg


def _parse_pid_parameters(node: dict[str, Any], section: str) -> PidParameters:
    """Parse a PID parameter block from a YAML mapping."""

    def vec3(key: str) -> np.ndarray:
        if key not in node:
            raise ValueError(f'{section}.{key} is required.')
        return _vector(node[key], 3, f'{section}.{key}')

    params = PidParameters(
        kp=vec3('kp'),
        ki=vec3('ki'),
        kd=vec3('kd'),
    )
    if 'antiwindup_cte' in node:
        cte = float(node['antiwindup_cte'])
        params.antiwindup_cte = np.full(3, cte)
    if 'alpha' in node:
        alpha = float(node['alpha'])
        params.alpha = np.full(3, alpha)
    if 'saturation_upper' in node and 'saturation_lower' in node:
        params.saturation_upper = _vector(
            node['saturation_upper'], 3, f'{section}.saturation_upper')
        params.saturation_lower = _vector(
            node['saturation_lower'], 3, f'{section}.saturation_lower')
    return params


def load_pid_controller_config(path: str) -> PidControllerConfig:
    """Load PID and geometric controller configuration from a YAML file.

    Expected YAML structure::

        v_max: 1.0
        d_max: 2.0
        controller:
          position:
            kp: [...]
            ki: [...]
            kd: [...]
            antiwindup_cte: ...
            alpha: ...
          velocity:
            kp: [...]
            ki: [...]
            kd: [...]
            antiwindup_cte: ...
            alpha: ...
            saturation_upper: [...]
            saturation_lower: [...]
          geometric:
            mass: ...
            rotation_kp: [...]

    Args:
        path: Path to the YAML configuration file.

    Returns:
        PidControllerConfig with all controller parameters.
    """
    data = _load_yaml_file(path, required=True)

    cfg = PidControllerConfig()
    cfg.v_max = float(data.get('v_max', cfg.v_max))
    cfg.d_max = float(data.get('d_max', cfg.d_max))

    ctrl = _require_mapping(
        _required_value(data, 'controller', 'controller'), 'controller')

    cfg.position = _parse_pid_parameters(
        _require_mapping(_required_value(ctrl, 'position', 'controller.position'),
                         'controller.position'),
        'controller.position')

    cfg.velocity = _parse_pid_parameters(
        _require_mapping(_required_value(ctrl, 'velocity', 'controller.velocity'),
                         'controller.velocity'),
        'controller.velocity')

    geo = _require_mapping(
        _required_value(ctrl, 'geometric', 'controller.geometric'), 'controller.geometric')
    cfg.mass = float(_required_value(geo, 'mass', 'controller.geometric.mass'))
    cfg.rotation_kp = _vector(
        _required_value(geo, 'rotation_kp', 'controller.geometric.rotation_kp'),
        3, 'controller.geometric.rotation_kp')

    return cfg


def load_pid_trajectory_controller_config(path: str) -> PidTrajectoryControllerConfig:
    """Load PID trajectory controller configuration from a YAML file.

    Expected YAML structure::

        controller:
          trajectory:
            kp: [...]
            ki: [...]
            kd: [...]
            antiwindup_cte: ...
            alpha: ...
          geometric:
            mass: ...
            rotation_kp: [...]

    Args:
        path: Path to the YAML configuration file.

    Returns:
        PidTrajectoryControllerConfig with all controller parameters.
    """
    data = _load_yaml_file(path, required=True)

    cfg = PidTrajectoryControllerConfig()

    ctrl = _require_mapping(
        _required_value(data, 'controller', 'controller'), 'controller')

    cfg.trajectory = _parse_pid_parameters(
        _require_mapping(_required_value(ctrl, 'trajectory', 'controller.trajectory'),
                         'controller.trajectory'),
        'controller.trajectory')

    geo = _require_mapping(
        _required_value(ctrl, 'geometric', 'controller.geometric'), 'controller.geometric')
    cfg.mass = float(_required_value(geo, 'mass', 'controller.geometric.mass'))
    cfg.rotation_kp = _vector(
        _required_value(geo, 'rotation_kp', 'controller.geometric.rotation_kp'),
        3, 'controller.geometric.rotation_kp')

    return cfg


def load_simulator_config(path: str) -> Any:
    """Load simulator parameters through mav_simulator public API."""
    return load_simulator_parameters_from_yaml(path)


def load_mpc_runtime_config(path: str) -> MpcRuntimeConfig:
    """Load MPC runtime metadata and soft speed margin.

    The MPC gains/constraints are configured through
    ``mpc_position.configure_mpc_from_yaml`` using the same YAML file.
    """
    data = _load_yaml_file(path, required=True)

    controller_cfg = _require_mapping(
        _required_value(data, 'controller', 'controller'), 'controller')

    ocp_json_file_path = str(
        _required_value(controller_cfg, 'ocp_json_file_path', 'controller.ocp_json_file_path'))

    soft_speed_margin = 1.0
    mpc_cfg = data.get('mpc')
    if mpc_cfg is not None:
        mpc_cfg = _require_mapping(mpc_cfg, 'mpc')
        if 'soft_speed_margin' in mpc_cfg and mpc_cfg['soft_speed_margin'] is not None:
            soft_speed_margin = float(mpc_cfg['soft_speed_margin'])

    return MpcRuntimeConfig(
        ocp_json_file_path=ocp_json_file_path,
        soft_speed_margin=soft_speed_margin)


def parse_pid_arguments(argv: list[str] | None = None) -> 'PidExampleArgs':
    """Parse CLI arguments for the PID geometric position controller example."""
    parser = argparse.ArgumentParser(
        description='PID geometric position controller + MAV Simulator integrated example')
    parser.add_argument(
        '-c', '--example_config',
        default='configs/simulation/config_example.yaml',
        help='Example config YAML')
    parser.add_argument(
        '-s', '--simulator_config',
        default='configs/simulation/config_simulator.yaml',
        help='Simulator config YAML')
    parser.add_argument(
        '-p', '--pid_config',
        default='configs/controllers/config_pid.yaml',
        help='PID controller config YAML')
    parser.add_argument(
        '-f', '--output_file',
        default=PidExampleArgs.output_file,
        help='Output CSV file')
    args = parser.parse_args(argv)

    return PidExampleArgs(
        example_config_path=args.example_config,
        simulator_config_path=args.simulator_config,
        pid_config_path=args.pid_config,
        output_file=_normalize_output_path(args.output_file))


def parse_pid_trajectory_arguments(
        argv: list[str] | None = None) -> 'PidTrajectoryExampleArgs':
    """Parse CLI arguments for the PID trajectory controller example."""
    parser = argparse.ArgumentParser(
        description='PID trajectory controller + MAV Simulator integrated example')
    parser.add_argument(
        '-c', '--example_config',
        default=PidTrajectoryExampleArgs.example_config_path,
        help='Example config YAML')
    parser.add_argument(
        '-s', '--simulator_config',
        default=PidTrajectoryExampleArgs.simulator_config_path,
        help='Simulator config YAML')
    parser.add_argument(
        '-p', '--pid_trajectory_config',
        default=PidTrajectoryExampleArgs.pid_trajectory_config_path,
        help='PID trajectory controller config YAML')
    parser.add_argument(
        '-f', '--output_file',
        default=PidTrajectoryExampleArgs.output_file,
        help='Output CSV file')
    args = parser.parse_args(argv)

    return PidTrajectoryExampleArgs(
        example_config_path=args.example_config,
        simulator_config_path=args.simulator_config,
        pid_trajectory_config_path=args.pid_trajectory_config,
        output_file=_normalize_output_path(args.output_file))


def parse_mpc_trajectory_arguments(
        argv: list[str] | None = None) -> 'MpcTrajectoryExampleArgs':
    """Parse CLI arguments for the trajectory-tracking MPC example."""
    parser = argparse.ArgumentParser(
        description='Trajectory-tracking MPC + MAV Simulator integrated example')
    parser.add_argument(
        '-c', '--example_config',
        default=MpcTrajectoryExampleArgs.example_config_path,
        help='Example config YAML')
    parser.add_argument(
        '-s', '--simulator_config',
        default=MpcTrajectoryExampleArgs.simulator_config_path,
        help='Simulator config YAML')
    parser.add_argument(
        '-m', '--mpc_config',
        default=MpcTrajectoryExampleArgs.mpc_config_path,
        help='Trajectory-tracking MPC config YAML')
    parser.add_argument(
        '-f', '--output_file',
        default=MpcTrajectoryExampleArgs.output_file,
        help='Output CSV file')
    args = parser.parse_args(argv)

    return MpcTrajectoryExampleArgs(
        example_config_path=args.example_config,
        simulator_config_path=args.simulator_config,
        mpc_config_path=args.mpc_config,
        output_file=_normalize_output_path(args.output_file))


def parse_arguments(argv: list[str] | None = None) -> ExampleArgs:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description='MPC + MAV Simulator integrated example')
    parser.add_argument(
        '-c', '--example_config',
        default=ExampleArgs.example_config_path,
        help='Example config YAML')
    parser.add_argument(
        '-s', '--simulator_config',
        default=ExampleArgs.simulator_config_path,
        help='Simulator config YAML')
    parser.add_argument(
        '-m', '--mpc_config',
        default=ExampleArgs.mpc_config_path,
        help='MPC config YAML')
    parser.add_argument(
        '-f', '--output_file',
        default=ExampleArgs.output_file,
        help='Output CSV file')
    args = parser.parse_args(argv)

    return ExampleArgs(
        example_config_path=args.example_config,
        simulator_config_path=args.simulator_config,
        mpc_config_path=args.mpc_config,
        output_file=_normalize_output_path(args.output_file))
