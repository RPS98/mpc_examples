#!/usr/bin/env python3

# Copyright 2025 Universidad Politecnica de Madrid
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
#    * Neither the name of the Universidad Politecnica de Madrid nor the names
#      of its contributors may be used to endorse or promote products derived
#      from this software without specific prior written permission.
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

import argparse
from dataclasses import dataclass
import os
from typing import Any

import numpy as np
import yaml

from mavpy.simulator import load_simulator_parameters_from_yaml


@dataclass
class ExampleConfig:
    """Example-specific simulation configuration."""

    sim_time: float
    model_dt: float
    controller_dt: float
    mpc_dt: float
    max_speed: float
    hover_time: float
    path_facing: bool
    waypoints: list[np.ndarray]


@dataclass
class MpcRuntimeConfig:
    """Runtime metadata required to construct the MPC object."""

    ocp_json_file_path: str
    soft_speed_margin: float = 1.0


@dataclass
class ExampleArgs:
    """CLI arguments."""

    example_config_path: str = 'config_example.yaml'
    simulator_config_path: str = 'config_simulator.yaml'
    mpc_config_path: str = 'config_mpc.yaml'
    output_file: str = 'mpc_log.csv'


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
    """Validate model_dt | controller_dt | mpc_dt divisibility."""
    if cfg.model_dt <= 0.0:
        raise ValueError('sim_config.model_dt must be greater than zero.')
    if cfg.controller_dt <= 0.0:
        raise ValueError('sim_config.controller_dt must be greater than zero.')
    if cfg.mpc_dt <= 0.0:
        raise ValueError('sim_config.mpc_dt must be greater than zero.')

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

    cfg = ExampleConfig(
        sim_time=_required_float(sim_node, 'sim_time', 'sim_config.sim_time'),
        model_dt=_required_float(sim_node, 'model_dt', 'sim_config.model_dt'),
        controller_dt=_required_float(sim_node, 'controller_dt', 'sim_config.controller_dt'),
        mpc_dt=_required_float(sim_node, 'mpc_dt', 'sim_config.mpc_dt'),
        max_speed=_required_float(sim_node, 'max_speed', 'sim_config.max_speed'),
        hover_time=_required_float(sim_node, 'hover_time', 'sim_config.hover_time'),
        path_facing=path_facing,
        waypoints=waypoints,
    )
    _validate_dt_divisibility(cfg)
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
        output_file=args.output_file)
