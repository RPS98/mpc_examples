#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Common YAML utilities shared by every integrated-framework Python example.

Mirrors the C++ ``ExampleConfig`` / ``loadExampleConfig`` from
``examples/utils/example_config_utils.hpp``.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

import math
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import yaml


@dataclass
class ExampleConfig:
    """Scenario-level configuration shared by all examples."""

    sim_time: float = 0.0
    model_dt: float = 0.0
    controller_dt: float = 0.0
    mpc_dt: float = 0.0
    pid_dt: float = 0.0
    max_speed: float = 0.0
    hover_time: float = 0.0
    path_facing: bool = True
    benchmark: bool = False
    silent: bool = False
    waypoints: List[np.ndarray] = field(default_factory=list)


def _read_vec3(node, name: str) -> np.ndarray:
    if not isinstance(node, (list, tuple)) or len(node) != 3:
        raise ValueError(f'{name} must be a sequence with 3 elements.')
    try:
        return np.array([float(x) for x in node], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must contain numeric values.') from exc


def _read_double_required(node, path: str) -> float:
    if node is None:
        raise ValueError(f'Missing required configuration key: {path}')
    try:
        return float(node)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{path} must be a numeric value.') from exc


def _read_bool_required(node, path: str) -> bool:
    if node is None:
        raise ValueError(f'Missing required configuration key: {path}')
    if not isinstance(node, bool):
        raise ValueError(f'{path} must be a boolean value.')
    return node


def _read_bool_optional(node, path: str, default: bool) -> bool:
    if node is None:
        return default
    if not isinstance(node, bool):
        raise ValueError(f'{path} must be a boolean value.')
    return node


def _validate_dt_divisibility(cfg: ExampleConfig) -> None:
    if cfg.model_dt <= 0.0:
        raise ValueError('sim_config.model_dt must be greater than zero.')
    if cfg.controller_dt <= 0.0:
        raise ValueError('sim_config.controller_dt must be greater than zero.')
    if cfg.mpc_dt <= 0.0:
        raise ValueError('sim_config.mpc_dt must be greater than zero.')
    if cfg.pid_dt <= 0.0:
        raise ValueError('sim_config.pid_dt must be greater than zero.')

    ratio = cfg.controller_dt / cfg.model_dt
    if ratio < 1.0 or abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(
            'sim_config.model_dt must divide sim_config.controller_dt exactly.')

    ratio = cfg.mpc_dt / cfg.controller_dt
    if ratio < 1.0 or abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(
            'sim_config.controller_dt must divide sim_config.mpc_dt exactly.')

    ratio = cfg.pid_dt / cfg.controller_dt
    if ratio < 1.0 or abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(
            'sim_config.controller_dt must divide sim_config.pid_dt exactly.')


def load_example_config(path: str) -> ExampleConfig:
    """Load an :class:`ExampleConfig` from a YAML file.

    Expected structure::

        sim_config:
          sim_time: 36.0
          model_dt: 0.001
          controller_dt: 0.002
          mpc_dt: 0.01
          pid_dt: 0.01
          max_speed: 3.0
          hover_time: 5.0
          path_facing: true
          benchmark: false       # optional
          silent: false          # optional
          waypoints:
            - [0.0, 0.0, 10.0]
            - [10.0, 0.0, 10.0]
    """
    if not os.path.isfile(path):
        raise ValueError(f'Config file not found at {os.path.abspath(path)}.')

    with open(path, 'r') as f:
        try:
            root = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse YAML file '{path}': {exc}") from exc

    if not isinstance(root, dict):
        raise ValueError('Root YAML node must be a mapping.')
    sim = root.get('sim_config')
    if not isinstance(sim, dict):
        raise ValueError('sim_config must be a mapping.')

    cfg = ExampleConfig()
    cfg.sim_time = _read_double_required(sim.get('sim_time'), 'sim_config.sim_time')
    cfg.model_dt = _read_double_required(sim.get('model_dt'), 'sim_config.model_dt')
    cfg.controller_dt = _read_double_required(
        sim.get('controller_dt'), 'sim_config.controller_dt')
    cfg.mpc_dt = _read_double_required(sim.get('mpc_dt'), 'sim_config.mpc_dt')
    cfg.pid_dt = _read_double_required(sim.get('pid_dt'), 'sim_config.pid_dt')
    cfg.max_speed = _read_double_required(sim.get('max_speed'), 'sim_config.max_speed')
    cfg.hover_time = _read_double_required(sim.get('hover_time'), 'sim_config.hover_time')
    cfg.path_facing = _read_bool_required(sim.get('path_facing'), 'sim_config.path_facing')
    cfg.benchmark = _read_bool_optional(sim.get('benchmark'), 'sim_config.benchmark', False)
    cfg.silent = _read_bool_optional(sim.get('silent'), 'sim_config.silent', False)

    waypoints = sim.get('waypoints')
    if not isinstance(waypoints, list) or len(waypoints) == 0:
        raise ValueError('sim_config.waypoints must be a non-empty list.')
    cfg.waypoints = [
        _read_vec3(wp, f'sim_config.waypoints[{i}]') for i, wp in enumerate(waypoints)
    ]

    _validate_dt_divisibility(cfg)
    return cfg


def normalize_output_path(output_path: str) -> str:
    """Resolve plain file names into ``simulator_logs/<name>``."""
    if os.path.dirname(output_path):
        return output_path
    return os.path.join('simulator_logs', output_path)


# Avoid flagging ``math`` as unused when type checkers reimport the module.
_ = math
