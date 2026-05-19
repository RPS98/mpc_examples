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
from enum import Enum
from typing import List

import numpy as np
import yaml


class DelayMode(Enum):
    """Whether the delay applied to cmd/ref streams is measured or fixed."""

    MEASURED = 'measured'
    FIXED = 'fixed'


@dataclass
class RunSpec:
    """One controller/generator combination to execute."""

    controller: str = ''
    generator: str = ''
    enabled: bool = True
    controller_config: str = ''
    generator_config: str = ''
    # Optional scope hint. ``"position"`` restricts the entry to
    # run_position_examples; ``"trajectory"`` restricts it to
    # run_trajectory_examples. Empty (default) means both binaries honour
    # the legacy generator-based scope check.
    scope: str = ''


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
    settle_margin_s: float = 2.0
    # Effective-speed factor used by WaypointScheduler when budgeting time
    # per segment. The heuristic assumes the drone travels at
    # max_speed * scheduler_speed_factor on average. Bell-shaped (gcopter,
    # mav_traj_gen) and trapezoidal (jerk_limited) generators never
    # sustain max_speed during the whole hop, so a factor < 1.0 buys the
    # scheduler enough time for the drone to settle before the next
    # waypoint switch. Must lie in (0, 1]. Default 1.0 keeps the legacy
    # distance/max_speed heuristic.
    scheduler_speed_factor: float = 1.0
    # Mission scheduling mode. "stepwise" (default) keeps the legacy
    # behaviour: the scheduler waits ``settle_margin_s`` between hops,
    # producing near-zero velocity at every waypoint. "continuous"
    # overrides settle to 0 and speed_factor to 1, chaining the
    # waypoints into a continuous flow (aerostack2 mission_moving_path
    # parity).
    mission_mode: str = 'stepwise'
    # follow_reference emulation parameters (continuous_mode only).
    target_modify_period_s: float = 0.2
    target_modify_threshold_m: float = 1.0
    target_start_delay_s: float = 0.0
    # Park / degenerate-hold threshold (m). Mirrors aerostack2's
    # `kDegenerateDistanceM` constant. When the moving target is closer
    # than this to the vehicle, the local generator is bypassed and a
    # static horizon (target, v=0, a=0, latched yaw) is sent instead.
    degenerate_distance_m: float = 0.05
    # Synthetic takeoff/land waypoints prepended/appended to the
    # mission. Mirrors aerostack2's takeoff_behavior + land_behavior so
    # both backends record an identical 3-phase MCAP. Set
    # takeoff_altitude_m=0 to skip the takeoff, land_at_end=False to
    # skip the landing.
    takeoff_altitude_m: float = 0.0
    land_at_end: bool = False
    path_facing: bool = True
    # Publication frequency of `/drone0/debug/mission/reference/pose`.
    # Matches aerostack2's mission scripts: 10 Hz for the stepwise triangle
    # mission (constant in `mission.py`) and the mission.yaml
    # `execution.broadcaster_rate_hz` for the continuous moving_path
    # mission. Set <= 0 to suppress the topic entirely.
    mission_pose_ref_freq: float = 10.0
    benchmark: bool = False
    silent: bool = False
    parallel: bool = False

    controller_delay_mode: DelayMode = DelayMode.MEASURED
    controller_delay_fixed_s: float = 0.0
    generator_delay_mode: DelayMode = DelayMode.MEASURED
    generator_delay_fixed_s: float = 0.0

    waypoints: List[np.ndarray] = field(default_factory=list)
    runs: List[RunSpec] = field(default_factory=list)


def _read_vec3(node, name: str) -> np.ndarray:
    if not isinstance(node, (list, tuple)) or len(node) != 3:
        raise ValueError(f'{name} must be a sequence with 3 elements.')
    try:
        return np.array([float(x) for x in node], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must contain numeric values.') from exc


def _resolve_waypoint_token(token, name: str, distance: float, height: float) -> float:
    """Resolve a waypoint coordinate that may be a number or D/H token.

    Allowed values: numeric literals (int/float), strings ``D``, ``-D``,
    ``H``, ``-H``. Used by :func:`_read_waypoint` when the
    ``sim_config.evaluate`` block enables symbolic tokens.
    """
    if isinstance(token, (int, float)):
        return float(token)
    if isinstance(token, str):
        s = token.strip()
        sign = -1.0 if s.startswith('-') else 1.0
        if sign < 0:
            s = s[1:].strip()
        if s == 'D':
            return sign * distance
        if s == 'H':
            return sign * height
        try:
            return sign * float(s)
        except ValueError:
            pass
    raise ValueError(
        f'{name} has unsupported token {token!r}; allowed values are D, H, '
        '-D, -H, or a numeric literal.')


def _read_waypoint(node, name: str, distance: float, height: float,
                   takeoff_height: float) -> np.ndarray:
    """Read a 3-element waypoint that may use D/H tokens.

    When ``distance > 0`` and ``height > 0`` the entries are resolved as
    symbolic tokens and the z component receives an offset of
    ``takeoff_height``. Otherwise the legacy numeric-vector behaviour is
    used and ``takeoff_height`` is ignored.
    """
    if not isinstance(node, (list, tuple)) or len(node) != 3:
        raise ValueError(f'{name} must be a sequence with 3 elements.')
    if distance <= 0.0 and height <= 0.0:
        return _read_vec3(node, name)
    x = _resolve_waypoint_token(node[0], f'{name}[x]', distance, height)
    y = _resolve_waypoint_token(node[1], f'{name}[y]', distance, height)
    z = _resolve_waypoint_token(node[2], f'{name}[z]', distance, height)
    return np.array([x, y, takeoff_height + z], dtype=float)


def _read_double_required(node, path: str) -> float:
    if node is None:
        raise ValueError(f'Missing required configuration key: {path}')
    try:
        return float(node)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{path} must be a numeric value.') from exc


def _read_double_optional(node, path: str, default: float) -> float:
    if node is None:
        return default
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


def _read_str_optional(node, path: str, default: str) -> str:
    if node is None:
        return default
    if not isinstance(node, str):
        raise ValueError(f'{path} must be a string value.')
    return node


def _parse_delay_mode(value: str, path: str) -> DelayMode:
    if value == 'measured':
        return DelayMode.MEASURED
    if value == 'fixed':
        return DelayMode.FIXED
    raise ValueError(f"{path} must be 'measured' or 'fixed' (got '{value}').")


def _validate_dt_divisibility(cfg: ExampleConfig) -> None:
    if cfg.sim_time <= 0.0:
        raise ValueError('sim_config.sim_time must be greater than zero.')
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


def _load_runs(runs_node, out: List[RunSpec]) -> None:
    if runs_node is None:
        return
    if not isinstance(runs_node, list):
        raise ValueError('sim_config.runs must be a sequence of run specs.')
    for i, item in enumerate(runs_node):
        prefix = f'sim_config.runs[{i}]'
        if not isinstance(item, dict):
            raise ValueError(f'{prefix} must be a mapping.')
        if 'controller' not in item:
            raise ValueError(f'{prefix}.controller is required.')
        if 'generator' not in item:
            raise ValueError(f'{prefix}.generator is required.')
        scope = _read_str_optional(item.get('scope'), f'{prefix}.scope', '')
        if scope and scope not in ('position', 'trajectory'):
            raise ValueError(
                f"{prefix}.scope must be 'position', 'trajectory' or empty "
                f"(got '{scope}').")
        spec = RunSpec(
            controller=str(item['controller']),
            generator=str(item['generator']),
            enabled=_read_bool_optional(item.get('enabled'), f'{prefix}.enabled', True),
            controller_config=_read_str_optional(
                item.get('controller_config'), f'{prefix}.controller_config', ''),
            generator_config=_read_str_optional(
                item.get('generator_config'), f'{prefix}.generator_config', ''),
            scope=scope,
        )
        out.append(spec)


def load_example_config(path: str) -> ExampleConfig:
    """Load an :class:`ExampleConfig` from a YAML file."""
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
    cfg.settle_margin_s = _read_double_optional(
        sim.get('settle_margin_s'), 'sim_config.settle_margin_s', 2.0)
    cfg.scheduler_speed_factor = _read_double_optional(
        sim.get('scheduler_speed_factor'), 'sim_config.scheduler_speed_factor', 1.0)
    if cfg.scheduler_speed_factor <= 0.0 or cfg.scheduler_speed_factor > 1.0:
        raise ValueError(
            'sim_config.scheduler_speed_factor must lie in (0, 1] '
            f'(got {cfg.scheduler_speed_factor})')
    cfg.mission_mode = _read_str_optional(
        sim.get('mission_mode'), 'sim_config.mission_mode', 'stepwise')
    if cfg.mission_mode not in ('stepwise', 'continuous'):
        raise ValueError(
            "sim_config.mission_mode must be 'stepwise' or 'continuous' "
            f"(got '{cfg.mission_mode}')")
    cfg.target_modify_period_s = _read_double_optional(
        sim.get('target_modify_period_s'),
        'sim_config.target_modify_period_s', 0.2)
    if cfg.target_modify_period_s < 0.0:
        raise ValueError(
            'sim_config.target_modify_period_s must be >= 0 '
            f'(got {cfg.target_modify_period_s})')
    cfg.degenerate_distance_m = _read_double_optional(
        sim.get('degenerate_distance_m'),
        'sim_config.degenerate_distance_m', 0.05)
    cfg.takeoff_altitude_m = _read_double_optional(
        sim.get('takeoff_altitude_m'),
        'sim_config.takeoff_altitude_m', 0.0)
    if cfg.takeoff_altitude_m < 0.0:
        raise ValueError(
            'sim_config.takeoff_altitude_m must be >= 0 '
            f'(got {cfg.takeoff_altitude_m})')
    cfg.land_at_end = _read_bool_optional(
        sim.get('land_at_end'),
        'sim_config.land_at_end', False)
    cfg.target_modify_threshold_m = _read_double_optional(
        sim.get('target_modify_threshold_m'),
        'sim_config.target_modify_threshold_m', 1.0)
    if cfg.target_modify_threshold_m < 0.0:
        raise ValueError(
            'sim_config.target_modify_threshold_m must be >= 0 '
            f'(got {cfg.target_modify_threshold_m})')
    cfg.target_start_delay_s = _read_double_optional(
        sim.get('target_start_delay_s'),
        'sim_config.target_start_delay_s', 0.0)
    if cfg.target_start_delay_s < 0.0:
        raise ValueError(
            'sim_config.target_start_delay_s must be >= 0 '
            f'(got {cfg.target_start_delay_s})')
    cfg.path_facing = _read_bool_required(sim.get('path_facing'), 'sim_config.path_facing')
    cfg.mission_pose_ref_freq = _read_double_optional(
        sim.get('mission_pose_ref_freq'),
        'sim_config.mission_pose_ref_freq', 10.0)
    cfg.benchmark = _read_bool_optional(sim.get('benchmark'), 'sim_config.benchmark', False)
    cfg.silent = _read_bool_optional(sim.get('silent'), 'sim_config.silent', False)
    cfg.parallel = _read_bool_optional(sim.get('parallel'), 'sim_config.parallel', False)

    ctrl_delay_str = _read_str_optional(
        sim.get('controller_delay_mode'), 'sim_config.controller_delay_mode', 'measured')
    cfg.controller_delay_mode = _parse_delay_mode(
        ctrl_delay_str, 'sim_config.controller_delay_mode')
    cfg.controller_delay_fixed_s = _read_double_optional(
        sim.get('controller_delay_fixed_s'),
        'sim_config.controller_delay_fixed_s', 0.0)

    gen_delay_str = _read_str_optional(
        sim.get('generator_delay_mode'), 'sim_config.generator_delay_mode', 'measured')
    cfg.generator_delay_mode = _parse_delay_mode(
        gen_delay_str, 'sim_config.generator_delay_mode')
    cfg.generator_delay_fixed_s = _read_double_optional(
        sim.get('generator_delay_fixed_s'),
        'sim_config.generator_delay_fixed_s', 0.0)

    waypoints = sim.get('waypoints')
    if not isinstance(waypoints, list) or len(waypoints) == 0:
        raise ValueError('sim_config.waypoints must be a non-empty list.')

    # Optional `sim_config.evaluate.{distance,height}` enables D/H tokens
    # and adds `takeoff_height` to the z component.
    evaluate_distance = 0.0
    evaluate_height = 0.0
    takeoff_height = 0.0
    evaluate = sim.get('evaluate')
    if isinstance(evaluate, dict):
        evaluate_distance = _read_double_required(
            evaluate.get('distance'), 'sim_config.evaluate.distance')
        evaluate_height = _read_double_optional(
            evaluate.get('height'), 'sim_config.evaluate.height',
            evaluate_distance)
        takeoff_height = _read_double_optional(
            sim.get('takeoff_height'), 'sim_config.takeoff_height', 0.0)
        if evaluate_distance <= 0.0 or evaluate_height <= 0.0:
            raise ValueError(
                'sim_config.evaluate.distance and evaluate.height must be > 0.')

    cfg.waypoints = [
        _read_waypoint(wp, f'sim_config.waypoints[{i}]',
                       evaluate_distance, evaluate_height, takeoff_height)
        for i, wp in enumerate(waypoints)
    ]

    _load_runs(sim.get('runs'), cfg.runs)

    _validate_dt_divisibility(cfg)
    return cfg


def normalize_output_path(output_path: str) -> str:
    """Resolve plain file names into ``simulator_logs/<name>``."""
    if os.path.dirname(output_path):
        return output_path
    return os.path.join('simulator_logs', output_path)


_ = math
