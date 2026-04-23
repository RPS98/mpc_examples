# Copyright 2025 mav_trajectory_generation_lib contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Tests for the high-level Python ``Trajectory`` wrapper and YAML loader."""

from pathlib import Path

import numpy as np
import pytest

from mav_trajectory_generation_py import (
    GeneratorConfig,
    OptimizationConfig,
    Solver,
    Trajectory,
    TrajectoryPoint,
    load_generator_config,
)


def _three_waypoints():
    return [
        np.array([0.0, 0.0, 1.0]),
        np.array([5.0, 2.0, 1.5]),
        np.array([8.0, 0.0, 2.0]),
    ]


def test_generator_config_to_native():
    gc = GeneratorConfig(derivative_to_optimize=2, solver="nonlinear", a_max=3.0)
    native = gc.to_native()
    assert isinstance(native, OptimizationConfig)
    assert native.derivative_to_optimize == 2
    assert native.solver == Solver.nonlinear
    assert native.a_max == pytest.approx(3.0)


def test_generator_config_rejects_unknown_solver():
    with pytest.raises(ValueError):
        GeneratorConfig(solver="bogus").to_native()


def test_load_generator_config(tmp_path: Path):
    yaml_text = """
optimization:
  derivative_to_optimize: 3
  solver: nonlinear
  a_max: 5.5
  nl_max_iterations: 777
"""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml_text)
    cfg = load_generator_config(cfg_path)
    assert cfg.derivative_to_optimize == 3
    assert cfg.solver == "nonlinear"
    assert cfg.a_max == pytest.approx(5.5)
    assert cfg.nl_max_iterations == 777


def test_load_generator_config_defaults_on_empty(tmp_path: Path):
    cfg_path = tmp_path / "empty.yaml"
    cfg_path.write_text("optimization: {}\n")
    cfg = load_generator_config(cfg_path)
    default = GeneratorConfig()
    assert cfg.derivative_to_optimize == default.derivative_to_optimize
    assert cfg.a_max == default.a_max


def test_trajectory_generate_and_evaluate():
    traj = Trajectory()
    assert traj.generate(_three_waypoints(), 3.0) is True
    assert traj.is_valid
    assert traj.duration > 0.0

    point = traj.evaluate(0.5 * traj.duration)
    assert isinstance(point, TrajectoryPoint)
    assert point.position.shape == (3,)
    assert point.velocity.shape == (3,)
    assert point.acceleration.shape == (3,)


def test_trajectory_rejects_bad_inputs():
    traj = Trajectory()
    with pytest.raises(ValueError):
        traj.generate(_three_waypoints(), 0.0)
    with pytest.raises(ValueError):
        traj.generate([np.zeros(3)], 3.0)


def test_trajectory_accepts_generator_config():
    cfg = GeneratorConfig(solver="nonlinear", nl_max_iterations=500)
    traj = Trajectory(cfg)
    assert traj.generate(_three_waypoints(), 3.0) is True
    assert traj.is_valid
