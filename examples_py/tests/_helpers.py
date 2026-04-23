#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for the :mod:`examples_py` adapter unit tests.

The helpers intentionally load the **real** YAMLs from
``configs/{controllers,generators,simulation}/`` (not dedicated fixtures),
so any regression in the production configs also fails the tests.
"""

import os
from pathlib import Path
from typing import Iterable

import numpy as np
from mavpy.model import State

from examples_py.framework import (
    DelayMode,
    ExampleConfig,
    ReferenceSample,
    load_example_config,
)


# ``MPC_EXAMPLES_REPO_ROOT`` is injected by CMake's register_pytest_suite()
# helper so the tests can be run from any working directory.
REPO_ROOT = Path(os.environ.get(
    'MPC_EXAMPLES_REPO_ROOT',
    Path(__file__).resolve().parents[2],
))


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def load_test_sim_config() -> ExampleConfig:
    """Load the production sim config and trim the run duration for tests."""
    cfg = load_example_config(str(repo_path('configs/simulation/config_example.yaml')))
    cfg.sim_time = 0.5
    cfg.silent = True
    cfg.benchmark = False
    cfg.controller_delay_mode = DelayMode.FIXED
    cfg.controller_delay_fixed_s = 0.0
    cfg.generator_delay_mode = DelayMode.FIXED
    cfg.generator_delay_fixed_s = 0.0
    return cfg


def hover_state_at(position: Iterable[float]) -> State:
    s = State()
    s.position = np.asarray(position, dtype=float).tolist()
    s.orientation = [1.0, 0.0, 0.0, 0.0]
    s.linear_velocity = [0.0, 0.0, 0.0]
    return s


def horizon_at_position(target: Iterable[float], n: int) -> list:
    samples = []
    pos = np.asarray(target, dtype=float)
    for _ in range(n):
        s = ReferenceSample()
        s.position = pos.copy()
        s.velocity = np.zeros(3)
        s.acceleration = np.zeros(3)
        s.yaw = 0.0
        samples.append(s)
    return samples
