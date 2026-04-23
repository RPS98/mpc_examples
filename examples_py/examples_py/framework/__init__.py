#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""OOP framework for mpc_examples unified Python examples.

Mirrors the C++ framework (``mpc_examples::framework``) so the same dependency
injection pattern (IController + ITrajectoryGenerator + WaypointsSimulator) is
available in Python without pybind11 — only pure-Python on top of the existing
bindings under ``build/python/``.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

from .types import (
    ControlCommand,
    ReferenceField,
    ReferenceSample,
    has_field,
    make_mask,
)
from .controller_base import IController
from .trajectory_generator_base import ITrajectoryGenerator
from .example_config import (
    DelayMode,
    ExampleConfig,
    RunSpec,
    load_example_config,
    normalize_output_path,
)
from .delay_buffer import DelayBuffer
from .waypoint_scheduler import TickResult, WaypointScheduler
# TODO: remove once MCAP pipeline validated.
# from .unified_csv_logger import LogRow, RunMetadata, UnifiedCsvLogger
from .unified_mcap_logger import LogRow, RunMetadata, UnifiedMcapLogger
from .waypoints_simulator import BenchmarkStats, WaypointsSimulator

__all__ = [
    'BenchmarkStats',
    'ControlCommand',
    'DelayBuffer',
    'DelayMode',
    'ExampleConfig',
    'IController',
    'ITrajectoryGenerator',
    'LogRow',
    'ReferenceField',
    'ReferenceSample',
    'RunMetadata',
    'RunSpec',
    'TickResult',
    'UnifiedMcapLogger',
    'WaypointScheduler',
    'WaypointsSimulator',
    'has_field',
    'load_example_config',
    'make_mask',
    'normalize_output_path',
]
