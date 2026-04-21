#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Shared command-line plumbing for the 12 Python example mains.

Each ``run_example.py`` under ``examples/examples/<combo>/`` consumes the
helpers defined here so the pipeline stays identical to its C++ counterpart
and changes propagate to all combinations with a single edit.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class CommonArgs:
    """Parsed default set of command-line arguments for a full example."""

    example_config_path: str
    simulator_config_path: str
    controller_config_path: str
    trajectory_config_path: str
    output_file: str


def parse_full_args(
    prog: str,
    default_controller_cfg: str,
    default_trajectory_cfg: str,
    default_output_file: str,
    default_example_cfg: str = 'configs/simulation/config_example.yaml',
    default_simulator_cfg: str = 'configs/simulation/config_simulator.yaml',
    argv: Optional[list] = None,
) -> CommonArgs:
    """Parse the standard ``-c/-s/-k/-t/-f`` layout used by 11 of the 12 examples.

    ``pid_waypoints`` is the sole exception: it shares a single YAML for both
    controller and trajectory generator, so it calls :func:`parse_shared_args`
    instead.
    """
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument('-c', '--example_config', default=default_example_cfg,
                        help=f'example YAML (default: {default_example_cfg})')
    parser.add_argument('-s', '--simulator_config', default=default_simulator_cfg,
                        help=f'simulator YAML (default: {default_simulator_cfg})')
    parser.add_argument('-k', '--controller_config', default=default_controller_cfg,
                        help=f'controller YAML (default: {default_controller_cfg})')
    parser.add_argument('-t', '--trajectory_config', default=default_trajectory_cfg,
                        help=f'trajectory YAML (default: {default_trajectory_cfg})')
    parser.add_argument('-f', '--output_file', default=default_output_file,
                        help=f'output CSV (default: {default_output_file})')
    ns = parser.parse_args(argv)
    return CommonArgs(
        example_config_path=ns.example_config,
        simulator_config_path=ns.simulator_config,
        controller_config_path=ns.controller_config,
        trajectory_config_path=ns.trajectory_config,
        output_file=ns.output_file,
    )


@dataclass
class SharedCfgArgs:
    """Parsed CLI for the ``pid_waypoints`` case (single shared YAML)."""

    example_config_path: str
    simulator_config_path: str
    shared_config_path: str
    output_file: str


def parse_shared_args(
    prog: str,
    default_shared_cfg: str,
    default_output_file: str,
    default_example_cfg: str = 'configs/simulation/config_example.yaml',
    default_simulator_cfg: str = 'configs/simulation/config_simulator.yaml',
    argv: Optional[list] = None,
) -> SharedCfgArgs:
    """Parse a ``-p`` shared-config CLI (used by ``pid_waypoints``)."""
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument('-c', '--example_config', default=default_example_cfg)
    parser.add_argument('-s', '--simulator_config', default=default_simulator_cfg)
    parser.add_argument('-p', '--pid_config', default=default_shared_cfg)
    parser.add_argument('-f', '--output_file', default=default_output_file)
    ns = parser.parse_args(argv)
    return SharedCfgArgs(
        example_config_path=ns.example_config,
        simulator_config_path=ns.simulator_config,
        shared_config_path=ns.pid_config,
        output_file=ns.output_file,
    )


def ensure_examples_import_path() -> None:
    """Add ``examples/examples/`` to ``sys.path`` so ``_common`` imports work.

    Each ``run_example.py`` calls this at the top of ``__main__``; the helper
    is idempotent. Only needed when a main is invoked directly from disk
    rather than via a build-side installed entry point.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    examples_root = os.path.dirname(here)
    if examples_root not in sys.path:
        sys.path.insert(0, examples_root)
