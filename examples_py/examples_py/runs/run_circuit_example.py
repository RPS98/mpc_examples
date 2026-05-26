#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-loop circuit-mission showcase, Python entry point.

Iterates ``sim_config.runs[]`` from
``configs/simulation/config_circuit_example.yaml`` and executes only the
entries whose generator is the closed-loop circuit (``circuit``).
By default the YAML pre-wires the two relevant cases:

    - mpcc           + circuit  (MPCC controller, self-contained spline)
    - mpc_trajectory + circuit  (trajectory MPC, samples from generator)

Any other run defined in the YAML is skipped with a log line.

Extra CLI flags (consumed here, stripped before delegating to
``run_with_filter``):

    --mission-yaml PATH    Override the per-gate mission YAML for the
                           current invocation. Exported as
                           ``CIRCUIT_MISSION_YAML`` so both
                           :class:`CircuitGenerator` and
                           :class:`MpccController` pick it up without
                           needing to edit any of the YAML configs
                           shipped with the repo.
    --gates-yaml   PATH    Same idea for the gate-pose YAML. Exported
                           as ``CIRCUIT_GATES_YAML``.

Use case: the same generic ``configs/missions/demo_*.yaml`` ships with
this repo for standalone runs; downstream consumers (e.g. mpcc_v2's
launchers) inject their own mission via these flags without touching
``mav_examples``.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

import argparse
import os
import sys
from typing import List, Tuple

from examples_py.framework import RunSpec
from examples_py.runs._runner import run_with_filter


def _is_circuit_run(spec: RunSpec) -> bool:
    return spec.generator == 'circuit'


def _extract_mission_overrides(argv: List[str]) -> Tuple[List[str], str, str]:
    """Pull ``--mission-yaml`` / ``--gates-yaml`` out of ``argv``.

    Returns ``(remaining_argv, mission_yaml, gates_yaml)``. Recognises
    both ``--flag value`` and ``--flag=value`` forms. Unknown args are
    left untouched for the inner argparse in ``_runner.py`` to handle.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mission-yaml', '--mission_yaml',
                        dest='mission_yaml', default='')
    parser.add_argument('--gates-yaml', '--gates_yaml',
                        dest='gates_yaml', default='')
    parsed, remaining = parser.parse_known_args(argv)
    return remaining, parsed.mission_yaml, parsed.gates_yaml


def main() -> int:
    remaining, mission_yaml, gates_yaml = _extract_mission_overrides(sys.argv[1:])
    if mission_yaml:
        if not os.path.isfile(mission_yaml):
            sys.stderr.write(
                f"--mission-yaml: file not found at '{mission_yaml}'.\n")
            return 1
        os.environ['CIRCUIT_MISSION_YAML'] = os.path.abspath(mission_yaml)
    if gates_yaml:
        if not os.path.isfile(gates_yaml):
            sys.stderr.write(
                f"--gates-yaml: file not found at '{gates_yaml}'.\n")
            return 1
        os.environ['CIRCUIT_GATES_YAML'] = os.path.abspath(gates_yaml)
    sys.argv = [sys.argv[0]] + remaining
    return run_with_filter(
        prog='run_circuit_example.py',
        label='circuit',
        generator_whitelist=_is_circuit_run,
    )


if __name__ == '__main__':
    sys.exit(main())
