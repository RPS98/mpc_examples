#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Shared runner used by :mod:`examples_py.runs.run_position_examples` and
:mod:`examples_py.runs.run_trajectory_examples`. Reads
``sim_config.runs[]`` from a YAML, filters by generator-type whitelist,
and dispatches each eligible entry through :class:`WaypointsSimulator`.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional

from mavpy.simulator.simulator_yaml import load_simulator_parameters_from_yaml

from examples_py.framework import (
    BenchmarkStats,
    RunMetadata,
    RunSpec,
    WaypointsSimulator,
    load_example_config,
)
from examples_py.framework.example_config import DelayMode
from examples_py.framework.factories import make_controller, make_generator
from examples_py.framework.stdout_progress import (
    print_case_banner,
    print_rule,
)


_DEFAULT_EXAMPLE_CFG = 'configs/simulation/config_example.yaml'
_DEFAULT_SIM_CFG = 'configs/simulation/config_simulator.yaml'


@dataclass
class _CaseResult:
    controller: str = ''
    generator: str = ''
    succeeded: bool = False
    csv_path: str = ''
    stats: BenchmarkStats = field(default_factory=BenchmarkStats)
    error: str = ''


def _parse_args(prog: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument('-c', '--example_config',
                        default=_DEFAULT_EXAMPLE_CFG,
                        help=f'Example YAML (default: {_DEFAULT_EXAMPLE_CFG}).')
    parser.add_argument('-s', '--simulator_config',
                        default=_DEFAULT_SIM_CFG,
                        help=f'Simulator YAML (default: {_DEFAULT_SIM_CFG}).')
    parser.add_argument('--output-dir', '--output_dir',
                        dest='output_dir', default='',
                        help='Output directory (default: simulator_logs/<run_id>).')
    parser.add_argument('--only-controller', dest='only_controller', default='',
                        help='Run only entries matching this controller name.')
    parser.add_argument('--only-generator', dest='only_generator', default='',
                        help='Run only entries matching this generator name.')
    parser.add_argument('--parallel', action='store_true',
                        help='Override sim_config.parallel = true. Launches one '
                             'subprocess per enabled run.')
    return parser.parse_args()


def _make_run_id() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _run_case(spec, example_cfg, simulator_params,
              py_dir: str, run_id: str, index: int, total: int,
              print_banner: bool = True) -> _CaseResult:
    """Execute one (controller, generator) case.

    Set ``print_banner=False`` when the caller (e.g. the parallel dispatcher)
    handles stdout decoration itself.
    """
    result = _CaseResult(controller=spec.controller, generator=spec.generator)
    # TODO: remove once MCAP pipeline validated.
    # csv_name = f'{spec.controller}_{spec.generator}.csv'
    csv_name = f'{spec.controller}_{spec.generator}.mcap'
    result.csv_path = os.path.join(py_dir, csv_name)

    if print_banner:
        print_case_banner(index, total, spec.controller, spec.generator, run_id)
    try:
        # Determine scope. The explicit ``scope`` field on the run entry
        # wins so the same (controller, generator) pair can map to two
        # different controller classes depending on which binary picks
        # the entry (e.g. ``pid + gcopter`` runs the cascade
        # ``PidPositionGeometricController`` in the position scope and
        # the parallel ``PidTrajectoryGeometricController`` in the
        # trajectory scope — see config_example_continuous.yaml).
        # Falls back to the legacy generator-based heuristic when the
        # scope is left unset on the run entry.
        if spec.scope:
            is_trajectory_scope = (spec.scope == 'trajectory')
        else:
            trajectory_generators = {'gcopter', 'jerk_limited', 'dynamic', 'mav_traj_gen'}
            is_trajectory_scope = spec.generator in trajectory_generators
        controller = make_controller(spec.controller, spec.controller_config, is_trajectory_scope)
        generator = make_generator(spec.generator, spec.generator_config)

        metadata = RunMetadata(
            controller_name=spec.controller,
            generator_name=spec.generator,
            run_id=run_id,
            language='py',
        )

        sim = WaypointsSimulator(
            controller=controller,
            traj_gen=generator,
            example_cfg=example_cfg,
            simulator_params=simulator_params,
            output_csv=result.csv_path,
            metadata=metadata,
        )
        sim.run()
        result.stats = sim.stats
        result.succeeded = True
    except Exception as exc:  # noqa: BLE001 — report and continue
        result.error = f'{type(exc).__name__}: {exc}'
        if print_banner:
            sys.stderr.write(f'\n  [FAILED] {result.error}\n')
    return result


def _case_prefix(index: int, total: int, controller: str, generator: str) -> str:
    return f'[{index + 1}/{total} {controller}×{generator}] '


def _run_case_in_subprocess(spec: RunSpec,
                            example_config_path: str,
                            simulator_config_path: str,
                            py_dir: str,
                            run_id: str,
                            index: int,
                            total: int) -> _CaseResult:
    """Worker entry-point invoked by ProcessPoolExecutor.

    Each subprocess re-loads the YAML configs so the only payload pickled
    across the process boundary is the (small, primitive) ``RunSpec`` plus a
    handful of strings — keeping the worker fully self-contained, mirroring
    the C++ "1 std::thread per case" model. The example config is forced to
    ``silent=True`` so per-tick progress bars from each worker do not
    interleave on the parent's terminal.
    """
    prefix = _case_prefix(index, total, spec.controller, spec.generator)
    sys.stdout.write(f'{prefix}start (run_id={run_id})\n')
    sys.stdout.flush()

    example_cfg = load_example_config(example_config_path)
    example_cfg.silent = True
    simulator_params = load_simulator_parameters_from_yaml(simulator_config_path)

    result = _run_case(
        spec=spec,
        example_cfg=example_cfg,
        simulator_params=simulator_params,
        py_dir=py_dir,
        run_id=run_id,
        index=index,
        total=total,
        print_banner=False,
    )

    if result.succeeded:
        gen_us = (result.stats.generator_update_mean_us
                  + result.stats.generator_eval_mean_us)
        sys.stdout.write(
            f'{prefix}done in {result.stats.real_time_s:.2f}s'
            f' · rmse={result.stats.tracking_rmse_m:.3f}m'
            f' · ctrl_mean={result.stats.controller_mean_us:.0f}µs'
            f' · gen_mean={gen_us:.0f}µs\n')
        sys.stdout.flush()
    else:
        sys.stderr.write(f'{prefix}[FAILED] {result.error}\n')
        sys.stderr.flush()
    return result


def _print_summary(label: str, results: List[_CaseResult]) -> None:
    print_rule()
    print(f'Final summary ({len(results)} {label} cases)')
    print_rule('-')
    header = (
        f'{"controller":<20}{"generator":<18}'
        f'{"rmse[m]":>10}{"ctrl_us":>14}{"gen_us":>14}{"real[s]":>10}'
    )
    print(header)
    print_rule('-')
    for r in results:
        if not r.succeeded:
            print(f'{r.controller:<20}{r.generator:<18}  FAILED: {r.error}')
            continue
        gen_us = r.stats.generator_update_mean_us + r.stats.generator_eval_mean_us
        print(
            f'{r.controller:<20}{r.generator:<18}'
            f'{r.stats.tracking_rmse_m:>10.3f}'
            f'{r.stats.controller_mean_us:>14.0f}'
            f'{gen_us:>14.0f}'
            f'{r.stats.real_time_s:>10.2f}'
        )
    print_rule()


def run_with_filter(
    prog: str,
    label: str,
    generator_whitelist: Callable[[RunSpec], bool],
) -> int:
    """Entry-point common to both runner variants.

    Parameters
    ----------
    prog : str
        Program name shown in --help.
    label : str
        Human-readable scope label ('position' or 'trajectory') used in logs.
    generator_whitelist : Callable
        Predicate that returns True for :class:`RunSpec` entries that belong
        to this runner's scope (e.g. ``spec.generator == 'waypoints'``).
    """
    args = _parse_args(prog)

    if not os.path.isfile(args.example_config):
        sys.stderr.write(f"Example config not found: '{args.example_config}'.\n")
        return 1
    if not os.path.isfile(args.simulator_config):
        sys.stderr.write(f"Simulator config not found: '{args.simulator_config}'.\n")
        return 1

    example_cfg = load_example_config(args.example_config)
    simulator_params = load_simulator_parameters_from_yaml(args.simulator_config)

    if not example_cfg.runs:
        sys.stderr.write("No runs defined in 'sim_config.runs' — nothing to do.\n")
        return 1

    def case_selected(spec: RunSpec) -> bool:
        if args.only_controller and spec.controller != args.only_controller:
            return False
        if args.only_generator and spec.generator != args.only_generator:
            return False
        return True

    # When the caller pins both --only-controller and --only-generator (the
    # single-script invocation pattern), the matching catalog entry runs even
    # if it is `enabled: false`. The two `*_config` paths are still taken from
    # the YAML entry — the override only relaxes the enabled gate.
    explicit_combo = bool(args.only_controller) and bool(args.only_generator)

    run_id = _make_run_id()
    out_root = (
        args.output_dir
        if args.output_dir
        else os.path.join('simulator_logs', run_id)
    )
    py_dir = os.path.join(out_root, 'py')
    os.makedirs(py_dir, exist_ok=True)

    parallel = args.parallel or example_cfg.parallel

    suffix = ' · mode=parallel' if parallel else ''
    print(f'{prog} · run_id={run_id} · output_dir={out_root}{suffix}')

    # Single pass: announce skips (mirroring the C++ runners) and collect the
    # in-scope specs preserving YAML order.
    enabled_in_scope: List[RunSpec] = []
    explicit_combo_found = False
    for spec in example_cfg.runs:
        matches_filters = case_selected(spec)
        if explicit_combo and matches_filters:
            explicit_combo_found = True
        if not spec.enabled and not (explicit_combo and matches_filters):
            continue
        if not generator_whitelist(spec):
            print(f'[skipped] {spec.controller} + {spec.generator} '
                  f'(not in {prog} scope)')
            continue
        if not matches_filters:
            print(f'[skipped] {spec.controller} + {spec.generator} '
                  f'(filtered out by --only-*)')
            continue
        enabled_in_scope.append(spec)

    if not enabled_in_scope:
        if explicit_combo and not explicit_combo_found:
            sys.stderr.write(
                f"No entry matching --only-controller='{args.only_controller}' "
                f"--only-generator='{args.only_generator}' found in "
                f"sim_config.runs[]. Add it to "
                f"configs/simulation/config_example.yaml.\n")
        else:
            sys.stderr.write(
                f"No enabled runs match {prog}'s scope — nothing to do.\n")
        return 0

    n = len(enabled_in_scope)
    results: List[Optional[_CaseResult]] = [None] * n
    t_wall_start = time.perf_counter()

    if not parallel:
        for i, spec in enumerate(enabled_in_scope):
            results[i] = _run_case(
                spec=spec,
                example_cfg=example_cfg,
                simulator_params=simulator_params,
                py_dir=py_dir,
                run_id=run_id,
                index=i,
                total=n,
            )
    else:
        n_cores = os.cpu_count() or 1
        if n > n_cores:
            sys.stderr.write(
                f'[warning] {n} parallel runs vs {n_cores} CPU cores — '
                'wall-clock metrics may inflate due to CPU contention.\n')
        if (example_cfg.controller_delay_mode == DelayMode.MEASURED
                or example_cfg.generator_delay_mode == DelayMode.MEASURED):
            sys.stderr.write(
                "[warning] parallel mode + 'measured' delay → CPU contention "
                'contaminates *_compute_time_us logged in CSVs. Use '
                'controller_delay_mode/generator_delay_mode: fixed for '
                'reproducible timing.\n')

        # One subprocess per case. Each worker re-loads the YAMLs and forces
        # silent=True so per-tick progress bars do not interleave on the
        # parent's terminal.
        with ProcessPoolExecutor(max_workers=n) as pool:
            futures = {
                pool.submit(_run_case_in_subprocess, spec,
                            args.example_config, args.simulator_config,
                            py_dir, run_id, i, n): i
                for i, spec in enumerate(enabled_in_scope)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                results[i] = fut.result()

    t_wall_total = time.perf_counter() - t_wall_start

    _print_summary(label, results)
    print(f'Done · run_id={run_id} · output_dir={out_root}'
          f' · wall={t_wall_total:.2f}s')

    # Mirror the C++ runner: per-case failures are reported in the printed
    # summary table; the entry-point itself exits 0 so wrapper scripts can
    # post-process the partial results (metrics, plots) without aborting on
    # ``set -euo pipefail``.
    return 0
