#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight stdout progress reporter shared by the unified Python runner.

Python mirror of
``examples/framework/include/framework/stdout_progress.hpp``.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

import sys


def print_rule(ch: str = '=', width: int = 72) -> None:
    print('\n' + ch * width)


def print_case_banner(
    index: int,
    total: int,
    controller: str,
    generator: str,
    run_id: str,
) -> None:
    print(f'\n[{index + 1:>2}/{total}] {controller} × {generator}  '
          f'(run_id={run_id})', flush=True)


def print_status(
    t: float,
    t_total: float,
    waypoint_index: int,
    n_waypoints: int,
    last_err_m: float,
    ctrl_time_us: float,
) -> None:
    progress = max(0.0, min(1.0, t / t_total)) if t_total > 0.0 else 0.0
    bar_width = 30
    pos = int(bar_width * progress)
    bar = '['
    for i in range(bar_width):
        if i < pos:
            bar += '='
        elif i == pos:
            bar += '>'
        else:
            bar += ' '
    bar += ']'
    line = (f'  {bar}  t={t:.2f}/{t_total:.2f}s  '
            f'wp={waypoint_index + 1}/{n_waypoints}  '
            f'err={last_err_m:.3f}m  ctrl={ctrl_time_us:.0f}µs      ')
    sys.stdout.write('\r' + line)
    sys.stdout.flush()


def print_case_summary(
    wall_time_s: float,
    rmse_m: float,
    ctrl_mean_us: float,
    gen_mean_us: float,
) -> None:
    print(f'\n  done in {wall_time_s:.2f}s wall · rmse={rmse_m:.3f}m'
          f' · ctrl_mean={ctrl_mean_us:.0f}µs'
          f' · gen_mean={gen_mean_us:.0f}µs', flush=True)
