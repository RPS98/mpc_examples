#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
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
#    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
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

"""Compute and persist control performance metrics from a simulation log CSV.

Pipeline:
    simulation → log CSV → compute_metrics.py → metrics CSV + segments CSV

The metrics engine is imported from plot_results.py (same directory).

Usage (single file):
    python3 compute_metrics.py -f mpc_log.csv --label MPC

Usage (two files with comparison table):
    python3 compute_metrics.py -f mpc_log.csv --label MPC \\
                                -f2 pid_log.csv --label2 PID

Output files (written next to each input CSV by default):
    {stem}_metrics.csv   — aggregate scalar metrics (1 header + 1 data row)
    {stem}_segments.csv  — per-segment metrics (1 header + N rows)
"""

__authors__ = 'Rafael Perez-Segui'
__copyright__ = 'Copyright (c) 2025 Universidad Politécnica de Madrid'
__license__ = 'BSD-3-Clause'

import argparse
import csv
import dataclasses
import os
import sys
from typing import Optional

import numpy as np

# Import metrics engine from plot_results (same package directory).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_results import (  # noqa: E402
    METRICS_CFG, MetricsConfig,
    compute_metrics, print_metrics, print_metrics_comparison, read_csv,
)

# ---------------------------------------------------------------------------
# CSV column definitions
# ---------------------------------------------------------------------------

AGGREGATE_COLUMNS = [
    'label',
    'settling_threshold_m',
    'effective_max_speed',
    'compute_time_mean_us',
    'compute_time_std_us',
    'settling_time_s',
    'settling_time_max_s',
    'steady_state_error_m',
    'overshoot_m',
    'rise_time_mean_s',
    'rise_time_min_s',
    'rise_time_max_s',
    'jerk_energy',
    'accel_energy',
    'speed_violation_pct',
]

SEGMENT_COLUMNS = [
    'label',
    'segment_idx',
    't_start',
    't_end',
    't_settle',
    'rise_time',
    'settling_time',
    'steady_state_error',
    'overshoot',
    'd0',
    'settle_thr',
    'ref_x',
    'ref_y',
    'ref_z',
]

# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def _val(v: Optional[float]) -> str:
    """Format a nullable float for CSV (empty string when None)."""
    return '' if v is None else repr(v)


def write_metrics_csv(metrics: dict, label: str, path: str) -> None:
    """Write aggregate metrics to a wide-format CSV (1 header + 1 data row)."""
    row = {
        'label': label,
        'settling_threshold_m': _val(metrics.get('_settle_thr_m')),
        'effective_max_speed': _val(metrics.get('_effective_max_speed')),
        'compute_time_mean_us': _val(metrics.get('compute_time_mean_us')),
        'compute_time_std_us': _val(metrics.get('compute_time_std_us')),
        'settling_time_s': _val(metrics.get('settling_time_s')),
        'settling_time_max_s': _val(metrics.get('settling_time_max_s')),
        'steady_state_error_m': _val(metrics.get('steady_state_error_m')),
        'overshoot_m': _val(metrics.get('overshoot_m')),
        'rise_time_mean_s': _val(metrics.get('rise_time_mean_s')),
        'rise_time_min_s': _val(metrics.get('rise_time_min_s')),
        'rise_time_max_s': _val(metrics.get('rise_time_max_s')),
        'jerk_energy': _val(metrics.get('jerk_energy')),
        'accel_energy': _val(metrics.get('accel_energy')),
        'speed_violation_pct': _val(metrics.get('speed_violation_pct')),
    }
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=AGGREGATE_COLUMNS)
        writer.writeheader()
        writer.writerow(row)


def write_segments_csv(metrics: dict, label: str, path: str) -> None:
    """Write per-segment metrics to a wide-format CSV (1 header + N rows)."""
    segs = metrics.get('seg_metrics', [])
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SEGMENT_COLUMNS)
        writer.writeheader()
        for i, sm in enumerate(segs):
            ref = sm.get('ref_pos', np.zeros(3))
            row = {
                'label': label,
                'segment_idx': i,
                't_start': _val(sm.get('t_start')),
                't_end': _val(sm.get('t_end')),
                't_settle': _val(sm.get('t_settle')),
                'rise_time': _val(sm.get('rise_time')),
                'settling_time': _val(sm.get('settling_time')),
                'steady_state_error': _val(sm.get('steady_state_error')),
                'overshoot': _val(sm.get('overshoot')),
                'd0': _val(sm.get('d0')),
                'settle_thr': _val(sm.get('settle_thr')),
                'ref_x': repr(float(ref[0])),
                'ref_y': repr(float(ref[1])),
                'ref_z': repr(float(ref[2])),
            }
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _process_file(file_path: str, label: str, cfg: MetricsConfig,
                  outdir: str) -> dict:
    """Read log CSV, compute metrics, print, and write output CSVs."""
    abs_path = os.path.abspath(file_path)
    print(f'Reading: {abs_path}')
    data = read_csv(abs_path)
    if not data or not data.get('time'):
        print(f'  ERROR: no data in {abs_path}')
        sys.exit(1)

    m = compute_metrics(data, cfg)
    print_metrics(m, label=label, cfg=cfg)

    os.makedirs(outdir, exist_ok=True)
    stem = _stem(file_path)
    metrics_path = os.path.join(outdir, f'{stem}_metrics.csv')
    segments_path = os.path.join(outdir, f'{stem}_segments.csv')
    write_metrics_csv(m, label, metrics_path)
    write_segments_csv(m, label, segments_path)
    print(f'\n  Metrics  → {metrics_path}')
    print(f'  Segments → {segments_path}')
    return m


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compute control metrics from a simulation log CSV')
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='Primary simulation log CSV')
    parser.add_argument('--label', type=str, default='',
                        help='Label for the primary dataset (default: filename stem)')
    parser.add_argument('--max_speed', type=float, default=None,
                        help='Override max speed for violation metric (m/s)')
    parser.add_argument('-o', '--outdir', type=str, default=None,
                        help='Output directory for metrics CSVs (default: same dir as input)')
    parser.add_argument('-f2', '--file2', type=str, default=None,
                        help='Second simulation log CSV (enables comparison table)')
    parser.add_argument('--label2', type=str, default='',
                        help='Label for the second dataset')
    args = parser.parse_args()

    cfg = dataclasses.replace(
        METRICS_CFG,
        max_speed=args.max_speed if args.max_speed is not None else METRICS_CFG.max_speed,
    )

    outdir1 = args.outdir or os.path.dirname(os.path.abspath(args.file)) or '.'
    label1 = args.label or _stem(args.file)
    m1 = _process_file(args.file, label1, cfg, outdir1)

    if args.file2 is not None:
        outdir2 = args.outdir or os.path.dirname(os.path.abspath(args.file2)) or '.'
        label2 = args.label2 or _stem(args.file2)
        m2 = _process_file(args.file2, label2, cfg, outdir2)
        print_metrics_comparison(m1, m2, label1=label1, label2=label2, cfg=cfg)


if __name__ == '__main__':
    main()
