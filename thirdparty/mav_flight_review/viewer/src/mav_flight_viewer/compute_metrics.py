#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Aggregate an mpc_examples run directory of ``.mcap`` logs into metrics CSVs.

Pipeline (equivalent to the retired ``mav_flight_logger.compute_metrics``):

    <run_dir>/{cpp,py}/<stem>.mcap
        → <run_dir>/{cpp,py}/<stem>_metrics.csv    (1 header + 1 data row)
        → <run_dir>/{cpp,py}/<stem>_segments.csv   (1 header + N rows)
        → <run_dir>/metrics/summary.csv            (all runs, one row each)

The CSV schemas (column order and names) match the legacy pipeline so the
downstream dashboard and any external consumers keep working unchanged.
"""

from __future__ import annotations

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

import argparse
import csv
import dataclasses
import os
import sys
from typing import Optional

import numpy as np

from .flight_frame import FlightFrame, load_flight_frame
from .metrics import METRICS_CFG, MetricsConfig, compute_metrics


# ---------------------------------------------------------------------------
# CSV column definitions (kept in sync with the legacy mav_flight_logger tool).
# ---------------------------------------------------------------------------

AGGREGATE_COLUMNS = [
    'label',
    'settling_threshold_m',
    'effective_max_speed',
    'compute_time_mean_us',
    'compute_time_std_us',
    'compute_time_p95_us',
    'generator_update_mean_us',
    'generator_eval_mean_us',
    'controller_delay_mean_us',
    'generator_delay_mean_us',
    'tracking_rmse_m',
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

SUMMARY_COLUMNS = [
    'language',
    'controller',
    'generator',
    'mcap_path',
] + [c for c in AGGREGATE_COLUMNS if c != 'label']

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
# Helpers
# ---------------------------------------------------------------------------


def _val(v: Optional[float]) -> str:
    """Format a nullable float for CSV (empty string when None)."""
    return '' if v is None else repr(v)


def _metrics_row(metrics: dict, label: str) -> dict:
    return {
        'label': label,
        'settling_threshold_m': _val(metrics.get('_settle_thr_m')),
        'effective_max_speed': _val(metrics.get('_effective_max_speed')),
        'compute_time_mean_us': _val(metrics.get('compute_time_mean_us')),
        'compute_time_std_us': _val(metrics.get('compute_time_std_us')),
        'compute_time_p95_us': _val(metrics.get('compute_time_p95_us')),
        'generator_update_mean_us': _val(metrics.get('generator_update_mean_us')),
        'generator_eval_mean_us': _val(metrics.get('generator_eval_mean_us')),
        'controller_delay_mean_us': _val(metrics.get('controller_delay_mean_us')),
        'generator_delay_mean_us': _val(metrics.get('generator_delay_mean_us')),
        'tracking_rmse_m': _val(metrics.get('tracking_rmse_m')),
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


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _parse_combo_from_stem(stem: str) -> tuple:
    """Recover (controller, generator) from a filename stem like 'pid_waypoints'."""
    known_controllers = ('pid', 'mpc_position', 'mpc_trajectory')
    for ctrl in sorted(known_controllers, key=len, reverse=True):
        prefix = ctrl + '_'
        if stem.startswith(prefix):
            return ctrl, stem[len(prefix):]
    if '_' in stem:
        head, tail = stem.split('_', 1)
        return head, tail
    return stem, ''


def write_metrics_csv(metrics: dict, label: str, path: str) -> None:
    """Write aggregate metrics to a wide-format CSV (1 header + 1 data row)."""
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=AGGREGATE_COLUMNS)
        writer.writeheader()
        writer.writerow(_metrics_row(metrics, label))


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


def _has_signal(frame: FlightFrame) -> bool:
    return 'time' in frame.data and len(frame['time']) > 0


# ---------------------------------------------------------------------------
# Per-run aggregation
# ---------------------------------------------------------------------------


def _aggregate_run_dir(run_dir: str, cfg: MetricsConfig) -> int:
    """Walk ``<run_dir>/{cpp,py}/*.mcap`` and produce ``metrics/summary.csv``.

    Returns the number of MCAPs successfully processed.
    """
    summary_rows = []
    for lang in ('cpp', 'py'):
        lang_dir = os.path.join(run_dir, lang)
        if not os.path.isdir(lang_dir):
            continue
        for fname in sorted(os.listdir(lang_dir)):
            if not fname.endswith('.mcap'):
                continue
            mcap_path = os.path.join(lang_dir, fname)
            stem = _stem(mcap_path)
            controller, generator = _parse_combo_from_stem(stem)
            label = f'{lang}:{controller}×{generator}'
            print(f'Reading: {mcap_path}')
            frame = load_flight_frame(mcap_path)
            if not _has_signal(frame):
                print(f'  WARNING: skipping empty MCAP {mcap_path}')
                continue
            m = compute_metrics(frame.data, cfg)

            # Per-run metrics next to the MCAP (backwards compat with the
            # legacy layout that lived next to each CSV).
            write_metrics_csv(m, label, os.path.join(lang_dir, f'{stem}_metrics.csv'))
            write_segments_csv(m, label, os.path.join(lang_dir, f'{stem}_segments.csv'))

            row = _metrics_row(m, label)
            row_out = {
                'language': lang,
                'controller': controller,
                'generator': generator,
                'mcap_path': os.path.relpath(mcap_path, run_dir),
            }
            for key in SUMMARY_COLUMNS:
                if key in row_out:
                    continue
                row_out[key] = row.get(key, '')
            summary_rows.append(row_out)

    if not summary_rows:
        print('No MCAPs found under run_dir — nothing to summarise.')
        return 0

    metrics_dir = os.path.join(run_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    summary_path = os.path.join(metrics_dir, 'summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f'\n  Summary → {summary_path} ({len(summary_rows)} rows)')
    return len(summary_rows)


def _process_single(mcap_path: str, label: str, cfg: MetricsConfig,
                    outdir: str) -> dict:
    abs_path = os.path.abspath(mcap_path)
    print(f'Reading: {abs_path}')
    frame = load_flight_frame(abs_path)
    if not _has_signal(frame):
        print(f'  ERROR: no data in {abs_path}')
        sys.exit(1)

    m = compute_metrics(frame.data, cfg)
    os.makedirs(outdir, exist_ok=True)
    stem = _stem(mcap_path)
    metrics_path = os.path.join(outdir, f'{stem}_metrics.csv')
    segments_path = os.path.join(outdir, f'{stem}_segments.csv')
    write_metrics_csv(m, label, metrics_path)
    write_segments_csv(m, label, segments_path)
    print(f'  Metrics  → {metrics_path}')
    print(f'  Segments → {segments_path}')
    return m


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compute control metrics from one or more mpc_examples MCAP logs.')
    parser.add_argument('-f', '--file', type=str, default=None,
                        help='Single simulation log MCAP')
    parser.add_argument('--label', type=str, default='',
                        help='Label for the primary dataset (default: filename stem)')
    parser.add_argument('--max_speed', type=float, default=None,
                        help='Override max speed for violation metric (m/s)')
    parser.add_argument('-o', '--outdir', type=str, default=None,
                        help='Output directory for metrics CSVs (default: same dir as input)')
    parser.add_argument('--run-dir', dest='run_dir', type=str, default=None,
                        help='Aggregate every MCAP under <run_dir>/{cpp,py}/ and '
                             'write <run_dir>/metrics/summary.csv.')
    args = parser.parse_args()

    cfg = dataclasses.replace(
        METRICS_CFG,
        max_speed=args.max_speed if args.max_speed is not None else METRICS_CFG.max_speed,
    )

    if args.run_dir is not None:
        if not os.path.isdir(args.run_dir):
            print(f"ERROR: run_dir '{args.run_dir}' is not a directory.")
            sys.exit(1)
        n = _aggregate_run_dir(args.run_dir, cfg)
        sys.exit(0 if n > 0 else 2)

    if args.file is None:
        parser.error('--file or --run-dir is required.')

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.file)) or '.'
    label = args.label or _stem(args.file)
    _process_single(args.file, label, cfg, outdir)


if __name__ == '__main__':
    main()
