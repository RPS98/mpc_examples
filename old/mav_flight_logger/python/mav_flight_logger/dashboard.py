#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Comparative dashboard for a full mpc_examples unified run.

Reads ``<run_dir>/{cpp,py}/*.csv`` (plus
``<run_dir>/metrics/summary.csv`` if present) and produces a single
multi-panel figure at ``<run_dir>/plots/dashboard.png``:

1. 3D trajectories overlaid, colour-coded by generator, linestyle per language.
2. Tracking RMSE bar chart grouped by controller × generator.
3. Controller compute-time boxplot per combination.
4. Generator compute-time (update + eval) boxplot per combination.
5. Settling time + jerk energy bar charts.

Usage:
    python3 scripts/dashboard.py --run-dir simulator_logs/<run_id>
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Reuse the log CSV reader from the sibling plot_results module.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from plot_results import read_csv  # noqa: E402


_CONTROLLER_ORDER = ['pid', 'mpc_position', 'mpc_trajectory']
_GENERATOR_ORDER = ['waypoints', 'jerk_limited', 'gcopter', 'dynamic']
_GENERATOR_COLORS = {
    'waypoints':    '#1f77b4',
    'jerk_limited': '#ff7f0e',
    'gcopter':      '#2ca02c',
    'dynamic':      '#d62728',
}
_LANG_LINESTYLE = {'cpp': '-', 'py': '--'}


def _parse_combo_from_stem(stem: str) -> Tuple[str, str]:
    for ctrl in sorted(_CONTROLLER_ORDER, key=len, reverse=True):
        prefix = ctrl + '_'
        if stem.startswith(prefix):
            return ctrl, stem[len(prefix):]
    if '_' in stem:
        head, tail = stem.split('_', 1)
        return head, tail
    return stem, ''


def _combo_key(controller: str, generator: str) -> str:
    return f'{controller}×{generator}'


def _combo_sort_key(label: str) -> Tuple[int, int]:
    ctrl, gen = label.split('×', 1)
    ci = _CONTROLLER_ORDER.index(ctrl) if ctrl in _CONTROLLER_ORDER else 99
    gi = _GENERATOR_ORDER.index(gen) if gen in _GENERATOR_ORDER else 99
    return (ci, gi)


def _load_summary_csv(run_dir: str) -> List[dict]:
    path = os.path.join(run_dir, 'metrics', 'summary.csv')
    if not os.path.isfile(path):
        return []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == '':
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _collect_logs(run_dir: str) -> Dict[Tuple[str, str, str], dict]:
    """Return dict keyed by (lang, controller, generator) → parsed CSV dict."""
    logs: Dict[Tuple[str, str, str], dict] = {}
    for lang in ('cpp', 'py'):
        lang_dir = os.path.join(run_dir, lang)
        if not os.path.isdir(lang_dir):
            continue
        for fname in sorted(os.listdir(lang_dir)):
            if not fname.endswith('.csv') or fname.endswith('_metrics.csv') \
                    or fname.endswith('_segments.csv'):
                continue
            stem = os.path.splitext(fname)[0]
            ctrl, gen = _parse_combo_from_stem(stem)
            data = read_csv(os.path.join(lang_dir, fname))
            if not data or not data.get('time'):
                continue
            logs[(lang, ctrl, gen)] = data
    return logs


def _plot_trajectories_3d(ax, logs: Dict[Tuple[str, str, str], dict]) -> None:
    ax.set_title('3D trajectories (colour=generator, dashed=Python)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    plotted_ref = False
    for (lang, ctrl, gen), data in logs.items():
        color = _GENERATOR_COLORS.get(gen, 'gray')
        ls = _LANG_LINESTYLE.get(lang, '-')
        x = np.asarray(data['x'], dtype=float)
        y = np.asarray(data['y'], dtype=float)
        z = np.asarray(data['z'], dtype=float)
        ax.plot(x, y, z, color=color, linestyle=ls, linewidth=0.8, alpha=0.7,
                label=f'{lang}:{ctrl}×{gen}')
        if not plotted_ref and all(k in data for k in ('x_ref', 'y_ref', 'z_ref')):
            xr = np.asarray(data['x_ref'], dtype=float)
            yr = np.asarray(data['y_ref'], dtype=float)
            zr = np.asarray(data['z_ref'], dtype=float)
            ax.plot(xr, yr, zr, color='black', linestyle=':', linewidth=0.6,
                    alpha=0.5, label='reference')
            plotted_ref = True
    ax.legend(loc='upper left', fontsize=6, ncol=2)


def _bar_grouped(ax, summary: List[dict], column: str, title: str,
                 ylabel: str) -> None:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    by_combo: Dict[str, Dict[str, Optional[float]]] = defaultdict(dict)
    for row in summary:
        combo = _combo_key(row['controller'], row['generator'])
        by_combo[combo][row['language']] = _safe_float(row.get(column))
    labels = sorted(by_combo.keys(), key=_combo_sort_key)
    x = np.arange(len(labels))
    width = 0.4
    cpp_vals = [by_combo[lb].get('cpp') or 0.0 for lb in labels]
    py_vals = [by_combo[lb].get('py') or 0.0 for lb in labels]
    ax.bar(x - width / 2, cpp_vals, width, label='cpp', color='#4c72b0')
    ax.bar(x + width / 2, py_vals, width, label='py', color='#dd8452')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=7)


def _boxplot_column(ax, logs: Dict[Tuple[str, str, str], dict], column: str,
                    title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    series: Dict[str, List[float]] = {}
    for (lang, ctrl, gen), data in logs.items():
        if column not in data:
            continue
        arr = np.asarray(data[column], dtype=float)
        arr = arr[arr > 0.0]
        if len(arr) == 0:
            continue
        key = f'{lang}:{ctrl}×{gen}'
        series[key] = arr.tolist()
    if not series:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes)
        return
    labels = sorted(series.keys(),
                    key=lambda s: _combo_sort_key(s.split(':', 1)[1]))
    values = [series[lb] for lb in labels]
    ax.boxplot(values, labels=labels, showfliers=False)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.grid(axis='y', alpha=0.3)


def _boxplot_generator_sum(ax, logs: Dict[Tuple[str, str, str], dict]) -> None:
    ax.set_title('Generator compute time (update + eval)')
    ax.set_ylabel('time [µs]')
    series: Dict[str, List[float]] = {}
    for (lang, ctrl, gen), data in logs.items():
        upd = np.asarray(data.get('generator_update_time_us', []), dtype=float)
        ev = np.asarray(data.get('generator_eval_time_us', []), dtype=float)
        if len(upd) == 0 and len(ev) == 0:
            continue
        n = max(len(upd), len(ev))
        if len(upd) < n:
            upd = np.pad(upd, (0, n - len(upd)))
        if len(ev) < n:
            ev = np.pad(ev, (0, n - len(ev)))
        total = upd + ev
        total = total[total > 0.0]
        if len(total) == 0:
            continue
        series[f'{lang}:{ctrl}×{gen}'] = total.tolist()
    if not series:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes)
        return
    labels = sorted(series.keys(),
                    key=lambda s: _combo_sort_key(s.split(':', 1)[1]))
    values = [series[lb] for lb in labels]
    ax.boxplot(values, labels=labels, showfliers=False)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.grid(axis='y', alpha=0.3)


def build_dashboard(run_dir: str, output_path: Optional[str] = None,
                    show: bool = False) -> str:
    summary = _load_summary_csv(run_dir)
    logs = _collect_logs(run_dir)
    if not logs:
        raise RuntimeError(
            f'No CSV logs found under {run_dir}/{{cpp,py}}/ — nothing to plot.')

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.35)

    ax_traj = fig.add_subplot(gs[0:2, 0:2], projection='3d')
    _plot_trajectories_3d(ax_traj, logs)

    ax_rmse = fig.add_subplot(gs[0, 2])
    if summary:
        _bar_grouped(ax_rmse, summary, 'tracking_rmse_m',
                     'Tracking RMSE', 'RMSE [m]')
    else:
        ax_rmse.text(0.5, 0.5, 'metrics/summary.csv missing\n(run compute_metrics.py --run-dir first)',
                     ha='center', va='center', transform=ax_rmse.transAxes)

    ax_settle = fig.add_subplot(gs[1, 2])
    if summary:
        _bar_grouped(ax_settle, summary, 'settling_time_s',
                     'Mean settling time', 't [s]')
    else:
        ax_settle.axis('off')

    ax_ctrl = fig.add_subplot(gs[2, 0])
    _boxplot_column(ax_ctrl, logs, 'controller_compute_time_us',
                    'Controller compute time', 'time [µs]')

    ax_gen = fig.add_subplot(gs[2, 1])
    _boxplot_generator_sum(ax_gen, logs)

    ax_jerk = fig.add_subplot(gs[2, 2])
    if summary:
        _bar_grouped(ax_jerk, summary, 'jerk_energy',
                     'Jerk energy', '∫ ||j||² dt')
    else:
        ax_jerk.axis('off')

    fig.suptitle(f'mpc_examples dashboard — {os.path.basename(run_dir.rstrip("/"))}',
                 fontsize=14)

    out_path = output_path or os.path.join(run_dir, 'plots', 'dashboard.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    if not show:
        plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Render a comparative dashboard for a unified run.')
    parser.add_argument('--run-dir', required=True,
                        help='Run output directory (e.g. simulator_logs/<run_id>).')
    parser.add_argument('-o', '--output', default=None,
                        help='Output PNG path (default: <run_dir>/plots/dashboard.png).')
    parser.add_argument('--show', action='store_true',
                        help='Display the dashboard window interactively '
                             '(in addition to saving the PNG).')
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        sys.stderr.write(f"ERROR: run_dir '{args.run_dir}' is not a directory.\n")
        return 1

    out_path = build_dashboard(args.run_dir, args.output, show=args.show)
    print(f'Dashboard → {out_path}')
    if args.show:
        plt.show()
    return 0


if __name__ == '__main__':
    sys.exit(main())
