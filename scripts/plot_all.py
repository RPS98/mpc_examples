#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
"""Batch figure generation for every ``simulator_logs/*_log.csv`` produced by
``scripts/run_all.sh``.

For each log found the script runs ``examples/utils/compute_metrics.py``
(unless the companion ``*_metrics.csv`` already exists) and then
``examples/utils/plot_results.py`` with ``--save --no-show``. A per-run
output directory is created under ``simulator_logs/plots/<combination>/``.

Optional ``--pairs`` flag accepts ``"a,b;c,d"`` to additionally generate
paired comparison figures (delegated to ``plot_results.py -f -f2``), useful
for PID vs MPC on the same generator or for a C++ binary vs its Python twin.

Usage::

    python3 scripts/plot_all.py
    python3 scripts/plot_all.py --filter pid mpc_position
    python3 scripts/plot_all.py --pairs \\
        "mpc_trajectory_gcopter,pid_gcopter;mpc_position_waypoints,mpc_position_waypoints_py"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UTILS_DIR = REPO_ROOT / "examples" / "utils"
COMPUTE_METRICS = UTILS_DIR / "compute_metrics.py"
PLOT_RESULTS = UTILS_DIR / "plot_results.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", default=str(REPO_ROOT / "simulator_logs"),
                        help="Directory containing <name>_log.csv files.")
    parser.add_argument("--plots-dir", default=None,
                        help="Root directory for generated plots "
                             "(default: <logs-dir>/plots).")
    parser.add_argument("--filter", nargs="*", default=None,
                        help="Only include logs whose filename contains any of "
                             "the given tokens.")
    parser.add_argument("--pairs", default=None,
                        help='Semicolon-separated comparison pairs, each as "a,b" '
                             'where a and b are log stems (without _log.csv).')
    parser.add_argument("--force-metrics", action="store_true",
                        help="Recompute metrics even if *_metrics.csv exists.")
    parser.add_argument("--max-speed", type=float, default=None,
                        help="Upper bound for the speed-violation metric "
                             "(forwarded to plot_results.py).")
    return parser.parse_args()


def match(name: str, tokens: list[str] | None) -> bool:
    if not tokens:
        return True
    return any(token in name for token in tokens)


def ensure_metrics(csv_path: Path, label: str, *, force: bool) -> Path | None:
    """Run ``compute_metrics.py`` on *csv_path* if the metrics file is missing."""
    metrics_csv = csv_path.with_name(csv_path.stem + "_metrics.csv")
    if metrics_csv.exists() and not force:
        return metrics_csv
    cmd = [sys.executable, str(COMPUTE_METRICS), "-f", str(csv_path), "--label", label]
    try:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    except subprocess.CalledProcessError as exc:
        print(f"[warn] compute_metrics failed for {csv_path.name}: {exc}",
              file=sys.stderr)
        return None
    return metrics_csv if metrics_csv.exists() else None


def plot_single(csv_path: Path, plots_root: Path, max_speed: float | None,
                *, force_metrics: bool) -> bool:
    label = csv_path.stem.replace("_log", "")
    metrics_csv = ensure_metrics(csv_path, label, force=force_metrics)
    outdir = plots_root / label
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(PLOT_RESULTS),
        "-f", str(csv_path),
        "--label1", label,
        "--no-show", "--save",
        "--outdir", str(outdir),
    ]
    if metrics_csv is not None:
        cmd.extend(["-m", str(metrics_csv)])
    if max_speed is not None:
        cmd.extend(["--max_speed", str(max_speed)])
    try:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    except subprocess.CalledProcessError as exc:
        print(f"[warn] plot_results failed for {csv_path.name}: {exc}",
              file=sys.stderr)
        return False
    print(f"  -> {outdir}")
    return True


def plot_pair(stem_a: str, stem_b: str, logs_dir: Path, plots_root: Path,
              max_speed: float | None, *, force_metrics: bool) -> bool:
    csv_a = logs_dir / f"{stem_a}_log.csv"
    csv_b = logs_dir / f"{stem_b}_log.csv"
    if not csv_a.exists() or not csv_b.exists():
        print(f"[skip pair] missing: {csv_a if not csv_a.exists() else csv_b}",
              file=sys.stderr)
        return False
    metrics_a = ensure_metrics(csv_a, stem_a, force=force_metrics)
    metrics_b = ensure_metrics(csv_b, stem_b, force=force_metrics)
    outdir = plots_root / f"_pair_{stem_a}__vs__{stem_b}"
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(PLOT_RESULTS),
        "-f", str(csv_a), "--label1", stem_a,
        "-f2", str(csv_b), "--label2", stem_b,
        "--no-show", "--save",
        "--outdir", str(outdir),
    ]
    if metrics_a is not None:
        cmd.extend(["-m", str(metrics_a)])
    if metrics_b is not None:
        cmd.extend(["-m2", str(metrics_b)])
    if max_speed is not None:
        cmd.extend(["--max_speed", str(max_speed)])
    try:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    except subprocess.CalledProcessError as exc:
        print(f"[warn] plot_results pair failed ({stem_a} vs {stem_b}): {exc}",
              file=sys.stderr)
        return False
    print(f"  -> {outdir}")
    return True


def parse_pairs(spec: str | None) -> list[tuple[str, str]]:
    if not spec:
        return []
    pairs: list[tuple[str, str]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, _, b = chunk.partition(",")
        a, b = a.strip(), b.strip()
        if not a or not b:
            raise SystemExit(f"error: invalid --pairs entry '{chunk}' (expected 'a,b')")
        pairs.append((a, b))
    return pairs


def main() -> int:
    args = parse_args()
    logs_dir = Path(args.logs_dir)
    if not logs_dir.is_dir():
        print(f"error: {logs_dir} does not exist", file=sys.stderr)
        return 2

    plots_root = Path(args.plots_dir) if args.plots_dir else logs_dir / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    pairs = parse_pairs(args.pairs)

    csvs = sorted(p for p in logs_dir.glob("*_log.csv"))
    csvs = [p for p in csvs if match(p.name, args.filter)]
    if not csvs and not pairs:
        print("No matching *_log.csv under "
              f"{logs_dir} — run scripts/run_all.sh first.", file=sys.stderr)
        return 1

    ok_single = 0
    for csv_path in csvs:
        print(f"[plot] {csv_path.name}")
        if plot_single(csv_path, plots_root, args.max_speed,
                       force_metrics=args.force_metrics):
            ok_single += 1

    ok_pairs = 0
    for stem_a, stem_b in pairs:
        print(f"[pair] {stem_a} vs {stem_b}")
        if plot_pair(stem_a, stem_b, logs_dir, plots_root, args.max_speed,
                     force_metrics=args.force_metrics):
            ok_pairs += 1

    print()
    print(f"Done. {ok_single}/{len(csvs)} per-run figure sets; "
          f"{ok_pairs}/{len(pairs)} comparison figure sets.")
    print(f"Output: {plots_root}")
    return 0 if (ok_single == len(csvs) and ok_pairs == len(pairs)) else 1


if __name__ == "__main__":
    sys.exit(main())
