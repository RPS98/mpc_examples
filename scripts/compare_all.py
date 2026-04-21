#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
"""Compute aggregate metrics for every ``simulator_logs/<name>_log.csv``
produced by ``scripts/run_all.sh`` and print a side-by-side comparison.

For each CSV found, ``examples/utils/compute_metrics.py`` is invoked in
programmatic mode; the resulting aggregate row is appended to a summary
table. No plots are produced — pair this script with ``plot_results.py``
on the CSVs of interest for figures.

Usage::

    python3 scripts/compare_all.py
    python3 scripts/compare_all.py --filter pid mpc_position
    python3 scripts/compare_all.py --logs-dir simulator_logs --out summary.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UTILS_DIR = REPO_ROOT / "examples" / "utils"

COMBINATIONS = [
    (ctrl, gen)
    for ctrl in ("pid", "mpc_position", "mpc_trajectory")
    for gen in ("waypoints", "jerk_limited", "gcopter", "dynamic")
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", default=str(REPO_ROOT / "simulator_logs"),
                        help="Directory with <name>_log.csv files.")
    parser.add_argument("--filter", nargs="*", default=None,
                        help="Only include runs whose controller or generator "
                             "matches any of the given tokens.")
    parser.add_argument("--out", default=None,
                        help="Optional path to write the aggregate summary CSV.")
    return parser.parse_args()


def match(name: str, tokens: list[str] | None) -> bool:
    if not tokens:
        return True
    return any(token in name for token in tokens)


def compute_metrics(csv_path: Path, label: str) -> dict[str, str] | None:
    """Run compute_metrics.py on ``csv_path`` and read the resulting metrics row."""
    cmd = [
        sys.executable,
        str(UTILS_DIR / "compute_metrics.py"),
        "-f", str(csv_path),
        "--label", label,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO_ROOT)
    except subprocess.CalledProcessError as exc:  # surface metric errors without aborting
        print(f"[warn] metrics failed for {csv_path.name}: {exc.stderr.strip()}",
              file=sys.stderr)
        return None

    metrics_csv = csv_path.with_name(csv_path.stem + "_metrics.csv")
    if not metrics_csv.exists():
        return None
    with metrics_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            return row
    return None


def main() -> int:
    args = parse_args()
    logs_dir = Path(args.logs_dir)
    if not logs_dir.is_dir():
        print(f"error: {logs_dir} does not exist", file=sys.stderr)
        return 2

    rows: list[dict[str, str]] = []
    for ctrl, gen in COMBINATIONS:
        name = f"{ctrl}_{gen}"
        csv_path = logs_dir / f"{name}_log.csv"
        if not csv_path.exists():
            continue
        if not match(name, args.filter):
            continue
        row = compute_metrics(csv_path, label=name)
        if row is None:
            continue
        row["combination"] = name
        rows.append(row)

    if not rows:
        print("No metrics computed — run scripts/run_all.sh first.")
        return 1

    ordered_keys = ["combination"] + [k for k in rows[0].keys() if k != "combination"]

    if args.out:
        out_path = Path(args.out)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ordered_keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Summary written to {out_path}")

    # Pretty-print to stdout
    widths = {k: max(len(k), *(len(str(r.get(k, ""))) for r in rows)) for k in ordered_keys}
    header = "  ".join(k.ljust(widths[k]) for k in ordered_keys)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(row.get(k, "")).ljust(widths[k]) for k in ordered_keys))
    return 0


if __name__ == "__main__":
    sys.exit(main())
