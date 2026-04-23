#!/usr/bin/env python3
# Copyright 2025 mav_trajectory_generation_lib contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Plot one or several trajectories logged by run_example.cpp / run_example.py.

Reads CSVs with columns:
    t, x, y, z, vx, vy, vz, ax, ay, az

Two modes:
    * Single-trajectory (default): ``-f <csv>``. Optionally overlay
      waypoints from the example YAML via ``-c <cfg>``. Use
      ``--trajectory-index N`` to pick which entry of ``trajectories:`` is
      used for waypoint overlay when the YAML follows the multi-trajectory
      schema.
    * All-trajectories (``--plot-all``, requires ``-c <cfg>``): iterate over
      the YAML's ``trajectories:`` list and plot each CSV that exists.

Produces for each trajectory:
    1. 3D trajectory overlaid with its waypoints.
    2. 4x3 time history (pos/vel/acc per axis + Euclidean norms).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_COLUMNS = ["t", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"]

AXIS_COLORS = ("tab:blue", "tab:orange", "tab:green")
NORM_COLOR = "tab:purple"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a mav_trajectory_generation_lib CSV.")
    parser.add_argument(
        "-f", "--file", default="trajectory.csv",
        help="Path to the trajectory CSV (single-trajectory mode).",
    )
    parser.add_argument(
        "-c", "--example-config", default=None,
        help="Example config YAML (required with --plot-all; optional for waypoint overlay "
             "in single-trajectory mode).",
    )
    parser.add_argument(
        "--trajectory-index", type=int, default=0,
        help="Index into the YAML's 'trajectories:' list to pick waypoints from "
             "(single-trajectory mode only).",
    )
    parser.add_argument(
        "--plot-all", action="store_true",
        help="Iterate over 'trajectories:' in the config and plot each existing CSV.",
    )
    parser.add_argument(
        "--vmax", type=float, default=None,
        help="Reference max velocity [m/s] drawn as a horizontal line on |v|.",
    )
    parser.add_argument(
        "--save", default=None,
        help="If given, save figures to <save>_3d.png and <save>_time.png (single-trajectory) "
             "or <save>_<label>_3d.png / <save>_<label>_time.png (--plot-all).",
    )
    return parser.parse_args()


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        sys.exit(f"CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"CSV {path} is missing columns: {missing}")
    return df


def _yaml_load(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        print(f"Warning: config not found at {path}.", file=sys.stderr)
        return None
    try:
        import yaml
    except ImportError:
        print("Warning: PyYAML not installed.", file=sys.stderr)
        return None
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    return doc if isinstance(doc, dict) else None


def _waypoints_from_entry(entry: dict) -> Optional[np.ndarray]:
    wps = entry.get("waypoints")
    if not wps:
        return None
    return np.asarray(wps, dtype=float)


def load_waypoints(cfg_path: Optional[str],
                   trajectory_index: int = 0) -> Optional[np.ndarray]:
    """Load the waypoints of a single trajectory entry from the YAML.

    Supports both schemas:
      * Legacy flat ``waypoints:`` at top level.
      * Current ``trajectories: - waypoints:`` list.
    """
    if cfg_path is None:
        return None
    doc = _yaml_load(cfg_path)
    if doc is None:
        return None
    # Legacy schema.
    if "waypoints" in doc and doc.get("waypoints"):
        return np.asarray(doc["waypoints"], dtype=float)
    # Multi-trajectory schema.
    trajs = doc.get("trajectories")
    if isinstance(trajs, list) and trajs:
        if trajectory_index < 0 or trajectory_index >= len(trajs):
            print(f"Warning: trajectory-index {trajectory_index} out of range "
                  f"(have {len(trajs)} entries).", file=sys.stderr)
            return None
        return _waypoints_from_entry(trajs[trajectory_index])
    return None


def load_trajectory_entries(cfg_path: str) -> List[Tuple[str, str, np.ndarray]]:
    """Return the ``(label, output_csv, waypoints_array)`` tuples from a YAML.

    Only usable with the multi-trajectory schema.
    """
    doc = _yaml_load(cfg_path)
    if doc is None:
        sys.exit(f"Cannot load config: {cfg_path}")
    trajs = doc.get("trajectories")
    if not isinstance(trajs, list) or not trajs:
        sys.exit(f"'{cfg_path}' has no 'trajectories:' sequence.")
    out: List[Tuple[str, str, np.ndarray]] = []
    for i, entry in enumerate(trajs):
        label = str(entry.get("label", f"traj_{i}"))
        csv_path = str(entry.get("output_csv", ""))
        wps = _waypoints_from_entry(entry)
        out.append((label, csv_path, wps if wps is not None else np.empty((0, 3))))
    return out


def plot_trajectory_3d(df: pd.DataFrame,
                       waypoints: Optional[np.ndarray],
                       title: str = "3D trajectory") -> plt.Figure:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    z = df["z"].to_numpy()
    ax.plot(x, y, z, color="tab:blue", label="trajectory")
    ax.scatter(x[0], y[0], z[0], color="tab:green", s=60, label="start")
    ax.scatter(x[-1], y[-1], z[-1], color="tab:red", s=60, label="end")
    if waypoints is not None and waypoints.size > 0:
        ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
                   color="black", marker="x", s=80, label="waypoints")
        for idx, wp in enumerate(waypoints):
            ax.text(wp[0], wp[1], wp[2], f"  w{idx}", color="black", fontsize=9)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _waypoint_closest_times(df: pd.DataFrame, waypoints: np.ndarray) -> np.ndarray:
    pos = df[["x", "y", "z"]].to_numpy()
    t = df["t"].to_numpy()
    idx = np.argmin(
        np.linalg.norm(pos[:, None, :] - waypoints[None, :, :], axis=2),
        axis=0,
    )
    return t[idx]


def plot_time_history(df: pd.DataFrame,
                      waypoints: Optional[np.ndarray],
                      vmax: Optional[float],
                      title: str = "Trajectory time history") -> plt.Figure:
    fig, axes = plt.subplots(4, 3, sharex=True, figsize=(15, 11))
    t = df["t"].to_numpy()

    pos_cols = ("x", "y", "z")
    vel_cols = ("vx", "vy", "vz")
    acc_cols = ("ax", "ay", "az")

    wp_times = None
    if waypoints is not None and waypoints.size > 0:
        wp_times = _waypoint_closest_times(df, waypoints)

    for row in range(3):
        color = AXIS_COLORS[row]
        axes[row, 0].plot(t, df[pos_cols[row]].to_numpy(), color=color)
        axes[row, 0].set_ylabel(f"{pos_cols[row]} [m]")
        axes[row, 0].grid(True, alpha=0.3)
        if wp_times is not None:
            axes[row, 0].scatter(
                wp_times, waypoints[:, row],
                color="black", marker="x", s=60, zorder=5,
                label="waypoints" if row == 0 else None,
            )

        axes[row, 1].plot(t, df[vel_cols[row]].to_numpy(), color=color)
        axes[row, 1].set_ylabel(f"{vel_cols[row]} [m/s]")
        axes[row, 1].grid(True, alpha=0.3)

        axes[row, 2].plot(t, df[acc_cols[row]].to_numpy(), color=color)
        axes[row, 2].set_ylabel(f"{acc_cols[row]} [m/s^2]")
        axes[row, 2].grid(True, alpha=0.3)

    norm_p = np.linalg.norm(df[list(pos_cols)].to_numpy(), axis=1)
    norm_v = np.linalg.norm(df[list(vel_cols)].to_numpy(), axis=1)
    norm_a = np.linalg.norm(df[list(acc_cols)].to_numpy(), axis=1)

    axes[3, 0].plot(t, norm_p, color=NORM_COLOR)
    axes[3, 0].set_ylabel("|p| [m]")
    axes[3, 0].grid(True, alpha=0.3)

    axes[3, 1].plot(t, norm_v, color=NORM_COLOR, label="|v|")
    axes[3, 1].set_ylabel("|v| [m/s]")
    axes[3, 1].grid(True, alpha=0.3)
    if vmax is not None:
        axes[3, 1].axhline(vmax, color="tab:red", linestyle="--",
                           label=f"v_max = {vmax} m/s")
        axes[3, 1].legend(loc="best", fontsize=8)

    axes[3, 2].plot(t, norm_a, color=NORM_COLOR)
    axes[3, 2].set_ylabel("|a| [m/s^2]")
    axes[3, 2].grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("t [s]")

    axes[0, 0].set_title("Position")
    axes[0, 1].set_title("Velocity")
    axes[0, 2].set_title("Acceleration")
    if wp_times is not None:
        axes[0, 0].legend(loc="best", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _plot_one(df: pd.DataFrame,
              waypoints: Optional[np.ndarray],
              vmax: Optional[float],
              label: str,
              save_prefix: Optional[str]) -> None:
    title_3d = f"3D trajectory — {label}" if label else "3D trajectory"
    title_th = f"Time history — {label}" if label else "Trajectory time history"
    fig_3d = plot_trajectory_3d(df, waypoints, title=title_3d)
    fig_th = plot_time_history(df, waypoints, vmax, title=title_th)
    if save_prefix is not None:
        # Create parent directory if needed
        save_dir = str(Path(save_prefix).parent)
        if save_dir and save_dir != ".":
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        suffix = f"_{label}" if label else ""
        fig_3d.savefig(f"{save_prefix}{suffix}_3d.png", dpi=120)
        fig_th.savefig(f"{save_prefix}{suffix}_time.png", dpi=120)
        print(f"Saved {save_prefix}{suffix}_3d.png and {save_prefix}{suffix}_time.png")
        plt.close(fig_3d)
        plt.close(fig_th)


def main() -> None:
    args = parse_args()

    if args.plot_all:
        if args.example_config is None:
            sys.exit("--plot-all requires -c / --example-config.")
        entries = load_trajectory_entries(args.example_config)
        plotted = 0
        for label, csv_path, waypoints in entries:
            if not csv_path or not os.path.isfile(csv_path):
                print(f"Skipping '{label}': CSV '{csv_path}' not found.")
                continue
            df = load_csv(csv_path)
            wps = waypoints if waypoints.size > 0 else None
            _plot_one(df, wps, args.vmax, label, args.save)
            plotted += 1
        if plotted == 0:
            sys.exit("No trajectories could be plotted.")
        if args.save is None:
            plt.show()
        return

    # Single-trajectory mode.
    df = load_csv(args.file)
    waypoints = load_waypoints(args.example_config, args.trajectory_index)
    _plot_one(df, waypoints, args.vmax, label="", save_prefix=args.save)
    if args.save is None:
        plt.show()


if __name__ == "__main__":
    main()
