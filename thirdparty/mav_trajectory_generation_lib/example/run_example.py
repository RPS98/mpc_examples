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

"""Python mirror of run_example.cpp.

Reads a list of trajectories from the example YAML and plans each with the
same ``TrajectoryGenerator`` instance (reused across all trajectories). The
second trajectory is chained to the previous one by seeding its initial
velocity with ``evaluate(max_time).velocity``; the third one intentionally
fails to show graceful error handling.

Usage:
    python run_example.py <config_example.yaml> <config_trajectory.yaml>
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

from mav_trajectory_generation_py import (
    EndWaypoint,
    TrajectoryGenerator,
    Waypoint,
)
from mav_trajectory_generation_py import load_generator_config

LOGS_DIR = "logs"


@dataclass
class TrajectorySpec:
    label: str = ""
    output_csv: str = ""
    waypoints: List[np.ndarray] = field(default_factory=list)


@dataclass
class ExampleParams:
    step_dt: float = 0.01
    max_speed: float = 3.0
    trajectories: List[TrajectorySpec] = field(default_factory=list)


def load_example_yaml(path: str) -> ExampleParams:
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    out = ExampleParams()
    sim = doc.get("simulation", {}) or {}
    out.step_dt = float(sim.get("step_dt", out.step_dt))
    out.max_speed = float(sim.get("max_speed", out.max_speed))

    trajs = doc.get("trajectories")
    if not isinstance(trajs, list) or not trajs:
        raise ValueError(f"Expected 'trajectories:' sequence at top level of '{path}'.")
    for entry in trajs:
        spec = TrajectorySpec()
        spec.label = str(entry.get("label", "")).strip()
        spec.output_csv = str(entry.get("output_csv", "")).strip()
        if not spec.output_csv:
            raise ValueError(f"Trajectory '{spec.label}' missing 'output_csv' in '{path}'.")
        wps = entry.get("waypoints") or []
        if not isinstance(wps, list) or not wps:
            raise ValueError(f"Trajectory '{spec.label}' has no waypoints.")
        for w in wps:
            arr = np.asarray(w, dtype=np.float64).reshape(-1)
            if arr.size != 3:
                raise ValueError("Each waypoint must have exactly 3 entries (x, y, z).")
            spec.waypoints.append(arr)
        out.trajectories.append(spec)
    return out


def build_waypoints(
    spec: TrajectorySpec,
    initial_velocity: Optional[np.ndarray],
) -> List[Waypoint]:
    """Mirror of the C++ buildWaypoints(): EndWaypoint for endpoints by
    default; if an initial_velocity is provided for the first entry, wrap it
    in a plain Waypoint whose velocity is pinned to that value for
    continuous chaining with the previous trajectory."""
    out: List[Waypoint] = []
    n = len(spec.waypoints)
    for i, pos in enumerate(spec.waypoints):
        is_first = i == 0
        is_last = i == n - 1
        if is_first and initial_velocity is not None:
            wp = Waypoint(pos)
            wp.velocity = np.asarray(initial_velocity, dtype=np.float64).reshape(3)
            out.append(wp)
        elif is_first or is_last:
            out.append(EndWaypoint(pos))
        else:
            out.append(Waypoint(pos))
    return out


def write_csv(path: str, gen: TrajectoryGenerator, step_dt: float) -> bool:
    out_path = Path(path)
    try:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["t", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"])
            t = gen.min_time()
            t_end = gen.max_time()
            while t <= t_end + 1.0e-9:
                s = gen.evaluate(t)
                writer.writerow([
                    f"{t:.6f}",
                    f"{s.position[0]:.6f}", f"{s.position[1]:.6f}", f"{s.position[2]:.6f}",
                    f"{s.velocity[0]:.6f}", f"{s.velocity[1]:.6f}", f"{s.velocity[2]:.6f}",
                    f"{s.acceleration[0]:.6f}", f"{s.acceleration[1]:.6f}", f"{s.acceleration[2]:.6f}",
                ])
                t += step_dt
    except OSError as exc:
        print(f"  Could not open output file: {path} ({exc})", file=sys.stderr)
        return False
    return True


def main(argv: List[str]) -> int:
    if len(argv) < 3:
        print(f"Usage: {argv[0]} <config_example.yaml> <config_trajectory.yaml>", file=sys.stderr)
        return 1
    try:
        # Create logs directory if it doesn't exist
        Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

        example = load_example_yaml(argv[1])
        cfg = load_generator_config(argv[2])
        gen = TrajectoryGenerator(cfg.to_native())

        carry_velocity: Optional[np.ndarray] = None
        successes = 0

        for spec in example.trajectories:
            waypoints = build_waypoints(spec, carry_velocity)
            ok = gen.generate(waypoints, example.max_speed)
            if not ok:
                plural = "" if len(spec.waypoints) == 1 else "s"
                print(
                    f"[FAIL] {spec.label}: generate() failed "
                    f"({len(spec.waypoints)} waypoint{plural})",
                    file=sys.stderr,
                )
                carry_velocity = None
                continue
            csv_path = str(Path(LOGS_DIR) / Path(spec.output_csv).name)
            if not write_csv(csv_path, gen, example.step_dt):
                carry_velocity = None
                continue
            print(f"[OK]   {spec.label}: wrote {csv_path} "
                  f"(duration={gen.duration():.6f} s)")
            successes += 1
            carry_velocity = np.asarray(gen.evaluate(gen.max_time()).velocity,
                                        dtype=np.float64).reshape(3)

        if successes == 0:
            print("No trajectories were produced.", file=sys.stderr)
            return 1
    except Exception as exc:  # noqa: BLE001 — surface any failure to the CLI caller
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
