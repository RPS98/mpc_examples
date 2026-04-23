"""Minimal Python example: write an MCAP flight log from Python.

Produces the same helical trajectory and schema as the C++ example so that
either file can be opened by the same ``mav-view`` dashboard.

Run:
    python examples/basic_python_logging.py /tmp/flight_python.mcap
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from mav_flight_mcap import MCAPRecorder, euler_to_quaternion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="flight_python.mcap",
                        type=Path)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.01)
    args = parser.parse_args(argv)

    n_motors = 4
    extra_fields = ["solve_time_us", "waypoint_index"]
    radius = 2.0
    ascent_rate = 0.2
    hover_w = 500.0

    with MCAPRecorder(args.output, n_motors=n_motors,
                      extra_fields=extra_fields) as recorder:
        n_steps = int(args.duration / args.dt)
        for k in range(n_steps + 1):
            t = k * args.dt
            theta = 0.5 * t

            position = np.array([radius * math.cos(theta),
                                 radius * math.sin(theta),
                                 1.0 + ascent_rate * t])
            linear_velocity = np.array([-radius * 0.5 * math.sin(theta),
                                         radius * 0.5 * math.cos(theta),
                                         ascent_rate])
            angular_velocity = np.array([0.0, 0.0, 0.5])
            orientation = euler_to_quaternion(0.0, 0.0, theta)

            reference_position = position.copy()
            reference_velocity = linear_velocity.copy()
            reference_orientation = euler_to_quaternion(0.0, 0.0, theta + 0.05)
            reference_angular_velocity = np.array([0.0, 0.0, 0.5])

            thrust = 9.81
            command_angular_velocity = np.array([0.0, 0.0, 0.5])
            motor_w = np.full(n_motors, hover_w)

            extras = {
                "solve_time_us": 850.0 + 10.0 * math.sin(3.0 * t),
                "waypoint_index": float(k // 200),
            }

            recorder.save(
                t, position, orientation, linear_velocity, angular_velocity,
                reference_position, reference_velocity, reference_orientation,
                reference_angular_velocity, thrust,
                command_angular_velocity, motor_w,
                extras=extras,
            )
        print(f"Wrote {n_steps + 1} steps to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
