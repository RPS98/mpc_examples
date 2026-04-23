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

"""End-to-end Python example matching basic_cpp_logging.cpp."""

from __future__ import annotations

import argparse
import math

import numpy as np

from mav_flight_mcap import LoggerConfig, MCAPLogger, TimeMode, TrajectoryPoint


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    """Convert a yaw angle (rad) around +Z into a [w, x, y, z] quaternion."""
    half = 0.5 * yaw
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)])


def main() -> None:
    """Log a 10 s helical trajectory with every spec topic plus debug extras."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', nargs='?', default='/tmp/flight_py.mcap',
                        help='output MCAP path')
    args = parser.parse_args()

    cfg = LoggerConfig()
    cfg.file_path = args.path
    cfg.time_mode = TimeMode.SIMULATION

    # Extras must be registered before start().
    logger = MCAPLogger(cfg)
    logger.add_float64_topic('/drone0/debug/solve_time_us')
    logger.add_int32_topic('/drone0/debug/waypoint_index')
    logger.start()

    dt = 0.01
    t_end = 10.0
    radius = 2.0
    omega = 1.0
    ascent = 0.2
    max_speed = 3.0

    setpoints: list[TrajectoryPoint] = []
    for k in range(5):
        theta = k * (omega * t_end) / 5.0
        p = TrajectoryPoint()
        p.id = f'wp{k}'
        p.position = np.array([radius * math.cos(theta),
                               radius * math.sin(theta),
                               1.0 + ascent * theta / omega])
        p.twist = np.zeros(3)
        p.acceleration = np.zeros(3)
        p.yaw_angle = float(theta)
        setpoints.append(p)

    n_steps = int(t_end / dt)
    for i in range(n_steps + 1):
        t = i * dt
        theta = omega * t

        pos = np.array([radius * math.cos(theta),
                        radius * math.sin(theta),
                        1.0 + ascent * t])
        vel = np.array([-radius * omega * math.sin(theta),
                        radius * omega * math.cos(theta),
                        ascent])
        ang_body = np.array([0.0, 0.0, omega])
        q_wxyz = yaw_to_quat_wxyz(theta)

        logger.save_state(t, pos, q_wxyz, vel, ang_body)
        logger.save_position_reference(t, pos, q_wxyz, np.full(3, max_speed))
        if i % 100 == 0:
            logger.save_trajectory_reference(t, setpoints)
        logger.save_actuation(t, 9.81, ang_body)
        logger.save_float64('/drone0/debug/solve_time_us', t, 800.0 + 10.0 * i)
        logger.save_int32('/drone0/debug/waypoint_index', t, i // 200)

        if i % 100 == 0:
            print(f'  t = {t:.2f} s  pos = ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})')

    logger.close()
    print(f'Wrote {args.path}')


if __name__ == '__main__':
    main()
