#!/usr/bin/env python3

# Copyright 2025 Universidad Politecnica de Madrid
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
#    * Neither the name of the Universidad Politecnica de Madrid nor the names
#      of its contributors may be used to endorse or promote products derived
#      from this software without specific prior written permission.
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

"""MPC + MAV Simulator integrated example utilities."""

__authors__ = 'Rafael Perez-Segui'
__copyright__ = 'Copyright (c) 2025 Universidad Politecnica de Madrid'
__license__ = 'BSD-3-Clause'

import math
import numpy as np


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles to quaternion [w, x, y, z]."""
    sr = math.sin(roll * 0.5)
    cr = math.cos(roll * 0.5)
    sp = math.sin(pitch * 0.5)
    cp = math.cos(pitch * 0.5)
    sy = math.sin(yaw * 0.5)
    cy = math.cos(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw]."""
    w, x, y, z = q

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def compute_path_facing(direction: np.ndarray) -> np.ndarray:
    """Compute quaternion [w, x, y, z] facing in the direction of the vector."""
    yaw = math.atan2(direction[1], direction[0])
    return euler_to_quaternion(0.0, 0.0, yaw)


def advance_reference_position(
        current_ref: np.ndarray,
        target: np.ndarray,
        max_speed: float,
        dt: float) -> np.ndarray:
    """Move reference position toward target with bounded speed."""
    delta = target - current_ref
    distance = np.linalg.norm(delta)
    if distance < 1e-9:
        return target.copy()

    max_step = max(0.0, float(max_speed)) * float(dt)
    if distance <= max_step:
        return target.copy()

    return current_ref + delta * (max_step / distance)


class CsvLogger:
    """Log MPC + simulator data to a CSV file.

    Columns: time,
        x, y, z, qw, qx, qy, qz, roll, pitch, yaw,
        vx, vy, vz, wx, wy, wz,
        x_ref, y_ref, z_ref,
        qw_ref, qx_ref, qy_ref, qz_ref, roll_ref, pitch_ref, yaw_ref,
        thrust, wx_cmd, wy_cmd, wz_cmd,
        motor_w0, motor_w1, motor_w2, motor_w3
    """

    def __init__(self, file_name: str) -> None:
        self.file_name = file_name
        print(f'Saving to file: {self.file_name}')
        self.file = open(self.file_name, 'w')
        self.file.write(
            'time,'
            'x,y,z,qw,qx,qy,qz,roll,pitch,yaw,'
            'vx,vy,vz,wx,wy,wz,'
            'x_ref,y_ref,z_ref,'
            'qw_ref,qx_ref,qy_ref,qz_ref,roll_ref,pitch_ref,yaw_ref,'
            'thrust,wx_cmd,wy_cmd,wz_cmd,'
            'motor_w0,motor_w1,motor_w2,motor_w3\n')

    def _write_value(self, value: float, comma: bool = True) -> None:
        self.file.write(f'{value}')
        if comma:
            self.file.write(',')

    def _write_vector(self, data: np.ndarray, comma: bool = True) -> None:
        for i in range(data.size):
            last = (i == data.size - 1)
            self._write_value(data[i], comma or not last)

    def save(
            self,
            time: float,
            position: np.ndarray,
            orientation: np.ndarray,
            linear_velocity: np.ndarray,
            angular_velocity: np.ndarray,
            reference_position: np.ndarray,
            reference_orientation: np.ndarray,
            thrust: float,
            command_angular_velocity: np.ndarray,
            motor_w: np.ndarray) -> None:
        """Save one simulation step."""
        euler = quaternion_to_euler(orientation)
        euler_ref = quaternion_to_euler(reference_orientation)

        # Time
        self._write_value(time)
        # State
        self._write_vector(position)
        self._write_vector(orientation)
        self._write_vector(euler)
        self._write_vector(linear_velocity)
        self._write_vector(angular_velocity)
        # Reference
        self._write_vector(reference_position)
        self._write_vector(reference_orientation)
        self._write_vector(euler_ref)
        # MPC actuation
        self._write_value(thrust)
        self._write_vector(command_angular_velocity)
        # Motor state
        self._write_vector(motor_w, comma=False)
        self.file.write('\n')

    def close(self) -> None:
        """Close the CSV file."""
        self.file.close()
