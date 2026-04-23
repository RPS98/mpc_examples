#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Python mirror of ``thirdparty/mav_flight_logger/src/csv_logger.cpp``.

Defines the canonical 45-column schema shared by every MAV flight telemetry
CSV. Pure-Python, numpy-only.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

import math
import os
from dataclasses import dataclass, field
from typing import IO, Optional

import numpy as np


# Canonical 45-column header (comma-separated, no trailing newline).
COLUMN_HEADER = (
    'time,'
    'x,y,z,qw,qx,qy,qz,roll,pitch,yaw,'
    'vx,vy,vz,wx,wy,wz,'
    'x_ref,y_ref,z_ref,qw_ref,qx_ref,qy_ref,qz_ref,roll_ref,pitch_ref,yaw_ref,'
    'thrust,wx_cmd,wy_cmd,wz_cmd,'
    'motor_w0,motor_w1,motor_w2,motor_w3,'
    'controller_name,generator_name,'
    'controller_compute_time_us,generator_update_time_us,generator_eval_time_us,'
    'controller_delay_applied_us,generator_delay_applied_us,'
    'waypoint_index,hover_active,max_speed'
)

COLUMN_COUNT = len(COLUMN_HEADER.split(','))


def _zero_vec3() -> np.ndarray:
    return np.zeros(3, dtype=float)


def _identity_quat() -> np.ndarray:
    # [w, x, y, z]
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def _zero_motor() -> np.ndarray:
    return np.zeros(4, dtype=float)


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert a [w, x, y, z] quaternion to intrinsic ZYX Euler angles.

    Returns
    -------
    np.ndarray
        ``[roll, pitch, yaw]`` in radians (schema matches the C++ logger).
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])

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

    return np.array([roll, pitch, yaw], dtype=float)


@dataclass
class RunMetadata:
    controller_name: str = ''
    generator_name: str = ''
    run_id: str = ''
    language: str = 'py'


@dataclass
class LogRow:
    """Single CSV row. SI units, world frame, rad for angles."""

    time: float = 0.0

    position: np.ndarray = field(default_factory=_zero_vec3)
    orientation: np.ndarray = field(default_factory=_identity_quat)  # [w, x, y, z]
    linear_velocity: np.ndarray = field(default_factory=_zero_vec3)
    angular_velocity: np.ndarray = field(default_factory=_zero_vec3)

    reference_position: np.ndarray = field(default_factory=_zero_vec3)
    reference_orientation: np.ndarray = field(default_factory=_identity_quat)

    thrust_n: float = 0.0
    command_angular_velocity: np.ndarray = field(default_factory=_zero_vec3)
    motor_w: np.ndarray = field(default_factory=_zero_motor)

    controller_compute_time_us: float = 0.0
    generator_update_time_us: float = 0.0
    generator_eval_time_us: float = 0.0
    controller_delay_applied_us: float = 0.0
    generator_delay_applied_us: float = 0.0

    waypoint_index: int = 0
    hover_active: bool = False
    max_speed: float = 0.0


class CsvLogger:
    """CSV writer with the 45-column shared schema.

    The file is opened (creating parent directories as needed) in the
    constructor and closed in :meth:`close` (or on garbage collection / ``with``).
    """

    def __init__(self, output_path: str, metadata: RunMetadata) -> None:
        self._file_path = output_path
        self._metadata = metadata
        self._file: Optional[IO[str]] = None

        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        self._file = open(output_path, 'w', newline='')
        self._write_header(metadata)

    def __enter__(self) -> 'CsvLogger':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def path(self) -> str:
        return self._file_path

    @property
    def metadata(self) -> RunMetadata:
        return self._metadata

    def _write_header(self, metadata: RunMetadata) -> None:
        assert self._file is not None
        self._file.write(f'# controller: {metadata.controller_name}\n')
        self._file.write(f'# generator: {metadata.generator_name}\n')
        self._file.write(f'# run_id: {metadata.run_id}\n')
        self._file.write(f'# language: {metadata.language}\n')
        self._file.write(COLUMN_HEADER + '\n')

    def write_row(self, row: LogRow) -> None:
        if self._file is None:
            raise RuntimeError('CsvLogger: file already closed.')

        euler = quaternion_to_euler(row.orientation)
        euler_ref = quaternion_to_euler(row.reference_orientation)

        fields = [
            f'{row.time}',
            f'{row.position[0]}', f'{row.position[1]}', f'{row.position[2]}',
            f'{row.orientation[0]}', f'{row.orientation[1]}',
            f'{row.orientation[2]}', f'{row.orientation[3]}',
            f'{euler[0]}', f'{euler[1]}', f'{euler[2]}',
            f'{row.linear_velocity[0]}', f'{row.linear_velocity[1]}',
            f'{row.linear_velocity[2]}',
            f'{row.angular_velocity[0]}', f'{row.angular_velocity[1]}',
            f'{row.angular_velocity[2]}',
            f'{row.reference_position[0]}', f'{row.reference_position[1]}',
            f'{row.reference_position[2]}',
            f'{row.reference_orientation[0]}', f'{row.reference_orientation[1]}',
            f'{row.reference_orientation[2]}', f'{row.reference_orientation[3]}',
            f'{euler_ref[0]}', f'{euler_ref[1]}', f'{euler_ref[2]}',
            f'{row.thrust_n}',
            f'{row.command_angular_velocity[0]}',
            f'{row.command_angular_velocity[1]}',
            f'{row.command_angular_velocity[2]}',
            f'{row.motor_w[0]}', f'{row.motor_w[1]}',
            f'{row.motor_w[2]}', f'{row.motor_w[3]}',
            self._metadata.controller_name,
            self._metadata.generator_name,
            f'{row.controller_compute_time_us}',
            f'{row.generator_update_time_us}',
            f'{row.generator_eval_time_us}',
            f'{row.controller_delay_applied_us}',
            f'{row.generator_delay_applied_us}',
            f'{int(row.waypoint_index)}',
            '1' if row.hover_active else '0',
            f'{row.max_speed}',
        ]
        self._file.write(','.join(fields) + '\n')

    # Backwards-compatible alias mirroring the Python API of the previous
    # UnifiedCsvLogger (which used ``log_row``). Facades in examples_cpp /
    # examples_py can standardise on ``write_row`` going forward.
    def log_row(self, row: LogRow) -> None:
        self.write_row(row)

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None
