#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Facade mirror of ``examples_cpp/src/framework/unified_mcap_logger.cpp``.

Accepts the same Eigen-shaped ``LogRow`` the previous CSV facade used and
translates each row into the matching ``mav_flight_review.MCAPLogger.save_*``
calls. Keeping the ``LogRow`` schema local means ``WaypointsSimulator`` and
unit tests can switch backends without touching their row-building code.
"""

from __future__ import annotations

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from mav_flight_review import LoggerConfig, MCAPLogger, TimeMode


def _zero_vec3() -> np.ndarray:
    return np.zeros(3, dtype=float)


def _identity_quat() -> np.ndarray:
    # [w, x, y, z] order to match Eigen::Quaterniond's (w, x, y, z).
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def _zero_motor() -> np.ndarray:
    return np.zeros(4, dtype=float)


@dataclass
class RunMetadata:
    """Run-level metadata emitted once as String messages at t=0."""

    controller_name: str = ''
    generator_name: str = ''
    run_id: str = ''
    language: str = 'py'


@dataclass
class LogRow:
    """Single telemetry row. SI units, earth frame, rad for angles.

    Quaternions use the ``[w, x, y, z]`` order (matching ``Eigen::Quaterniond``
    and the ROS 2 ``geometry_msgs/Quaternion`` layout internally).
    """

    time: float = 0.0

    position: np.ndarray = field(default_factory=_zero_vec3)
    orientation: np.ndarray = field(default_factory=_identity_quat)  # [w, x, y, z]
    linear_velocity: np.ndarray = field(default_factory=_zero_vec3)    # earth frame
    angular_velocity: np.ndarray = field(default_factory=_zero_vec3)   # body frame

    # Position reference = active waypoint target (stepwise, no delay).
    # Identical across every combination of controller/generator.
    reference_position: np.ndarray = field(default_factory=_zero_vec3)

    # Trajectory sample the controller consumes (smooth, with delay applied,
    # or a virtual carrot advanced along the waypoint path).
    trajectory_position: np.ndarray = field(default_factory=_zero_vec3)
    trajectory_velocity: np.ndarray = field(default_factory=_zero_vec3)  # earth frame
    trajectory_orientation: np.ndarray = field(default_factory=_identity_quat)

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


# Custom topic names (kept in sync with examples_cpp/src/framework/unified_mcap_logger.cpp).
_TOPIC_CONTROLLER_COMPUTE = '/mpc_examples/controller_compute_time_us'
_TOPIC_GENERATOR_UPDATE = '/mpc_examples/generator_update_time_us'
_TOPIC_GENERATOR_EVAL = '/mpc_examples/generator_eval_time_us'
_TOPIC_CONTROLLER_DELAY = '/mpc_examples/controller_delay_applied_us'
_TOPIC_GENERATOR_DELAY = '/mpc_examples/generator_delay_applied_us'
_TOPIC_MAX_SPEED = '/mpc_examples/max_speed'
_TOPIC_WAYPOINT_INDEX = '/mpc_examples/waypoint_index'
_TOPIC_HOVER_ACTIVE = '/mpc_examples/hover_active'
_TOPIC_MOTOR_SPEEDS = '/drone0/actuator_command/motor_speeds'
_TOPIC_MOTION_REF_TRAJECTORY = '/drone0/motion_reference/trajectory'
_TOPIC_MOTION_REF_POSITION = '/drone0/motion_reference/position'
_TOPIC_META_CONTROLLER = '/mpc_examples/metadata/controller_name'
_TOPIC_META_GENERATOR = '/mpc_examples/metadata/generator_name'
_TOPIC_META_RUN_ID = '/mpc_examples/metadata/run_id'
_TOPIC_META_LANGUAGE = '/mpc_examples/metadata/language'


class UnifiedMcapLogger:
    """Facade MCAP logger mirroring the legacy ``UnifiedCsvLogger`` API."""

    def __init__(self, output_path: str, metadata: RunMetadata) -> None:
        """Open ``output_path`` for writing and emit the metadata block at t=0."""
        self._file_path = output_path
        self._metadata = metadata
        self._closed = False

        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        cfg = LoggerConfig()
        cfg.file_path = output_path
        cfg.time_mode = TimeMode.SIMULATION
        # Route the built-in pose reference channel to the
        # ``motion_reference/trajectory`` topic (generator output, with delay);
        # the stepwise waypoint target gets its own ``motion_reference/position``
        # channel below.
        cfg.pose_reference_topic = _TOPIC_MOTION_REF_TRAJECTORY

        self._impl: Optional[MCAPLogger] = MCAPLogger(cfg)

        # Declare custom topics before calling start().
        self._impl.add_float64_topic(_TOPIC_CONTROLLER_COMPUTE)
        self._impl.add_float64_topic(_TOPIC_GENERATOR_UPDATE)
        self._impl.add_float64_topic(_TOPIC_GENERATOR_EVAL)
        self._impl.add_float64_topic(_TOPIC_CONTROLLER_DELAY)
        self._impl.add_float64_topic(_TOPIC_GENERATOR_DELAY)
        self._impl.add_float64_topic(_TOPIC_MAX_SPEED)
        self._impl.add_int32_topic(_TOPIC_WAYPOINT_INDEX)
        self._impl.add_int32_topic(_TOPIC_HOVER_ACTIVE)
        self._impl.add_float64_multi_array_topic(_TOPIC_MOTOR_SPEEDS)
        self._impl.add_vector3_topic(_TOPIC_MOTION_REF_POSITION)
        self._impl.add_string_topic(_TOPIC_META_CONTROLLER)
        self._impl.add_string_topic(_TOPIC_META_GENERATOR)
        self._impl.add_string_topic(_TOPIC_META_RUN_ID)
        self._impl.add_string_topic(_TOPIC_META_LANGUAGE)

        self._impl.start()

        # Emit the metadata block once at t=0 so the MCAP is self-describing.
        self._impl.save_string(_TOPIC_META_CONTROLLER, 0.0, metadata.controller_name)
        self._impl.save_string(_TOPIC_META_GENERATOR, 0.0, metadata.generator_name)
        self._impl.save_string(_TOPIC_META_RUN_ID, 0.0, metadata.run_id)
        self._impl.save_string(_TOPIC_META_LANGUAGE, 0.0, metadata.language)

    # --- Context manager ------------------------------------------------------

    def __enter__(self) -> 'UnifiedMcapLogger':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass

    # --- Properties -----------------------------------------------------------

    @property
    def path(self) -> str:
        return self._file_path

    @property
    def metadata(self) -> RunMetadata:
        return self._metadata

    # --- API ------------------------------------------------------------------

    def log_row(self, row: LogRow) -> None:
        """Write a single telemetry row (state + reference + actuation + extras)."""
        if self._impl is None or self._closed:
            raise RuntimeError('UnifiedMcapLogger: file already closed.')

        t = float(row.time)

        pos = np.asarray(row.position, dtype=float).reshape(3)
        quat = np.asarray(row.orientation, dtype=float).reshape(4)
        vel = np.asarray(row.linear_velocity, dtype=float).reshape(3)
        ang = np.asarray(row.angular_velocity, dtype=float).reshape(3)

        self._impl.save_state(t, pos, quat, vel, ang)

        trj_pos = np.asarray(row.trajectory_position, dtype=float).reshape(3)
        trj_quat = np.asarray(row.trajectory_orientation, dtype=float).reshape(4)
        self._impl.save_pose_reference(t, trj_pos, trj_quat)
        trj_vel = np.asarray(row.trajectory_velocity, dtype=float).reshape(3)
        self._impl.save_twist_reference(t, trj_vel)

        ref_pos = np.asarray(row.reference_position, dtype=float).reshape(3)
        self._impl.save_vector3(_TOPIC_MOTION_REF_POSITION, t, ref_pos)

        ang_cmd = np.asarray(row.command_angular_velocity, dtype=float).reshape(3)
        self._impl.save_actuation(t, float(row.thrust_n), ang_cmd)

        motors = np.asarray(row.motor_w, dtype=float).reshape(-1).tolist()
        self._impl.save_float64_multi_array(_TOPIC_MOTOR_SPEEDS, t, motors, [4])

        self._impl.save_float64(_TOPIC_CONTROLLER_COMPUTE, t, row.controller_compute_time_us)
        self._impl.save_float64(_TOPIC_GENERATOR_UPDATE, t, row.generator_update_time_us)
        self._impl.save_float64(_TOPIC_GENERATOR_EVAL, t, row.generator_eval_time_us)
        self._impl.save_float64(_TOPIC_CONTROLLER_DELAY, t, row.controller_delay_applied_us)
        self._impl.save_float64(_TOPIC_GENERATOR_DELAY, t, row.generator_delay_applied_us)

        self._impl.save_int32(_TOPIC_WAYPOINT_INDEX, t, int(row.waypoint_index))
        self._impl.save_int32(_TOPIC_HOVER_ACTIVE, t, 1 if row.hover_active else 0)
        self._impl.save_float64(_TOPIC_MAX_SPEED, t, float(row.max_speed))

    # Backwards-compatible alias (the CSV facade exposed both names).
    def write_row(self, row: LogRow) -> None:
        self.log_row(row)

    def close(self) -> None:
        if self._impl is not None and not self._closed:
            try:
                self._impl.close()
            finally:
                self._closed = True
                self._impl = None


__all__ = [
    'LogRow',
    'RunMetadata',
    'UnifiedMcapLogger',
]
