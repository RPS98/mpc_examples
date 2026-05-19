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

from mav_flight_review import LoggerConfig, MCAPLogger, TimeMode, TrajectoryPoint


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
    # True only during the mission-active window: the first waypoint acts as
    # an implicit takeoff (drone starts at (0, 0, 0)) so we mark it
    # ``False``; the bool flips to ``True`` once the scheduler advances past
    # it (``waypoint_index >= 1``) and back to ``False`` when the final
    # hover phase begins. Mirrors the latched ``debug/mission/experiment_active``
    # topic of aerostack2's mission.py / mission_moving_path.py.
    experiment_active: bool = False
    max_speed: float = 0.0

    # Saturated linear velocity the active controller is tracking. Only
    # emitted when ``publishes_desired_velocity`` is True.
    desired_velocity: np.ndarray = field(default_factory=_zero_vec3)
    publishes_desired_velocity: bool = False

    # Mission-side emission gates so the MCAP mirrors aerostack2's
    # publish pattern: ``debug/mission/reference/pose`` is rate-limited
    # (10 Hz for triangle, broadcaster_rate_hz for moving_path) and the
    # four latched mission topics are emitted only when their value
    # changes. WaypointsSimulator computes the decisions; the logger
    # routes the matching ``save_*`` invocation under each flag.
    publish_mission_signals: bool = True
    publish_mission_pose_ref: bool = False
    publish_waypoint_index_change: bool = False
    publish_max_speed_change: bool = False
    publish_experiment_active_change: bool = False
    publish_hover_active_change: bool = False

    # Horizon of trajectory setpoints (position + velocity + acceleration
    # + yaw) the controller is consuming. Mirrors aerostack2's
    # `motion_reference/trajectory` (as2_msgs/msg/TrajectorySetpoints).
    # Empty by default; populated by WaypointsSimulator for trajectory-
    # scope runs and emitted at outer-loop cadence (one publication per
    # outer tick).
    trajectory_horizon: list = field(default_factory=list)
    publish_trajectory_horizon: bool = False


# Custom topic names (kept in sync with examples_cpp/src/framework/unified_mcap_logger.cpp).
# Aerostack2-native topic names. Aligned with
# mav_flight_review/pybind/python/mav_flight_review/data_model.py so the
# reviewer surfaces timing / mission extras from the same logger.
_TOPIC_CONTROLLER_COMPUTE = '/drone0/debug/controller/compute_output_time'
_TOPIC_GENERATOR_UPDATE = '/drone0/debug/behaviors/trajectory_generation/generation_time'
_TOPIC_GENERATOR_EVAL = '/drone0/debug/behaviors/trajectory_generation/eval_time'
_TOPIC_CONTROLLER_DELAY = '/drone0/debug/controller/delay_applied'
_TOPIC_GENERATOR_DELAY = '/drone0/debug/behaviors/trajectory_generation/delay_applied'
_TOPIC_MAX_SPEED = '/drone0/debug/mission/max_speed'
_TOPIC_WAYPOINT_INDEX = '/drone0/debug/mission/waypoint_index'
_TOPIC_HOVER_ACTIVE = '/drone0/debug/mission/hover_active'
# Mirror of the latched std_msgs/Bool topic published by aerostack2's
# mission scripts. Published as Int32 (0/1) because the MCAP writer
# doesn't expose a Bool API; mav_flight_review's flight_frame ingests
# either schema (only reads ``data``) and metrics-side masking does ``> 0.5``.
_TOPIC_EXPERIMENT_ACTIVE = '/drone0/debug/mission/experiment_active'
_TOPIC_MOTOR_SPEEDS = '/drone0/actuator_command/motor_speeds'
# Aerostack2-native pose reference topic. Aligned with
# mav_flight_review/flight_frame.py::_TOPIC_POSE_REF so load_flight_frame
# picks up the smooth reference automatically and clip_to_pose_ref_window
# has data to work with. The stepwise waypoint target is published
# separately under motion_reference/position (Vector3).
_TOPIC_MISSION_REF_POSE = '/drone0/debug/mission/reference/pose'
_TOPIC_MOTION_REF_POSITION = '/drone0/motion_reference/position'
# Velocity the active controller is tracking (post-saturation). Mirrors the
# aerostack2 plugins' `debug/controller/desired_velocity` topic.
_TOPIC_DESIRED_VELOCITY = '/drone0/debug/controller/desired_velocity'
_TOPIC_META_CONTROLLER = '/drone0/debug/mission/metadata/controller_name'
_TOPIC_META_GENERATOR = '/drone0/debug/mission/metadata/generator_name'
_TOPIC_META_RUN_ID = '/drone0/debug/mission/metadata/run_id'
_TOPIC_META_LANGUAGE = '/drone0/debug/mission/metadata/language'


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
        # Route the built-in pose reference channel to the aerostack2-native
        # mission-reference topic so the reviewer
        # (mav_flight_review.flight_frame::_TOPIC_POSE_REF) picks up the
        # controller-visible smooth reference. The stepwise waypoint target
        # gets its own motion_reference/position (Vector3) channel below.
        cfg.pose_reference_topic = _TOPIC_MISSION_REF_POSE

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
        self._impl.add_int32_topic(_TOPIC_EXPERIMENT_ACTIVE)
        self._impl.add_float64_multi_array_topic(_TOPIC_MOTOR_SPEEDS)
        self._impl.add_vector3_topic(_TOPIC_MOTION_REF_POSITION)
        self._impl.add_twist_stamped_topic(_TOPIC_DESIRED_VELOCITY)
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

        if row.publish_mission_pose_ref:
            # Mirror aerostack2's mission.py (triangle) and
            # mission_moving_path.py (continuous): this topic carries the
            # target the drone is currently asked to reach (active
            # waypoint in triangle, moving TF sample in moving_path).
            # The caller decides which goes through `mission_pose_ref_position`.
            ref_pos_pose = np.asarray(
                row.mission_pose_ref_position, dtype=float).reshape(3)
            identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            self._impl.save_pose_reference(t, ref_pos_pose, identity_quat)
        if row.publish_mission_signals:
            trj_vel = np.asarray(row.trajectory_velocity, dtype=float).reshape(3)
            self._impl.save_twist_reference(t, trj_vel)
            ref_pos = np.asarray(row.reference_position, dtype=float).reshape(3)
            self._impl.save_vector3(_TOPIC_MOTION_REF_POSITION, t, ref_pos)

        ang_cmd = np.asarray(row.command_angular_velocity, dtype=float).reshape(3)
        self._impl.save_actuation(t, float(row.thrust_n), ang_cmd)

        motors = np.asarray(row.motor_w, dtype=float).reshape(-1).tolist()
        self._impl.save_float64_multi_array(_TOPIC_MOTOR_SPEEDS, t, motors, [4])

        # Compute times and delays. The topic carries the value in
        # **seconds** (aerostack2 convention; mav_flight_review's
        # _TIMING_SECONDS_TO_US factor scales it back to microseconds at
        # ingest). The LogRow fields stay in microseconds for in-process
        # consumers, so we divide by 1e6 before each save_float64.
        us_to_s = 1.0e-6
        self._impl.save_float64(
            _TOPIC_CONTROLLER_COMPUTE, t,
            row.controller_compute_time_us * us_to_s)
        self._impl.save_float64(
            _TOPIC_GENERATOR_UPDATE, t,
            row.generator_update_time_us * us_to_s)
        self._impl.save_float64(
            _TOPIC_GENERATOR_EVAL, t,
            row.generator_eval_time_us * us_to_s)
        self._impl.save_float64(
            _TOPIC_CONTROLLER_DELAY, t,
            row.controller_delay_applied_us * us_to_s)
        self._impl.save_float64(
            _TOPIC_GENERATOR_DELAY, t,
            row.generator_delay_applied_us * us_to_s)

        # Mission-side latched topics: emit only when the upstream gate
        # flags the value as changed (computed in WaypointsSimulator),
        # mirroring aerostack2's latched publication semantics.
        if row.publish_waypoint_index_change:
            self._impl.save_int32(_TOPIC_WAYPOINT_INDEX, t, int(row.waypoint_index))
        if row.publish_max_speed_change:
            self._impl.save_float64(_TOPIC_MAX_SPEED, t, float(row.max_speed))
        if row.publish_hover_active_change:
            self._impl.save_int32(_TOPIC_HOVER_ACTIVE, t, 1 if row.hover_active else 0)
        if row.publish_experiment_active_change:
            self._impl.save_int32(
                _TOPIC_EXPERIMENT_ACTIVE, t, 1 if row.experiment_active else 0)

        if row.publishes_desired_velocity:
            self._impl.save_twist_stamped(
                _TOPIC_DESIRED_VELOCITY, t,
                np.asarray(row.desired_velocity, dtype=float),
                np.zeros(3, dtype=float))

        if row.publish_trajectory_horizon and row.trajectory_horizon:
            # Mirror aerostack2's `motion_reference/trajectory`. The
            # default trajectory_reference_topic of mav_flight_review's
            # MCAPLogger is already `/drone0/motion_reference/trajectory`,
            # so no override is needed.
            self._impl.save_trajectory_reference(
                t, list(row.trajectory_horizon))

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
