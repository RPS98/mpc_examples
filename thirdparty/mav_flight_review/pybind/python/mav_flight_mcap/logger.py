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

"""Python wrapper around the pybind11 MCAPLogger with typed validation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt

from ._logger_cpp import LoggerConfig, MCAPLogger as _MCAPLoggerCpp, TrajectoryPoint

Vec3 = npt.NDArray[np.float64]
Quat = npt.NDArray[np.float64]


def _asvec3(name: str, v: npt.ArrayLike) -> Vec3:
    """Return `v` as a float64 (3,) array or raise ValueError."""
    arr = np.asarray(v, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f'{name} must have shape (3,), got {arr.shape}.')
    return arr


def _asquat_wxyz(name: str, q: npt.ArrayLike) -> Quat:
    """Return `q` as a float64 (4,) array (order [w, x, y, z])."""
    arr = np.asarray(q, dtype=np.float64)
    if arr.shape != (4,):
        raise ValueError(f'{name} must have shape (4,) ([w, x, y, z]), got {arr.shape}.')
    return arr


class MCAPLogger:
    """ROS 2 Humble-compatible MCAP logger with typed per-topic save methods."""

    def __init__(self, cfg: LoggerConfig) -> None:
        """Construct the logger (does not open the file yet)."""
        self._impl = _MCAPLoggerCpp(cfg)

    # --- Context manager ------------------------------------------------------

    def __enter__(self) -> 'MCAPLogger':
        """Start the logger when used as a context manager."""
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close the logger when leaving the context."""
        self.close()

    # --- Topic setters --------------------------------------------------------

    def set_pose_reference_topic(self, topic: str) -> None:
        """Override the pose_reference topic."""
        self._impl.set_pose_reference_topic(topic)

    def set_twist_reference_topic(self, topic: str) -> None:
        """Override the twist_reference topic."""
        self._impl.set_twist_reference_topic(topic)

    def set_trajectory_reference_topic(self, topic: str) -> None:
        """Override the trajectory_reference topic."""
        self._impl.set_trajectory_reference_topic(topic)

    def set_thrust_command_topic(self, topic: str) -> None:
        """Override the thrust command topic."""
        self._impl.set_thrust_command_topic(topic)

    def set_twist_command_topic(self, topic: str) -> None:
        """Override the twist command topic."""
        self._impl.set_twist_command_topic(topic)

    def set_pose_state_topic(self, topic: str) -> None:
        """Override the pose state topic."""
        self._impl.set_pose_state_topic(topic)

    def set_twist_state_topic(self, topic: str) -> None:
        """Override the twist state topic."""
        self._impl.set_twist_state_topic(topic)

    def set_odom_state_topic(self, topic: str) -> None:
        """Override the odometry state topic."""
        self._impl.set_odom_state_topic(topic)

    # --- Extras registration --------------------------------------------------

    def add_int32_topic(self, topic: str) -> None:
        """Register an extra std_msgs/Int32 topic."""
        self._impl.add_int32_topic(topic)

    def add_string_topic(self, topic: str) -> None:
        """Register an extra std_msgs/String topic."""
        self._impl.add_string_topic(topic)

    def add_float64_topic(self, topic: str) -> None:
        """Register an extra std_msgs/Float64 topic."""
        self._impl.add_float64_topic(topic)

    def add_float64_multi_array_topic(self, topic: str) -> None:
        """Register an extra std_msgs/Float64MultiArray topic."""
        self._impl.add_float64_multi_array_topic(topic)

    def add_vector3_topic(self, topic: str) -> None:
        """Register an extra geometry_msgs/Vector3 topic."""
        self._impl.add_vector3_topic(topic)

    # --- Lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Open the MCAP file and lock the topic configuration."""
        self._impl.start()

    def close(self) -> None:
        """Flush and close the MCAP file (idempotent)."""
        self._impl.close()

    @property
    def is_running(self) -> bool:
        """Whether the logger is open (after start() and before close())."""
        return self._impl.is_running()

    # --- Single-topic saves ---------------------------------------------------

    def save_pose_reference(self, t: float, pos: npt.ArrayLike, quat_wxyz: npt.ArrayLike) -> None:
        """Save a PoseStamped on pose_reference_topic."""
        self._impl.save_pose_reference(t, _asvec3('pos', pos), _asquat_wxyz('quat_wxyz', quat_wxyz))

    def save_twist_reference(self, t: float, linear: npt.ArrayLike) -> None:
        """Save a TwistStamped (linear only) on twist_reference_topic."""
        self._impl.save_twist_reference(t, _asvec3('linear', linear))

    def save_trajectory_reference(
            self, t: float, points: Sequence[TrajectoryPoint]) -> None:
        """Save a TrajectorySetpoints on trajectory_reference_topic."""
        self._impl.save_trajectory_reference(t, list(points))

    def save_thrust_command(self, t: float, thrust: float) -> None:
        """Save a Thrust on thrust_command_topic."""
        self._impl.save_thrust_command(t, float(thrust))

    def save_twist_command(self, t: float, angular: npt.ArrayLike) -> None:
        """Save a TwistStamped (angular only) on twist_command_topic."""
        self._impl.save_twist_command(t, _asvec3('angular', angular))

    def save_pose_state(self, t: float, pos: npt.ArrayLike, quat_wxyz: npt.ArrayLike) -> None:
        """Save a PoseStamped on pose_state_topic."""
        self._impl.save_pose_state(t, _asvec3('pos', pos), _asquat_wxyz('quat_wxyz', quat_wxyz))

    def save_twist_state(
            self, t: float, linear: npt.ArrayLike, angular: npt.ArrayLike) -> None:
        """Save a TwistStamped on twist_state_topic."""
        self._impl.save_twist_state(t, _asvec3('linear', linear), _asvec3('angular', angular))

    def save_odom_state(
            self,
            t: float,
            pos_earth: npt.ArrayLike,
            quat_wxyz: npt.ArrayLike,
            linear_body: npt.ArrayLike,
            angular_body: npt.ArrayLike) -> None:
        """Save an Odometry on odom_state_topic."""
        self._impl.save_odom_state(
            t,
            _asvec3('pos_earth', pos_earth),
            _asquat_wxyz('quat_wxyz', quat_wxyz),
            _asvec3('linear_body', linear_body),
            _asvec3('angular_body', angular_body))

    # --- Aggregate saves ------------------------------------------------------

    def save_state(
            self,
            t: float,
            pos_earth: npt.ArrayLike,
            quat_wxyz: npt.ArrayLike,
            linear_earth: npt.ArrayLike,
            angular_body: npt.ArrayLike) -> None:
        """Save PoseStamped + TwistStamped + Odometry in a single call."""
        self._impl.save_state(
            t,
            _asvec3('pos_earth', pos_earth),
            _asquat_wxyz('quat_wxyz', quat_wxyz),
            _asvec3('linear_earth', linear_earth),
            _asvec3('angular_body', angular_body))

    def save_position_reference(
            self,
            t: float,
            pos_ref: npt.ArrayLike,
            quat_wxyz_ref: npt.ArrayLike,
            max_linear_speed: npt.ArrayLike) -> None:
        """Save PoseStamped + TwistStamped (max speed) in a single call."""
        self._impl.save_position_reference(
            t,
            _asvec3('pos_ref', pos_ref),
            _asquat_wxyz('quat_wxyz_ref', quat_wxyz_ref),
            _asvec3('max_linear_speed', max_linear_speed))

    def save_trajectory_reference_full(
            self,
            t: float,
            points: Sequence[TrajectoryPoint],
            also_emit_pose_and_twist: bool = False) -> None:
        """Save TrajectorySetpoints and optionally echo the first setpoint."""
        self._impl.save_trajectory_reference_full(
            t, list(points), also_emit_pose_and_twist)

    def save_actuation(self, t: float, thrust: float, angular_command_body: npt.ArrayLike) -> None:
        """Save Thrust + TwistStamped (angular) in a single call."""
        self._impl.save_actuation(
            t, float(thrust), _asvec3('angular_command_body', angular_command_body))

    # --- Extras ---------------------------------------------------------------

    def save_int32(self, topic: str, t: float, value: int) -> None:
        """Save a std_msgs/Int32 on a registered extra topic."""
        self._impl.save_int32(topic, t, int(value))

    def save_string(self, topic: str, t: float, value: str) -> None:
        """Save a std_msgs/String on a registered extra topic."""
        self._impl.save_string(topic, t, str(value))

    def save_float64(self, topic: str, t: float, value: float) -> None:
        """Save a std_msgs/Float64 on a registered extra topic."""
        self._impl.save_float64(topic, t, float(value))

    def save_vector3(self, topic: str, t: float, value: npt.ArrayLike) -> None:
        """Save a geometry_msgs/Vector3 on a registered extra topic."""
        self._impl.save_vector3(topic, t, _asvec3('value', value))

    def save_float64_multi_array(
            self,
            topic: str,
            t: float,
            value: npt.ArrayLike,
            dims: Sequence[int] | None = None) -> None:
        """Save a std_msgs/Float64MultiArray on a registered extra topic."""
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        dim_list = [int(x) for x in (dims or [])]
        self._impl.save_float64_multi_array(topic, t, arr.tolist(), dim_list)
