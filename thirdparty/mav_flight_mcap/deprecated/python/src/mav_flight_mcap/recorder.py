"""MCAP recorder with API parity to the C++ header.

Writes a ``.mcap`` file with protobuf-encoded messages on four logical channel
groups:

* ``/drone/state``     — :class:`mav_flight_mcap._proto.mav_flight_pb2.State`
* ``/drone/reference`` — :class:`...Reference`
* ``/drone/actuation`` — :class:`...Actuation`
* ``/drone/extras/<name>`` — :class:`...Scalar`, one channel per ``extra_fields``
  entry declared at construction time.

All time stamps are set to ``int(time * 1e9)`` (nanoseconds) for both
``log_time`` and ``publish_time``. Quaternions follow the ``[w, x, y, z]``
convention (body -> world). Units are SI.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import IO, Mapping, Sequence

import numpy as np
from mcap_protobuf.writer import Writer as ProtobufWriter

from ._proto.mav_flight_pb2 import (
    Actuation,
    Quaternion,
    Reference,
    Scalar,
    State,
    Vec3,
)

__all__ = ["MCAPRecorder", "euler_to_quaternion", "quaternion_to_euler"]


def _as_vec3(msg: Vec3, a: np.ndarray | Sequence[float]) -> None:
    arr = np.asarray(a, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"expected shape (3,), got {arr.shape}")
    msg.x = float(arr[0])
    msg.y = float(arr[1])
    msg.z = float(arr[2])


def _as_quat(msg: Quaternion, a: np.ndarray | Sequence[float]) -> None:
    arr = np.asarray(a, dtype=float)
    if arr.shape != (4,):
        raise ValueError(f"expected shape (4,), got {arr.shape} (order: [w, x, y, z])")
    msg.w = float(arr[0])
    msg.x = float(arr[1])
    msg.y = float(arr[2])
    msg.z = float(arr[3])


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return the quaternion ``[w, x, y, z]`` for a roll/pitch/yaw (ZYX) triple."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def quaternion_to_euler(q: np.ndarray | Sequence[float]) -> tuple[float, float, float]:
    """Inverse of :func:`euler_to_quaternion`. Returns (roll, pitch, yaw)."""
    w, x, y, z = (float(v) for v in q)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class MCAPRecorder:
    """Write a quadrotor flight log to an MCAP file.

    Parameters mirror the C++ ``mav_flight_mcap::MCAPRecorder`` exactly so both
    recorders produce interchangeable output.
    """

    TOPIC_STATE = "/drone/state"
    TOPIC_REFERENCE = "/drone/reference"
    TOPIC_ACTUATION = "/drone/actuation"
    TOPIC_EXTRA_PREFIX = "/drone/extras/"

    def __init__(
        self,
        file_path: str | Path,
        n_motors: int = 4,
        extra_fields: Sequence[str] = (),
    ) -> None:
        if n_motors < 0:
            raise ValueError("n_motors must be >= 0")
        self.file_path = Path(file_path)
        self.n_motors = int(n_motors)
        self.extra_fields: list[str] = list(extra_fields)
        if len(set(self.extra_fields)) != len(self.extra_fields):
            raise ValueError("extra_fields must be unique")

        self._fh: IO[bytes] = open(self.file_path, "wb")
        self._writer = ProtobufWriter(self._fh)
        self._closed = False

    def __enter__(self) -> "MCAPRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._writer.finish()
        self._fh.close()
        self._closed = True

    def save(
        self,
        time: float,
        position: np.ndarray,
        orientation: np.ndarray,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
        reference_position: np.ndarray,
        reference_velocity: np.ndarray,
        reference_orientation: np.ndarray,
        reference_angular_velocity: np.ndarray,
        thrust: float,
        command_angular_velocity: np.ndarray,
        motor_angular_velocity: np.ndarray,
        extras: Mapping[str, float] | Sequence[float] | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("MCAPRecorder is closed")

        motor_w = np.asarray(motor_angular_velocity, dtype=float).reshape(-1)
        if motor_w.size != self.n_motors:
            raise ValueError(
                f"motor_angular_velocity has {motor_w.size} elements, expected {self.n_motors}"
            )

        t_ns = int(time * 1e9)

        state = State(time=float(time))
        _as_vec3(state.position, position)
        _as_quat(state.orientation, orientation)
        _as_vec3(state.linear_velocity, linear_velocity)
        _as_vec3(state.angular_velocity, angular_velocity)
        self._writer.write_message(
            topic=self.TOPIC_STATE,
            message=state,
            log_time=t_ns,
            publish_time=t_ns,
        )

        reference = Reference(time=float(time))
        _as_vec3(reference.position, reference_position)
        _as_vec3(reference.linear_velocity, reference_velocity)
        _as_quat(reference.orientation, reference_orientation)
        _as_vec3(reference.angular_velocity, reference_angular_velocity)
        self._writer.write_message(
            topic=self.TOPIC_REFERENCE,
            message=reference,
            log_time=t_ns,
            publish_time=t_ns,
        )

        actuation = Actuation(time=float(time), thrust=float(thrust))
        _as_vec3(actuation.command_angular_velocity, command_angular_velocity)
        actuation.motor_angular_velocity.extend(motor_w.tolist())
        self._writer.write_message(
            topic=self.TOPIC_ACTUATION,
            message=actuation,
            log_time=t_ns,
            publish_time=t_ns,
        )

        for name, value in self._resolve_extras(extras):
            msg = Scalar(time=float(time), value=float(value))
            self._writer.write_message(
                topic=f"{self.TOPIC_EXTRA_PREFIX}{name}",
                message=msg,
                log_time=t_ns,
                publish_time=t_ns,
            )

    def _resolve_extras(
        self, extras: Mapping[str, float] | Sequence[float] | None
    ) -> list[tuple[str, float]]:
        if not self.extra_fields:
            if extras:
                raise ValueError("extras provided but no extra_fields declared")
            return []

        if extras is None:
            return [(name, math.nan) for name in self.extra_fields]

        if isinstance(extras, Mapping):
            unknown = set(extras).difference(self.extra_fields)
            if unknown:
                raise ValueError(f"unknown extra fields: {sorted(unknown)}")
            return [(name, float(extras.get(name, math.nan))) for name in self.extra_fields]

        seq = list(extras)
        if len(seq) != len(self.extra_fields):
            raise ValueError(
                f"extras sequence length {len(seq)} does not match extra_fields length "
                f"{len(self.extra_fields)}"
            )
        return list(zip(self.extra_fields, (float(v) for v in seq)))
