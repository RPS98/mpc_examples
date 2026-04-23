"""mav_flight_mcap — MCAP logger for quadrotor flight data (C++/Python parity)."""

from __future__ import annotations

from .recorder import MCAPRecorder, euler_to_quaternion, quaternion_to_euler
from ._proto.mav_flight_pb2 import (
    Actuation,
    Quaternion,
    Reference,
    Scalar,
    State,
    Vec3,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "MCAPRecorder",
    "euler_to_quaternion",
    "quaternion_to_euler",
    "State",
    "Reference",
    "Actuation",
    "Scalar",
    "Vec3",
    "Quaternion",
]
