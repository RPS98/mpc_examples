"""Park-style fallback when a moving follow_reference target collapses onto the drone position.

Python mirror of `framework/degenerate_hold.hpp`. Mirrors aerostack2's
`generate_polynomial_trajectory_behavior` degenerate-hold semantics:

* Enter the hold when the live single waypoint sits within
  ``threshold_m`` of the vehicle pose (the polynomial solver — LBFGS in
  gcopter, the jerk-limited integrator, the acados QP — degenerates on
  near-zero displacement, so we bypass planning altogether).
* While the hold is active, publish a static horizon: every reference
  sample equals the latched target with velocity / acceleration set to
  zero and the yaw latched to the vehicle's current heading.
* Leave the hold once the target drifts back outside ``threshold_m``;
  the caller is then expected to re-arm the generator with
  ``on_waypoint_changed(target, state, t)``.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from examples_py.framework.types import ReferenceSample


# Aerostack2 default constant (see `generate_polynomial_trajectory_base.hpp:125`).
# The runtime value is read from `ExampleConfig.degenerate_distance_m`.
DEFAULT_DEGENERATE_DISTANCE_M: float = 0.05


def is_degenerate_target(
    target: np.ndarray,
    vehicle_position: np.ndarray,
    threshold_m: float,
) -> bool:
    """Return True when the live target sits within ``threshold_m`` of the vehicle.

    A non-positive threshold disables the gate.
    """
    if threshold_m <= 0.0:
        return False
    return bool(np.linalg.norm(target - vehicle_position) < threshold_m)


def quat_to_yaw(quaternion: np.ndarray) -> float:
    """Extract yaw (rad) from a quaternion ``[w, x, y, z]``."""
    w, x, y, z = (
        float(quaternion[0]),
        float(quaternion[1]),
        float(quaternion[2]),
        float(quaternion[3]),
    )
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def fill_static_horizon(
    target: np.ndarray,
    yaw_rad: float,
    refs: List[ReferenceSample],
    n_samples: int,
) -> None:
    """Overwrite an existing prediction-horizon buffer with a static reference.

    Every sample is set to ``position = target``, ``velocity = 0``,
    ``acceleration = 0`` and ``yaw = yaw_rad``.
    """
    if len(refs) != n_samples:
        refs[:] = [ReferenceSample() for _ in range(n_samples)]
    pos = np.asarray(target, dtype=float).copy()
    zero = np.zeros(3, dtype=float)
    for sample in refs:
        sample.position = pos.copy()
        sample.velocity = zero.copy()
        sample.acceleration = zero.copy()
        sample.yaw = float(yaw_rad)
