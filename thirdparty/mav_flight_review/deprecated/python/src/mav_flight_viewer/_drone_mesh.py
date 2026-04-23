"""Procedural placeholder mesh for the quadrotor in the 3D world view.

The mesh is logged as static geometry under a single entity path and inherits
its parent's time-varying :class:`rerun.Transform3D` so it moves with the drone.
No external assets required.
"""

from __future__ import annotations

import numpy as np
import rerun as rr

__all__ = ["log_drone_mesh", "DroneGeometry"]


class DroneGeometry:
    """Geometry constants for the placeholder quadrotor mesh."""

    ARM_LENGTH: float = 0.20      # [m] horizontal distance to each motor
    MOTOR_RADIUS: float = 0.055   # [m] motor disc radius
    AXES_LENGTH: float = 0.18     # [m] body-frame arrow length

    # X-configuration motor layout (body frame).
    MOTOR_XY = np.array(
        [
            [+1.0, +1.0],
            [+1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, +1.0],
        ]
    )

    @classmethod
    def motor_positions(cls, n_motors: int = 4) -> np.ndarray:
        """Return motor positions in the body frame (z=0 plane)."""
        if n_motors == 4:
            return np.hstack(
                [cls.MOTOR_XY * cls.ARM_LENGTH, np.zeros((4, 1))]
            )
        # Generic circular layout for other motor counts.
        angles = np.linspace(0.0, 2.0 * np.pi, n_motors, endpoint=False)
        return np.stack(
            [
                cls.ARM_LENGTH * np.cos(angles),
                cls.ARM_LENGTH * np.sin(angles),
                np.zeros(n_motors),
            ],
            axis=1,
        )


def _ring(center: np.ndarray, radius: float, n: int = 24) -> np.ndarray:
    """Return a closed ring as an (n+1, 3) polyline in the XY plane at z=center[2]."""
    theta = np.linspace(0.0, 2.0 * np.pi, n + 1)
    pts = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)],
        axis=1,
    )
    return pts + center[None, :]


def log_drone_mesh(entity_path: str = "world/drone/body", n_motors: int = 4) -> None:
    """Log the placeholder drone mesh as static geometry under ``entity_path``.

    The parent of ``entity_path`` is expected to receive a time-varying
    :class:`rerun.Transform3D` for the drone pose; this mesh inherits that
    transform and moves accordingly.

    Components:

    * Body-frame axes as :class:`rerun.Arrows3D` (red X, green Y, blue Z).
    * One :class:`rerun.LineStrips3D` arm per motor (origin -> motor position).
    * One :class:`rerun.LineStrips3D` closed ring per motor (propeller disc).
    """
    positions = DroneGeometry.motor_positions(n_motors)

    arms = [np.stack([np.zeros(3), p], axis=0) for p in positions]
    rr.log(
        f"{entity_path}/arms",
        rr.LineStrips3D(arms, colors=[(160, 160, 160)], radii=0.01),
        static=True,
    )

    rings = [_ring(p, DroneGeometry.MOTOR_RADIUS) for p in positions]
    colors = np.tile(np.array([[210, 80, 80]]), (len(rings), 1))
    rr.log(
        f"{entity_path}/motors",
        rr.LineStrips3D(rings, colors=colors, radii=0.005),
        static=True,
    )

    axis_len = DroneGeometry.AXES_LENGTH
    rr.log(
        f"{entity_path}/frame",
        rr.Arrows3D(
            origins=[[0.0, 0.0, 0.0]] * 3,
            vectors=[[axis_len, 0.0, 0.0], [0.0, axis_len, 0.0], [0.0, 0.0, axis_len]],
            colors=[(230, 40, 40), (40, 200, 40), (40, 40, 230)],
            radii=0.006,
        ),
        static=True,
    )
