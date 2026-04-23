"""Read an MCAP flight log and stream it to the rerun.io viewer.

Subclass-friendly design: override the ``_log_state`` / ``_log_reference`` /
``_log_actuation`` / ``_log_extra`` hooks to customize what gets logged, or
override :meth:`blueprint` to change the dashboard.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from ._drone_mesh import DroneGeometry, log_drone_mesh
from .dashboard import default_blueprint

__all__ = ["MCAPViewer"]


TIMELINE = "flight_time"


def _quat_wxyz_to_xyzw(w: float, x: float, y: float, z: float) -> list[float]:
    """Rerun expects quaternions as (x, y, z, w)."""
    return [float(x), float(y), float(z), float(w)]


def _quat_to_rpy(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class MCAPViewer:
    """Decode an MCAP flight log and log it into the rerun SDK.

    Parameters
    ----------
    mcap_path:
        Path to an ``.mcap`` file produced by :class:`mav_flight_mcap.MCAPRecorder`.
    label:
        Rerun application id shown in the viewer title. Defaults to the file stem.
    """

    def __init__(self, mcap_path: str | Path, label: str | None = None) -> None:
        self.mcap_path = Path(mcap_path)
        if not self.mcap_path.is_file():
            raise FileNotFoundError(self.mcap_path)
        self.label = label or self.mcap_path.stem
        self._trajectory: list[list[float]] = []
        self._reference_trajectory: list[list[float]] = []

    # ----- public API -------------------------------------------------------

    def show(self, spawn: bool = True) -> None:
        """Open the rerun viewer and stream the log into it."""
        rr.init(self.label, spawn=spawn, default_blueprint=self.blueprint())
        self._log_static()
        self._stream()

    def save(self, rrd_path: str | Path) -> Path:
        """Write the recording to an ``.rrd`` file without opening the viewer."""
        out = Path(rrd_path)
        rr.init(self.label, spawn=False, default_blueprint=self.blueprint())
        self._log_static()
        self._stream()
        rr.save(out)
        return out

    def blueprint(self) -> rrb.Blueprint:
        """Return the rerun blueprint for the dashboard. Override to customize."""
        return default_blueprint()

    # ----- hooks ------------------------------------------------------------

    def _log_static(self) -> None:
        """Log world frame + drone mesh (time-independent geometry)."""
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log(
            "world/axes",
            rr.Arrows3D(
                origins=[[0.0, 0.0, 0.0]] * 3,
                vectors=[[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]],
                colors=[(230, 40, 40), (40, 200, 40), (40, 40, 230)],
                radii=0.004,
            ),
            static=True,
        )
        log_drone_mesh("world/drone/body", n_motors=4)
        log_drone_mesh("world/reference/body", n_motors=4)
        rr.log(
            "world/reference/body/arms",
            rr.LineStrips3D(
                [
                    np.stack([np.zeros(3), p])
                    for p in DroneGeometry.motor_positions(4)
                ],
                colors=[(120, 120, 220)],
                radii=0.006,
            ),
            static=True,
        )

    def _stream(self) -> None:
        for topic, proto_msg in self._iter_messages():
            t = float(proto_msg.time)
            rr.set_time(TIMELINE, duration=t)

            if topic == "/drone/state":
                self._log_state(proto_msg)
            elif topic == "/drone/reference":
                self._log_reference(proto_msg)
            elif topic == "/drone/actuation":
                self._log_actuation(proto_msg)
            elif topic.startswith("/drone/extras/"):
                name = topic[len("/drone/extras/") :]
                self._log_extra(name, proto_msg)

    def _iter_messages(self) -> Iterator[tuple[str, Any]]:
        with open(self.mcap_path, "rb") as fh:
            reader = make_reader(fh, decoder_factories=[DecoderFactory()])
            for _schema, channel, _message, proto in reader.iter_decoded_messages():
                yield channel.topic, proto

    # ----- per-message handlers (override-friendly) ------------------------

    def _log_state(self, state: Any) -> None:
        p = state.position
        q = state.orientation
        v = state.linear_velocity
        w = state.angular_velocity

        rr.log(
            "world/drone",
            rr.Transform3D(
                translation=[p.x, p.y, p.z],
                quaternion=_quat_wxyz_to_xyzw(q.w, q.x, q.y, q.z),
            ),
        )
        rr.log(
            "world/drone/velocity",
            rr.Arrows3D(
                origins=[[0.0, 0.0, 0.0]],
                vectors=[[v.x, v.y, v.z]],
                colors=[(30, 180, 30)],
                radii=0.01,
            ),
        )

        self._trajectory.append([p.x, p.y, p.z])
        if len(self._trajectory) >= 2:
            rr.log(
                "world/trajectory",
                rr.LineStrips3D(
                    [np.asarray(self._trajectory, dtype=float)],
                    colors=[(100, 200, 100)],
                    radii=0.005,
                ),
            )

        speed = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
        wnorm = math.sqrt(w.x * w.x + w.y * w.y + w.z * w.z)
        pnorm = math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
        roll, pitch, yaw = _quat_to_rpy(q.w, q.x, q.y, q.z)

        _scalar("scalars/position/x", p.x)
        _scalar("scalars/position/y", p.y)
        _scalar("scalars/position/z", p.z)
        _scalar("scalars/position/norm", pnorm)

        _scalar("scalars/velocity/vx", v.x)
        _scalar("scalars/velocity/vy", v.y)
        _scalar("scalars/velocity/vz", v.z)
        _scalar("scalars/velocity/speed", speed)

        _scalar("scalars/angular_velocity/wx", w.x)
        _scalar("scalars/angular_velocity/wy", w.y)
        _scalar("scalars/angular_velocity/wz", w.z)
        _scalar("scalars/angular_velocity/norm", wnorm)

        _scalar("scalars/rpy/roll", roll)
        _scalar("scalars/rpy/pitch", pitch)
        _scalar("scalars/rpy/yaw", yaw)

        # Keep latest position to compute tracking error when we see reference.
        self._last_position = (p.x, p.y, p.z)
        self._last_velocity = (v.x, v.y, v.z)

    def _log_reference(self, ref: Any) -> None:
        p = ref.position
        v = ref.linear_velocity
        q = ref.orientation
        w = ref.angular_velocity

        rr.log(
            "world/reference",
            rr.Transform3D(
                translation=[p.x, p.y, p.z],
                quaternion=_quat_wxyz_to_xyzw(q.w, q.x, q.y, q.z),
            ),
        )
        self._reference_trajectory.append([p.x, p.y, p.z])
        if len(self._reference_trajectory) >= 2:
            rr.log(
                "world/reference/trajectory",
                rr.LineStrips3D(
                    [np.asarray(self._reference_trajectory, dtype=float)],
                    colors=[(120, 120, 220)],
                    radii=0.003,
                ),
            )

        roll, pitch, yaw = _quat_to_rpy(q.w, q.x, q.y, q.z)

        _scalar("scalars/position/x_ref", p.x)
        _scalar("scalars/position/y_ref", p.y)
        _scalar("scalars/position/z_ref", p.z)

        _scalar("scalars/velocity/vx_ref", v.x)
        _scalar("scalars/velocity/vy_ref", v.y)
        _scalar("scalars/velocity/vz_ref", v.z)

        _scalar("scalars/angular_velocity/wx_ref", w.x)
        _scalar("scalars/angular_velocity/wy_ref", w.y)
        _scalar("scalars/angular_velocity/wz_ref", w.z)

        _scalar("scalars/rpy/roll_ref", roll)
        _scalar("scalars/rpy/pitch_ref", pitch)
        _scalar("scalars/rpy/yaw_ref", yaw)

        last_p = getattr(self, "_last_position", None)
        last_v = getattr(self, "_last_velocity", None)
        if last_p is not None:
            ex, ey, ez = last_p[0] - p.x, last_p[1] - p.y, last_p[2] - p.z
            _scalar("scalars/error/position", math.sqrt(ex * ex + ey * ey + ez * ez))
        if last_v is not None:
            ex, ey, ez = last_v[0] - v.x, last_v[1] - v.y, last_v[2] - v.z
            _scalar("scalars/error/velocity", math.sqrt(ex * ex + ey * ey + ez * ez))

    def _log_actuation(self, act: Any) -> None:
        w = act.command_angular_velocity
        _scalar("scalars/actuation/thrust", act.thrust)
        _scalar("scalars/actuation/wx_cmd", w.x)
        _scalar("scalars/actuation/wy_cmd", w.y)
        _scalar("scalars/actuation/wz_cmd", w.z)
        for i, motor_w in enumerate(act.motor_angular_velocity):
            _scalar(f"scalars/motors/motor_{i}", motor_w)

    def _log_extra(self, name: str, scalar: Any) -> None:
        _scalar(f"scalars/extras/{name}", scalar.value)


def _scalar(entity_path: str, value: float) -> None:
    rr.log(entity_path, rr.Scalars(float(value)))
