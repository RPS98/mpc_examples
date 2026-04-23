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

"""Tabular data model for visualising ROS 2-compatible flight MCAPs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

from .mcap_reader import FlightBag


@dataclass
class TopicMap:
    """Maps the plotter's roles to actual MCAP topic names."""

    pose_state: str = '/drone0/self_localization/pose'
    twist_state: str = '/drone0/self_localization/twist'
    odom_state: str = '/drone0/sensor_measurements/odom'
    pose_reference: str = '/drone0/motion_reference/pose'
    twist_reference: str = '/drone0/motion_reference/twist'
    thrust_command: str = '/drone0/actuator_command/thrust'
    twist_command: str = '/drone0/actuator_command/twist'


@dataclass
class Series:
    """A single 1-D time series (aligned to FlightData.time)."""

    t: np.ndarray
    y: np.ndarray
    label: str


@dataclass
class FlightData:
    """All relevant time series extracted from a flight MCAP."""

    path: Path
    topics: Dict[str, str] = field(default_factory=dict)
    series: Dict[str, List[Series]] = field(default_factory=dict)
    extras: Dict[str, Series] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str, mapping: TopicMap | None = None) -> 'FlightData':
        """Read the MCAP, decode every message and aggregate per topic role."""
        p = Path(path)
        m = mapping or TopicMap()
        bag = FlightBag(p)
        all_topics = bag.topics()
        self = cls(path=p, topics=all_topics)

        # Raw lists keyed by role.
        pose_state: List[tuple] = []
        pose_ref: List[tuple] = []
        twist_state: List[tuple] = []
        twist_ref: List[tuple] = []
        twist_cmd: List[tuple] = []
        thrust: List[tuple] = []
        extras_raw: Dict[str, List[tuple]] = {}

        # Extras = everything whose schema is std_msgs/* or geometry_msgs/Vector3
        # and whose topic is not one of the well-known roles.
        role_topics = {
            m.pose_state, m.twist_state, m.odom_state,
            m.pose_reference, m.twist_reference,
            m.thrust_command, m.twist_command, '/clock',
        }

        for msg in bag.iter_messages():
            t = msg.log_time_ns * 1e-9
            if msg.topic == m.pose_state:
                pos = msg.payload['pose']['position']
                pose_state.append((t, pos['x'], pos['y'], pos['z']))
            elif msg.topic == m.pose_reference:
                pos = msg.payload['pose']['position']
                pose_ref.append((t, pos['x'], pos['y'], pos['z']))
            elif msg.topic == m.twist_state:
                lin = msg.payload['twist']['linear']
                ang = msg.payload['twist']['angular']
                twist_state.append((t, lin['x'], lin['y'], lin['z'],
                                    ang['x'], ang['y'], ang['z']))
            elif msg.topic == m.twist_reference:
                lin = msg.payload['twist']['linear']
                twist_ref.append((t, lin['x'], lin['y'], lin['z']))
            elif msg.topic == m.twist_command:
                ang = msg.payload['twist']['angular']
                twist_cmd.append((t, ang['x'], ang['y'], ang['z']))
            elif msg.topic == m.thrust_command:
                thrust.append((t, msg.payload['thrust']))
            elif msg.topic == '/clock':
                continue
            elif msg.topic == m.odom_state:
                continue  # redundant with pose_state+twist_state.
            else:
                extras_raw.setdefault(msg.topic, []).append((t, msg))

        def to_arrays(rows: list) -> np.ndarray:
            return np.asarray(rows, dtype=float) if rows else np.empty((0, 0))

        ps = to_arrays(pose_state)
        pr = to_arrays(pose_ref)
        ts = to_arrays(twist_state)
        tr = to_arrays(twist_ref)
        tc = to_arrays(twist_cmd)
        th = to_arrays(thrust)

        self.series = {
            'pose_state_x':    [Series(ps[:, 0], ps[:, 1], 'x')] if ps.size else [],
            'pose_state_y':    [Series(ps[:, 0], ps[:, 2], 'y')] if ps.size else [],
            'pose_state_z':    [Series(ps[:, 0], ps[:, 3], 'z')] if ps.size else [],
            'pose_ref_x':      [Series(pr[:, 0], pr[:, 1], 'x_ref')] if pr.size else [],
            'pose_ref_y':      [Series(pr[:, 0], pr[:, 2], 'y_ref')] if pr.size else [],
            'pose_ref_z':      [Series(pr[:, 0], pr[:, 3], 'z_ref')] if pr.size else [],
            'twist_state_vx':  [Series(ts[:, 0], ts[:, 1], 'vx')] if ts.size else [],
            'twist_state_vy':  [Series(ts[:, 0], ts[:, 2], 'vy')] if ts.size else [],
            'twist_state_vz':  [Series(ts[:, 0], ts[:, 3], 'vz')] if ts.size else [],
            'twist_state_wx':  [Series(ts[:, 0], ts[:, 4], 'wx')] if ts.size else [],
            'twist_state_wy':  [Series(ts[:, 0], ts[:, 5], 'wy')] if ts.size else [],
            'twist_state_wz':  [Series(ts[:, 0], ts[:, 6], 'wz')] if ts.size else [],
            'twist_ref_vmax':  [Series(tr[:, 0],
                                       np.linalg.norm(tr[:, 1:4], axis=1),
                                       '|v|_max')] if tr.size else [],
            'twist_cmd_wx':    [Series(tc[:, 0], tc[:, 1], 'wx_cmd')] if tc.size else [],
            'twist_cmd_wy':    [Series(tc[:, 0], tc[:, 2], 'wy_cmd')] if tc.size else [],
            'twist_cmd_wz':    [Series(tc[:, 0], tc[:, 3], 'wz_cmd')] if tc.size else [],
            'thrust':          [Series(th[:, 0], th[:, 1], 'thrust')] if th.size else [],
        }
        # Linear-speed modulus is explicitly requested by the spec.
        if ts.size:
            self.series['twist_state_vnorm'] = [
                Series(ts[:, 0], np.linalg.norm(ts[:, 1:4], axis=1), '||v||')
            ]

        # Decode numeric extras so they can be plotted too.
        for topic, rows in extras_raw.items():
            if topic in role_topics:
                continue
            schema = all_topics.get(topic, '')
            y_vals: List[float] = []
            t_vals: List[float] = []
            for t, msg in rows:
                val = _extract_scalar(msg.schema_name, msg.payload)
                if val is None:
                    continue
                t_vals.append(t)
                y_vals.append(val)
            if y_vals:
                self.extras[topic] = Series(
                    np.asarray(t_vals), np.asarray(y_vals), f'{topic} ({schema})')

        return self

    # --- Convenience ----------------------------------------------------------

    @property
    def time_origin_s(self) -> float:
        """Lowest timestamp across all recorded series (for axis zeroing)."""
        ts = []
        for lst in self.series.values():
            for s in lst:
                if s.t.size:
                    ts.append(float(s.t[0]))
        for s in self.extras.values():
            if s.t.size:
                ts.append(float(s.t[0]))
        return min(ts) if ts else 0.0


def _extract_scalar(schema_name: str, payload: Dict) -> float | None:
    """Try to reduce a decoded payload to a single float for plotting."""
    if schema_name in ('std_msgs/msg/Int32', 'std_msgs/msg/Float64'):
        return float(payload.get('data', math.nan))
    if schema_name == 'geometry_msgs/msg/Vector3':
        return float(math.sqrt(payload['x'] ** 2 +
                               payload['y'] ** 2 +
                               payload['z'] ** 2))
    if schema_name == 'std_msgs/msg/Float64MultiArray':
        data = payload.get('data', [])
        return float(data[0]) if data else None
    return None
