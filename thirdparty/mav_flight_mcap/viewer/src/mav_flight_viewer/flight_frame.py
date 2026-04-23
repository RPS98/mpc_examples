# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Decode a mpc_examples MCAP into a flat ``dict`` of numpy arrays.

The resulting dictionary uses the same keys the retired CSV logger produced
(``time``, ``x``, ``y``, ``z``, ``x_ref``, ..., ``controller_compute_time_us``,
``motor_w0``, ``waypoint_index``, ``hover_active``, ``max_speed``, ...) so the
metrics / dashboard pipeline can stay almost identical once ported here.

Assumptions:
  * The MCAP was produced by ``UnifiedMcapLogger`` (or the equivalent Python
    facade), which emits every relevant save_* call with the same timestamp
    per outer-loop step.
  * All topics except ``/mpc_examples/metadata/*`` have the same length.
    When lengths diverge (e.g. mid-run truncation) every array is clipped to
    the minimum length so the resulting frame stays rectangular.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .mcap_reader import FlightBag


# Canonical topic names emitted by UnifiedMcapLogger. Kept in sync with the
# constants in examples_cpp/src/framework/unified_mcap_logger.cpp and
# examples_py/examples_py/framework/unified_mcap_logger.py.
_TOPIC_POSE_STATE = '/drone0/self_localization/pose'
_TOPIC_TWIST_STATE = '/drone0/self_localization/twist'
_TOPIC_POSE_REF = '/drone0/motion_reference/pose'
_TOPIC_THRUST = '/drone0/actuator_command/thrust'
_TOPIC_TWIST_CMD = '/drone0/actuator_command/twist'
_TOPIC_MOTOR_SPEEDS = '/drone0/actuator_command/motor_speeds'

_TOPIC_CTRL_COMPUTE = '/mpc_examples/controller_compute_time_us'
_TOPIC_GEN_UPDATE = '/mpc_examples/generator_update_time_us'
_TOPIC_GEN_EVAL = '/mpc_examples/generator_eval_time_us'
_TOPIC_CTRL_DELAY = '/mpc_examples/controller_delay_applied_us'
_TOPIC_GEN_DELAY = '/mpc_examples/generator_delay_applied_us'
_TOPIC_MAX_SPEED = '/mpc_examples/max_speed'
_TOPIC_WAYPOINT_INDEX = '/mpc_examples/waypoint_index'
_TOPIC_HOVER_ACTIVE = '/mpc_examples/hover_active'

_TOPIC_META_CONTROLLER = '/mpc_examples/metadata/controller_name'
_TOPIC_META_GENERATOR = '/mpc_examples/metadata/generator_name'
_TOPIC_META_RUN_ID = '/mpc_examples/metadata/run_id'
_TOPIC_META_LANGUAGE = '/mpc_examples/metadata/language'


@dataclass
class FlightFrameMetadata:
    """Metadata topics emitted once at t=0 by ``UnifiedMcapLogger``."""

    controller_name: str = ''
    generator_name: str = ''
    run_id: str = ''
    language: str = ''


@dataclass
class FlightFrame:
    """Flat dictionary-of-arrays representation of an mpc_examples MCAP."""

    path: Path
    data: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: FlightFrameMetadata = field(default_factory=FlightFrameMetadata)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str, default=None):
        return self.data.get(key, default)


def _quat_to_euler(qw: float, qx: float, qy: float, qz: float) -> tuple:
    """Convert ``[w, x, y, z]`` quaternion to intrinsic ZYX Euler angles (rad)."""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _euler_array(qw: np.ndarray, qx: np.ndarray,
                 qy: np.ndarray, qz: np.ndarray) -> tuple:
    """Vectorised ``_quat_to_euler`` for same-shape arrays."""
    if len(qw) == 0:
        empty = np.empty(0, dtype=float)
        return empty, empty, empty
    rolls = np.empty_like(qw, dtype=float)
    pitches = np.empty_like(qw, dtype=float)
    yaws = np.empty_like(qw, dtype=float)
    for i in range(len(qw)):
        r, p, y = _quat_to_euler(float(qw[i]), float(qx[i]),
                                 float(qy[i]), float(qz[i]))
        rolls[i] = r
        pitches[i] = p
        yaws[i] = y
    return rolls, pitches, yaws


def _clip_to_min_len(arrays: List[np.ndarray]) -> int:
    """Return the minimum length across ``arrays`` (0 if the list is empty)."""
    if not arrays:
        return 0
    return min(len(a) for a in arrays)


def _collect_scalar(rows: List[tuple], field_name: str) -> np.ndarray:
    """Extract ``(t, payload[field])`` pairs into a ``(t, value)`` numpy pair."""
    if not rows:
        return np.empty((0, 2), dtype=float)
    ts = np.fromiter((r[0] for r in rows), dtype=float, count=len(rows))
    vs = np.fromiter((float(r[1].payload.get(field_name, math.nan)) for r in rows),
                     dtype=float, count=len(rows))
    return np.column_stack([ts, vs])


def _collect_multi_array_first_n(rows: List[tuple], n: int) -> np.ndarray:
    """Extract the first ``n`` entries of each Float64MultiArray payload."""
    if not rows:
        return np.empty((0, 1 + n), dtype=float)
    ts = np.fromiter((r[0] for r in rows), dtype=float, count=len(rows))
    vals = np.empty((len(rows), n), dtype=float)
    for i, (_, msg) in enumerate(rows):
        data = msg.payload.get('data', [])
        for j in range(n):
            vals[i, j] = float(data[j]) if j < len(data) else math.nan
    return np.column_stack([ts, vals])


def load_flight_frame(path: Path | str) -> FlightFrame:
    """Parse ``path`` and return the per-column arrays used by the metrics."""
    p = Path(path)
    bag = FlightBag(p)

    pose_state: List[tuple] = []    # (t, x, y, z, qw, qx, qy, qz)
    twist_state: List[tuple] = []   # (t, vx, vy, vz, wx, wy, wz)
    pose_ref: List[tuple] = []      # (t, x, y, z, qw, qx, qy, qz)
    thrust: List[tuple] = []        # (t, thrust_n)
    twist_cmd: List[tuple] = []     # (t, wx, wy, wz)
    motor_rows: List[tuple] = []
    ctrl_compute: List[tuple] = []
    gen_update: List[tuple] = []
    gen_eval: List[tuple] = []
    ctrl_delay: List[tuple] = []
    gen_delay: List[tuple] = []
    max_speed: List[tuple] = []
    waypoint_idx: List[tuple] = []
    hover: List[tuple] = []
    metadata = FlightFrameMetadata()

    def _store_scalar(bucket: List[tuple], msg) -> None:
        bucket.append((msg.log_time_ns * 1e-9, msg))

    for msg in bag.iter_messages():
        topic = msg.topic
        t = msg.log_time_ns * 1e-9
        if topic == _TOPIC_POSE_STATE:
            pos = msg.payload['pose']['position']
            ori = msg.payload['pose']['orientation']
            pose_state.append((t, pos['x'], pos['y'], pos['z'],
                               ori['w'], ori['x'], ori['y'], ori['z']))
        elif topic == _TOPIC_POSE_REF:
            pos = msg.payload['pose']['position']
            ori = msg.payload['pose']['orientation']
            pose_ref.append((t, pos['x'], pos['y'], pos['z'],
                             ori['w'], ori['x'], ori['y'], ori['z']))
        elif topic == _TOPIC_TWIST_STATE:
            lin = msg.payload['twist']['linear']
            ang = msg.payload['twist']['angular']
            twist_state.append((t, lin['x'], lin['y'], lin['z'],
                                ang['x'], ang['y'], ang['z']))
        elif topic == _TOPIC_TWIST_CMD:
            ang = msg.payload['twist']['angular']
            twist_cmd.append((t, ang['x'], ang['y'], ang['z']))
        elif topic == _TOPIC_THRUST:
            thrust.append((t, float(msg.payload.get('thrust', math.nan))))
        elif topic == _TOPIC_MOTOR_SPEEDS:
            _store_scalar(motor_rows, msg)
        elif topic == _TOPIC_CTRL_COMPUTE:
            _store_scalar(ctrl_compute, msg)
        elif topic == _TOPIC_GEN_UPDATE:
            _store_scalar(gen_update, msg)
        elif topic == _TOPIC_GEN_EVAL:
            _store_scalar(gen_eval, msg)
        elif topic == _TOPIC_CTRL_DELAY:
            _store_scalar(ctrl_delay, msg)
        elif topic == _TOPIC_GEN_DELAY:
            _store_scalar(gen_delay, msg)
        elif topic == _TOPIC_MAX_SPEED:
            _store_scalar(max_speed, msg)
        elif topic == _TOPIC_WAYPOINT_INDEX:
            _store_scalar(waypoint_idx, msg)
        elif topic == _TOPIC_HOVER_ACTIVE:
            _store_scalar(hover, msg)
        elif topic == _TOPIC_META_CONTROLLER:
            metadata.controller_name = str(msg.payload.get('data', ''))
        elif topic == _TOPIC_META_GENERATOR:
            metadata.generator_name = str(msg.payload.get('data', ''))
        elif topic == _TOPIC_META_RUN_ID:
            metadata.run_id = str(msg.payload.get('data', ''))
        elif topic == _TOPIC_META_LANGUAGE:
            metadata.language = str(msg.payload.get('data', ''))
        else:
            continue

    ps = np.asarray(pose_state, dtype=float) if pose_state else np.empty((0, 8))
    ts = np.asarray(twist_state, dtype=float) if twist_state else np.empty((0, 7))
    pr = np.asarray(pose_ref, dtype=float) if pose_ref else np.empty((0, 8))
    th = np.asarray(thrust, dtype=float) if thrust else np.empty((0, 2))
    tc = np.asarray(twist_cmd, dtype=float) if twist_cmd else np.empty((0, 4))

    mw = _collect_multi_array_first_n(motor_rows, 4)
    cc = _collect_scalar(ctrl_compute, 'data')
    gu = _collect_scalar(gen_update, 'data')
    ge = _collect_scalar(gen_eval, 'data')
    cd = _collect_scalar(ctrl_delay, 'data')
    gd = _collect_scalar(gen_delay, 'data')
    ms = _collect_scalar(max_speed, 'data')
    wi = _collect_scalar(waypoint_idx, 'data')
    ha = _collect_scalar(hover, 'data')

    # Align all columns to the shortest log so the caller can trust len(time)
    # as the row count. Missing topics appear as empty arrays and are dropped.
    row_count_candidates = [len(a) for a in (ps, ts, pr, th, tc, mw, cc, gu, ge,
                                             cd, gd, ms, wi, ha) if len(a) > 0]
    n = min(row_count_candidates) if row_count_candidates else 0

    def _clip(arr: np.ndarray) -> np.ndarray:
        return arr[:n] if len(arr) >= n else arr

    ps = _clip(ps)
    ts = _clip(ts)
    pr = _clip(pr)
    th = _clip(th)
    tc = _clip(tc)
    mw = _clip(mw)
    cc = _clip(cc)
    gu = _clip(gu)
    ge = _clip(ge)
    cd = _clip(cd)
    gd = _clip(gd)
    ms = _clip(ms)
    wi = _clip(wi)
    ha = _clip(ha)

    data: Dict[str, np.ndarray] = {}

    if ps.size:
        data['time'] = ps[:, 0]
        data['x'] = ps[:, 1]
        data['y'] = ps[:, 2]
        data['z'] = ps[:, 3]
        data['qw'] = ps[:, 4]
        data['qx'] = ps[:, 5]
        data['qy'] = ps[:, 6]
        data['qz'] = ps[:, 7]
        rolls, pitches, yaws = _euler_array(ps[:, 4], ps[:, 5], ps[:, 6], ps[:, 7])
        data['roll'] = rolls
        data['pitch'] = pitches
        data['yaw'] = yaws

    if ts.size:
        data.setdefault('time', ts[:, 0])
        data['vx'] = ts[:, 1]
        data['vy'] = ts[:, 2]
        data['vz'] = ts[:, 3]
        data['wx'] = ts[:, 4]
        data['wy'] = ts[:, 5]
        data['wz'] = ts[:, 6]

    if pr.size:
        data['x_ref'] = pr[:, 1]
        data['y_ref'] = pr[:, 2]
        data['z_ref'] = pr[:, 3]
        data['qw_ref'] = pr[:, 4]
        data['qx_ref'] = pr[:, 5]
        data['qy_ref'] = pr[:, 6]
        data['qz_ref'] = pr[:, 7]
        rolls_r, pitches_r, yaws_r = _euler_array(
            pr[:, 4], pr[:, 5], pr[:, 6], pr[:, 7])
        data['roll_ref'] = rolls_r
        data['pitch_ref'] = pitches_r
        data['yaw_ref'] = yaws_r

    if th.size:
        data['thrust'] = th[:, 1]
    if tc.size:
        data['wx_cmd'] = tc[:, 1]
        data['wy_cmd'] = tc[:, 2]
        data['wz_cmd'] = tc[:, 3]
    if mw.size:
        data['motor_w0'] = mw[:, 1]
        data['motor_w1'] = mw[:, 2]
        data['motor_w2'] = mw[:, 3]
        data['motor_w3'] = mw[:, 4]

    if cc.size:
        data['controller_compute_time_us'] = cc[:, 1]
    if gu.size:
        data['generator_update_time_us'] = gu[:, 1]
    if ge.size:
        data['generator_eval_time_us'] = ge[:, 1]
    if cd.size:
        data['controller_delay_applied_us'] = cd[:, 1]
    if gd.size:
        data['generator_delay_applied_us'] = gd[:, 1]
    if ms.size:
        data['max_speed'] = ms[:, 1]
    if wi.size:
        data['waypoint_index'] = wi[:, 1]
    if ha.size:
        data['hover_active'] = ha[:, 1]

    return FlightFrame(path=p, data=data, metadata=metadata)


__all__ = ['FlightFrame', 'FlightFrameMetadata', 'load_flight_frame']
