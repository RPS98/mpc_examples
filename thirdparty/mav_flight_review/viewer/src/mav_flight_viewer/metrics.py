# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Control performance metrics computed directly from an MCAP log.

Ported from the retired ``mav_flight_logger.plot_results`` module so the MCAP
pipeline does not depend on the CSV backend. The algorithms (segment
detection, rise/settling/overshoot, jerk / accel energy, speed-violation %)
match the legacy implementation column-for-column.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass
class MetricsConfig:
    """Tuneable parameters for control performance metrics."""

    # Settling time threshold (absolute, metres).
    settling_threshold_m: float = 0.1

    # Rise time: time for 3D error to drop from upper to lower % of d0.
    rise_time_upper_pct: float = 90.0
    rise_time_lower_pct: float = 10.0

    # Steady-state error window: fraction of the settled window (from t_settle
    # to segment end) used for RMSE.
    steady_state_hover_fraction: float = 1.0

    # Maximum speed for constraint-violation metric (m/s).
    max_speed: Optional[float] = None

    # Velocity smoothing window (samples) applied before computing
    # acceleration and jerk.
    smoothing_window: int = 11

    # Minimum reference displacement (m) to count as a waypoint transition.
    min_step_distance: float = 0.05


METRICS_CFG = MetricsConfig()


def _np(series) -> np.ndarray:
    return np.asarray(series, dtype=float)


def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Symmetric box-car moving average (handles edges with 'same' mode)."""
    if window <= 1:
        return arr.copy()
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


def detect_segments(data: dict, min_step_dist: float) -> list:
    """Detect reference step transitions and return a list of segment dicts."""
    x_ref = _np(data['x_ref'])
    y_ref = _np(data['y_ref'])
    z_ref = _np(data['z_ref'])
    n = len(x_ref)

    ref = np.column_stack([x_ref, y_ref, z_ref])
    ref_jump = np.linalg.norm(np.diff(ref, axis=0), axis=1)
    jump_idx = np.where(ref_jump > min_step_dist)[0] + 1
    boundaries = [0] + jump_idx.tolist() + [n]

    segments = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e - s < 2:
            continue
        segments.append({
            'start_idx': s,
            'end_idx': e,
            'ref_pos': np.array([x_ref[s], y_ref[s], z_ref[s]]),
        })
    return segments


def _compute_segment_metrics(seg: dict, t: np.ndarray, err3d: np.ndarray,
                             cfg: MetricsConfig) -> Optional[dict]:
    """Compute rise time, settling time, SSE and overshoot for one segment."""
    s, e = seg['start_idx'], seg['end_idx']
    seg_t = t[s:e]
    seg_err = err3d[s:e]

    if len(seg_t) < 3:
        return None

    d0 = max(float(seg_err[0]), 1e-6)
    settle_thr = cfg.settling_threshold_m
    upper_thr = (cfg.rise_time_upper_pct / 100.0) * d0
    lower_thr = (cfg.rise_time_lower_pct / 100.0) * d0
    t0 = float(seg_t[0])

    below_upper = np.where(seg_err <= upper_thr)[0]
    below_lower = np.where(seg_err <= lower_thr)[0]
    if len(below_upper) > 0 and len(below_lower) > 0:
        rise_time = max(0.0, float(seg_t[below_lower[0]] - seg_t[below_upper[0]]))
    else:
        rise_time = None

    below_settle = np.where(seg_err <= settle_thr)[0]
    if len(below_settle) > 0:
        settling_time = float(seg_t[below_settle[0]] - t0)
        t_settle = float(seg_t[below_settle[0]])
    else:
        settling_time = None
        t_settle = None

    if settling_time is not None:
        settled_t = seg_t[seg_t >= t_settle]
        settled_e = seg_err[seg_t >= t_settle]
        n_settled = len(settled_t)
        skip = int(n_settled * (1.0 - cfg.steady_state_hover_fraction))
        used = settled_e[skip:]
        sse = float(np.sqrt(np.mean(used ** 2))) if len(used) > 0 else None
    else:
        sse = None

    min_idx = int(np.argmin(seg_err))
    overshoot = (float(np.max(seg_err[min_idx:]) - seg_err[min_idx])
                 if min_idx < len(seg_err) - 1 else 0.0)

    return {
        't_start': t0,
        't_end': float(seg_t[-1]),
        't_settle': t_settle,
        'rise_time': rise_time,
        'settling_time': settling_time,
        'steady_state_error': sse,
        'overshoot': overshoot,
        'd0': d0,
        'settle_thr': settle_thr,
        'ref_pos': seg['ref_pos'],
    }


def _compute_3d_error(data: dict) -> np.ndarray:
    pos = np.column_stack([_np(data['x']), _np(data['y']), _np(data['z'])])
    ref = np.column_stack([_np(data['x_ref']), _np(data['y_ref']),
                           _np(data['z_ref'])])
    return np.linalg.norm(pos - ref, axis=1)


def compute_metrics(data: dict, cfg: MetricsConfig) -> dict:
    """Compute all control performance metrics from a flight frame dict.

    Matches the return schema of the legacy mav_flight_logger implementation
    (same keys, same units) so downstream CSV writers/dashboard stay aligned.
    """
    t = _np(data['time'])
    err3d = _compute_3d_error(data)

    def _stats_positive(col: str):
        if col not in data:
            return None, None, None
        arr = _np(data[col])
        valid = arr[arr > 0.0]
        if len(valid) == 0:
            return None, None, None
        return (
            float(np.mean(valid)),
            float(np.std(valid)),
            float(np.percentile(valid, 95)),
        )

    compute_time_mean, compute_time_std, compute_time_p95 = _stats_positive(
        'controller_compute_time_us')

    def _mean_positive(col: str) -> Optional[float]:
        if col not in data:
            return None
        arr = _np(data[col])
        pos = arr[arr > 0.0]
        return float(np.mean(pos)) if len(pos) > 0 else None

    gen_update_mean_us = _mean_positive('generator_update_time_us')
    gen_eval_mean_us = _mean_positive('generator_eval_time_us')
    ctrl_delay_mean_us = _mean_positive('controller_delay_applied_us')
    gen_delay_mean_us = _mean_positive('generator_delay_applied_us')

    segments = detect_segments(data, cfg.min_step_distance)
    seg_metrics = [m for seg in segments
                   for m in [_compute_segment_metrics(seg, t, err3d, cfg)]
                   if m is not None]

    def _agg(key, agg='mean'):
        vals = [s[key] for s in seg_metrics if s.get(key) is not None]
        if not vals:
            return None
        reducer = {'mean': np.mean, 'max': np.max, 'min': np.min}[agg]
        return float(reducer(vals))

    vx = _smooth(_np(data['vx']), cfg.smoothing_window)
    vy = _smooth(_np(data['vy']), cfg.smoothing_window)
    vz = _smooth(_np(data['vz']), cfg.smoothing_window)
    dt = np.diff(t)
    dt = np.where(dt > 0, dt, 1e-6)

    ax_s = np.diff(vx) / dt
    ay_s = np.diff(vy) / dt
    az_s = np.diff(vz) / dt
    accel_norm = np.sqrt(ax_s ** 2 + ay_s ** 2 + az_s ** 2)
    accel_energy = float(np.sum(accel_norm ** 2 * dt))

    jx = np.diff(ax_s) / dt[:-1]
    jy = np.diff(ay_s) / dt[:-1]
    jz = np.diff(az_s) / dt[:-1]
    jerk_norm = np.sqrt(jx ** 2 + jy ** 2 + jz ** 2)
    jerk_energy = float(np.sum(jerk_norm ** 2 * dt[:-1]))

    effective_max_speed = cfg.max_speed
    if effective_max_speed is None and 'max_speed' in data:
        csv_ms = _np(data['max_speed'])
        pos = csv_ms[csv_ms > 0.0]
        if len(pos) > 0:
            effective_max_speed = float(pos[0])

    speed = np.sqrt(_np(data['vx']) ** 2 + _np(data['vy']) ** 2 +
                    _np(data['vz']) ** 2)
    speed_violation_pct = (
        float(100.0 * np.sum(speed > effective_max_speed) / len(speed))
        if effective_max_speed is not None and len(speed) > 0 else None
    )

    tracking_rmse_m = (
        float(np.sqrt(np.mean(err3d ** 2))) if len(err3d) > 0 else None
    )

    return {
        'compute_time_mean_us': compute_time_mean,
        'compute_time_std_us': compute_time_std,
        'compute_time_p95_us': compute_time_p95,
        'generator_update_mean_us': gen_update_mean_us,
        'generator_eval_mean_us': gen_eval_mean_us,
        'controller_delay_mean_us': ctrl_delay_mean_us,
        'generator_delay_mean_us': gen_delay_mean_us,
        'tracking_rmse_m': tracking_rmse_m,
        'settling_time_s': _agg('settling_time', 'mean'),
        'settling_time_max_s': _agg('settling_time', 'max'),
        'steady_state_error_m': _agg('steady_state_error', 'mean'),
        'overshoot_m': _agg('overshoot', 'max'),
        'rise_time_mean_s': _agg('rise_time', 'mean'),
        'rise_time_min_s': _agg('rise_time', 'min'),
        'rise_time_max_s': _agg('rise_time', 'max'),
        'jerk_energy': jerk_energy,
        'accel_energy': accel_energy,
        'speed_violation_pct': speed_violation_pct,
        '_effective_max_speed': effective_max_speed,
        '_settle_thr_m': cfg.settling_threshold_m,
        'seg_metrics': seg_metrics,
        '_t': t,
        '_err3d': err3d,
        '_accel_t': t[1:],
        '_accel_norm': accel_norm,
        '_jerk_t': t[2:],
        '_jerk_norm': jerk_norm,
        '_speed': speed,
    }


__all__ = [
    'METRICS_CFG',
    'MetricsConfig',
    'compute_metrics',
    'detect_segments',
]
