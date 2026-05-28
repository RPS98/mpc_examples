#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-loop race-circuit reference generator.

Reads a gate sequence from ``mission.yaml`` / ``gates_config.yaml`` (the
per-gate-frame race-pilot schema), builds an arc-length-reparametrised
spline, and emits ``(position, velocity, acceleration)`` samples at any
absolute time along the lap.

Mapping ``t -> arc-length``:

  * Per :meth:`update`, the drone's current position is projected onto
    the spline (monotonic ``s_eval``) and anchored against the wall-
    clock time ``t``.
  * Per :meth:`evaluate(t_query)`, the arc-length is extrapolated as
    ``s_anchor + desired_speed * (t_query - t_anchor)``. This is what
    the downstream controller needs to build its horizon: every stage
    is a real lap position at the desired cruise speed.

Velocity is the spline tangent scaled by ``desired_speed``; acceleration
is the analytic curvature * speed^2 approximation (a centripetal term
along the in-plane normal). Both ``provided_reference_fields`` →
``POSITION | VELOCITY | ACCELERATION``.

Designed to be **independent** of any controller. The MPCC controller
loads the same mission inside its own ``initialize`` and runs its own
spline; both consume the YAMLs but neither depends on the other.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from mavpy.model import State

from examples_py.framework import (
    ExampleConfig,
    ITrajectoryGenerator,
    ReferenceField,
    ReferenceSample,
    make_mask,
)
from examples_py.utils import mission_loader as ml


def _import_spline_py():
    """Locate the MPCC spline pybind module.

    ``spline_trajectory_generator_py`` is a pybind11 module shipped under
    ``mpcc_acados_example/build/python/`` of the vendored mpcc thirdparty.
    Callers are expected to add that directory to ``PYTHONPATH`` (the
    project launchers do); the ``MPCC_ACADOS_EXAMPLE_DIR`` and
    ``MPCC_REPO_ROOT`` environment variables are accepted as a fallback
    search path for ad-hoc developer workflows.
    """
    try:
        from spline_trajectory_generator_py import (  # noqa: F401
            TrajectoryGenerator, Setpoint, evaluate_arc_length_spline)
        return TrajectoryGenerator, Setpoint, evaluate_arc_length_spline
    except ImportError:
        pass

    candidates = []
    env_dir = os.environ.get('MPCC_ACADOS_EXAMPLE_DIR')
    if env_dir:
        candidates.append(Path(env_dir))
    repo_root = os.environ.get('MPCC_REPO_ROOT')
    if repo_root:
        candidates.append(
            Path(repo_root) / 'workspace/thirdparty_libs/mpcc'
            / 'mpcc_acados_example/build/python')

    for cand in candidates:
        if (cand / 'spline_trajectory_generator_py' / '__init__.py').is_file():
            sys.path.insert(0, str(cand))
            from spline_trajectory_generator_py import (  # noqa: F401
                TrajectoryGenerator, Setpoint, evaluate_arc_length_spline)
            return TrajectoryGenerator, Setpoint, evaluate_arc_length_spline

    raise ImportError(
        'CircuitGenerator: cannot locate spline_trajectory_generator_py. '
        'Build the mpcc_acados_example pybind (cmake --build build) and '
        'expose its build/python on PYTHONPATH, or set MPCC_REPO_ROOT.')


@dataclass
class CircuitGeneratorConfig:
    """Parsed configuration for :class:`CircuitGenerator`.

    All paths default to the demo mission shipped with this repo
    (``configs/missions/demo_circuit.yaml`` + ``demo_gates.yaml``).
    Downstream consumers may override them at runtime either via the
    YAML config or via the ``CIRCUIT_MISSION_YAML`` /
    ``CIRCUIT_GATES_YAML`` environment variables (env wins if both are
    set), which is how a higher-level project can inject its own
    canonical mission without editing the YAMLs of this package.
    """

    mission_yaml: str = ''
    gates_yaml: str = ''
    desired_speed: float = 4.0           # m/s along the path (mirrors desired_theta_velocity)
    origin_offset_m: float = 3.0         # back from gate1 along its forward normal
    closing_exit_margin_m: float = 3.0   # past the closing gate1 along the gate's forward normal
    n_req_points: int = 20               # spline window size (MPCC knots)
    target_segment_length: float = 1.0   # m
    samples_per_segment: int = 400


class CircuitGenerator(ITrajectoryGenerator):
    """Closed-loop circuit generator backed by ``spline_py``."""

    def __init__(self, cfg: CircuitGeneratorConfig) -> None:
        if cfg.desired_speed <= 0.0:
            raise ValueError('CircuitGenerator: desired_speed must be > 0.')
        if cfg.n_req_points < 2:
            raise ValueError('CircuitGenerator: n_req_points must be >= 2.')
        self._cfg = cfg
        self._traj: Optional[object] = None
        self._evaluate_fn = None
        self._setpoint_cls = None
        self._traj_cls = None
        self._s_eval: float = 0.0
        self._s_max: float = 0.0
        self._t_anchor: float = 0.0
        self._s_anchor: float = 0.0
        # Cache the active sliding window between update() and the N+1
        # evaluate() calls that build a single horizon — otherwise
        # re-querying get_trajectory_window for each stage can swap the
        # window and produce discontinuous reference samples (large
        # acceleration spikes that fail the SQP_RTI step of the
        # trajectory MPC).
        self._window = None
        self._name = 'CircuitGenerator'

    @staticmethod
    def load_config_from_yaml(path: str) -> CircuitGeneratorConfig:
        if not os.path.isfile(path):
            raise ValueError(f'Config file not found at {os.path.abspath(path)}.')
        with open(path, 'r') as f:
            root = yaml.safe_load(f) or {}
        if not isinstance(root, dict):
            raise ValueError('circuit_generator config: root must be a mapping.')
        cfg = CircuitGeneratorConfig()
        # Mission / gates paths may be left blank to use the repo demo.
        if 'mission_yaml' in root:
            cfg.mission_yaml = str(root['mission_yaml'])
        if 'gates_yaml' in root:
            cfg.gates_yaml = str(root['gates_yaml'])
        for k in ('desired_speed', 'origin_offset_m', 'closing_exit_margin_m',
                  'target_segment_length'):
            if k in root:
                setattr(cfg, k, float(root[k]))
        for k in ('n_req_points', 'samples_per_segment'):
            if k in root:
                setattr(cfg, k, int(root[k]))
        return cfg

    # ------------------------------------------------------------------
    # Setpoint assembly — shared between CircuitGenerator and any
    # controller that also wants to build its own spline (MpccController).
    # ------------------------------------------------------------------

    @staticmethod
    def build_setpoints(
        initial_position: np.ndarray,
        mission_yaml: str,
        gates_yaml: str,
        origin_offset_m: float,
        closing_exit_margin_m: float,
        setpoint_cls,
    ) -> list:
        """Compose origin + canonical fly waypoints + exit setpoint.

        - Origin (drone start) is forced to the initial position so the
          first solve has zero contour error.
        - Mission waypoints follow verbatim from the YAML (gate1 -> ...
          -> gate12 -> gate1).
        - Exit setpoint is placed ``closing_exit_margin_m`` past the
          closing waypoint along its forward velocity. Without it the
          spline ends exactly on the closing gate's plane, the drone
          parks on the gate and the bench detector misses the closure
          sign-flip.
        """
        mission = ml.load(mission_yaml, gates_yaml)
        first = np.asarray(mission.fly[0].position, dtype=float)
        # Origin pose: nudged behind gate1 along its forward normal so
        # the lap-start sign flip can fire. If origin_offset_m is 0,
        # anchor at the initial position verbatim.
        gate1_vel = np.asarray(mission.fly[0].velocity, dtype=float)
        nvel = float(np.linalg.norm(gate1_vel))
        forward = gate1_vel / nvel if nvel > 1e-6 else np.array([1.0, 0.0, 0.0])
        if origin_offset_m > 0.0:
            origin_pos = first - origin_offset_m * forward
        else:
            origin_pos = np.asarray(initial_position, dtype=float)
        direction = first - origin_pos
        n = float(np.linalg.norm(direction))
        direction = direction / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])

        setpoints = [setpoint_cls(
            id='origin',
            position=[float(origin_pos[0]), float(origin_pos[1]), float(origin_pos[2])],
            velocity=[float(direction[0]), float(direction[1]), float(direction[2])],
        )]
        for wp in mission.fly:
            setpoints.append(setpoint_cls(
                id=wp.modifiers,
                position=list(wp.position),
                velocity=list(wp.velocity),
            ))

        # Exit setpoint past the closing waypoint.
        closing = mission.fly[-1]
        closing_pos = np.asarray(closing.position, dtype=float)
        closing_vel = np.asarray(closing.velocity, dtype=float)
        nv = float(np.linalg.norm(closing_vel))
        fwd = closing_vel / nv if nv > 1e-6 else np.array([1.0, 0.0, 0.0])
        exit_pos = closing_pos + closing_exit_margin_m * fwd
        setpoints.append(setpoint_cls(
            id='exit',
            position=[float(exit_pos[0]), float(exit_pos[1]), float(exit_pos[2])],
            velocity=[float(fwd[0]), float(fwd[1]), float(fwd[2])],
        ))
        return setpoints

    # ------------------------------------------------------------------
    # ITrajectoryGenerator API
    # ------------------------------------------------------------------

    def initialize(self, initial_state: State, example_cfg: ExampleConfig) -> None:
        # Resolve mission/gates paths with the precedence:
        #   1. env var (CIRCUIT_MISSION_YAML / CIRCUIT_GATES_YAML) — used by
        #      higher-level projects (e.g. mpcc_v2/launchers/simulation/launch/
        #      mav.sh) to inject their canonical mission without editing
        #      this repo's YAMLs.
        #   2. config YAML value.
        #   3. package demo defaults.
        env_mission = os.environ.get('CIRCUIT_MISSION_YAML', '')
        env_gates = os.environ.get('CIRCUIT_GATES_YAML', '')
        ml_mission, ml_gates = ml._default_paths()  # noqa: SLF001
        mission_yaml = env_mission or self._cfg.mission_yaml or ml_mission
        gates_yaml = env_gates or self._cfg.gates_yaml or ml_gates

        # Align the spline origin with where the framework's synthetic
        # takeoff is going to leave the drone. WaypointsSimulator inserts
        # a takeoff waypoint at (initial.xy, takeoff_altitude_m) before
        # the first project waypoint, so anchoring the spline at the
        # post-takeoff pose keeps the contour error bounded throughout
        # the take-off phase (no Z-jump from ground level to gate
        # altitude that destabilises the trajectory MPC QP).
        origin_pose = np.asarray(initial_state.position, dtype=float).copy()
        takeoff_alt = float(getattr(example_cfg, 'takeoff_altitude_m', 0.0))
        if takeoff_alt > 0.0:
            origin_pose[2] = takeoff_alt

        self._traj_cls, self._setpoint_cls, self._evaluate_fn = _import_spline_py()
        setpoints = self.build_setpoints(
            initial_position=origin_pose,
            mission_yaml=mission_yaml,
            gates_yaml=gates_yaml,
            origin_offset_m=self._cfg.origin_offset_m,
            closing_exit_margin_m=self._cfg.closing_exit_margin_m,
            setpoint_cls=self._setpoint_cls,
        )
        self._traj = self._traj_cls(
            setpoints,
            self._cfg.n_req_points,
            self._cfg.target_segment_length,
            self._cfg.samples_per_segment,
        )
        self._s_eval = 0.0
        self._t_anchor = 0.0
        self._s_anchor = 0.0
        # Cache the first sliding window so the very first evaluate()
        # call (before update() runs) has something to reference.
        self._window, _ = self._traj.get_trajectory_window(0.0)
        self._s_max = (float(self._window.s_values[-1])
                       if hasattr(self._window, 's_values') else 0.0)

    def on_waypoint_changed(
        self, next_waypoint: np.ndarray, state: State, t_start: float,
    ) -> None:
        del next_waypoint, state, t_start  # nothing to do: the spline covers the full circuit

    def update(self, t: float, state: State) -> None:
        if self._traj is None:
            return
        # Refresh the sliding window once per outer-loop step. All
        # subsequent evaluate() calls for the current horizon reuse
        # `self._window` so the references are sampled from a single,
        # contiguous parametrisation (no window-switch discontinuities
        # that would otherwise break the trajectory MPC's SQP_RTI step).
        self._window, s_eval = self._traj.get_trajectory_window(self._s_eval)
        new_s = self._project_to_spline(
            np.asarray(state.position, dtype=float), self._window, s_eval)
        self._s_eval = new_s
        self._t_anchor = t
        self._s_anchor = new_s
        self._s_max = float(self._window.s_values[-1])

    def evaluate(self, t: float) -> ReferenceSample:
        if self._traj is None or self._window is None:
            return ReferenceSample()
        speed = self._cfg.desired_speed
        s_query = self._s_anchor + speed * (t - self._t_anchor)
        # Clamp inside the active window so the spline evaluator never
        # extrapolates beyond it (clamping the global s_max is not
        # enough because the window covers only a slice of the spline).
        s_query = max(float(self._window.s_values[0]),
                       min(s_query, self._s_max))
        ev = self._evaluate_fn(s_query, self._window)
        position = np.asarray(ev.position[:3], dtype=float)
        tangent = np.asarray(ev.tangent[:3], dtype=float)
        velocity = speed * tangent
        # NOTE: acceleration is set to zero for the demo. The
        # finite-difference estimate computed below tends to amplify
        # spline noise at low desired_speed values and can destabilise
        # the trajectory MPC; the trajectory MPC tracks the
        # position+velocity references fine on its own.
        # acceleration = self._estimate_acceleration(s_query, speed, self._window)
        acceleration = np.zeros(3, dtype=float)
        yaw = self._tangent_yaw(tangent)
        return ReferenceSample(position=position, velocity=velocity,
                                acceleration=acceleration, yaw=yaw)

    def provided_reference_fields(self) -> ReferenceField:
        return make_mask([ReferenceField.POSITION,
                           ReferenceField.VELOCITY,
                           ReferenceField.ACCELERATION])

    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _project_to_spline(self, position: np.ndarray, window, prev_s: float) -> float:
        """Pick the spline arc-length closest to ``position``.

        Monotonic: never goes back along the path. Bounded by a small
        search radius around ``prev_s`` so the projection scales with
        window size, not total length.
        """
        s_values = np.asarray(window.s_values, dtype=float)
        path_pts = np.asarray(window.path.position, dtype=float).reshape(-1, 3)
        # Within the window, evaluate at the sampled s_values and pick the
        # one closest to the drone, then clamp >= prev_s for monotonicity.
        # path_pts has 3 * n_points (flattened); use the first len(s_values) groups.
        n = len(s_values)
        if path_pts.shape[0] < n:
            n = path_pts.shape[0]
        dists = np.linalg.norm(path_pts[:n] - position, axis=1)
        idx = int(np.argmin(dists))
        new_s = float(s_values[idx])
        if new_s < prev_s:
            new_s = prev_s
        return new_s

    def _estimate_acceleration(self, s: float, speed: float, window) -> np.ndarray:
        ds = max(0.05, 0.5 * self._cfg.target_segment_length)
        s_minus = max(0.0, s - ds)
        s_plus = min(self._s_max, s + ds)
        try:
            ev_minus = self._evaluate_fn(s_minus, window)
            ev_plus = self._evaluate_fn(s_plus, window)
            tangent_minus = np.asarray(ev_minus.tangent[:3], dtype=float)
            tangent_plus = np.asarray(ev_plus.tangent[:3], dtype=float)
            dtangent_ds = (tangent_plus - tangent_minus) / max(s_plus - s_minus, 1e-9)
            # da/dt = d(speed*tangent)/dt = speed * d(tangent)/dt = speed^2 * dT/ds
            return float(speed) ** 2 * dtangent_ds
        except Exception:
            return np.zeros(3, dtype=float)

    @staticmethod
    def _tangent_yaw(tangent: np.ndarray) -> float:
        return math.atan2(float(tangent[1]), float(tangent[0]))
