#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Acados MPCC (Model Predictive Contouring Control) adapter.

Self-contained: the controller loads the mission YAML inside
:meth:`initialize` and builds its own arc-length-reparametrised spline.
``compute_command()`` ignores the ``references`` argument and instead
sets the solver's online parameters from a sliding window over the
internal spline. This mirrors the way
``mpc_acados_position.MpcPositionController`` cooks its own per-stage
references from a single goal pose — the controller "knows" the path,
the framework just provides the per-tick tick.

Required reference fields are reported as ``POSITION`` (the minimum
that lets ``WaypointsSimulator`` validate the controller/generator
match); whatever the paired generator emits is discarded by this
adapter.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml
from mavpy.model import State

from examples_py.framework import (
    ControlCommand,
    ExampleConfig,
    IController,
    ReferenceField,
    ReferenceSample,
    make_mask,
)
from examples_py.generators.circuit_generator import CircuitGenerator
from examples_py.utils import mission_loader as ml


def _import_mpcc_acados():
    """Locate the mpcc_acados Python bindings."""
    try:
        from mpcc_acados import MPC
        from mpcc_acados.utils.mpc_yaml import configure_mpc_from_yaml
        return MPC, configure_mpc_from_yaml
    except ImportError:
        pass
    candidates = []
    env_dir = os.environ.get('MPCC_ACADOS_EXAMPLE_DIR')
    if env_dir:
        candidates.append(Path(env_dir).parent)  # mpcc_acados is one level above
    repo_root = os.environ.get('MPCC_REPO_ROOT')
    if repo_root:
        candidates.append(Path(repo_root) / 'workspace/thirdparty_libs/mpcc')
    candidates.append(Path('/home/rafa/mpcc_v2/workspace/thirdparty_libs/mpcc'))
    for cand in candidates:
        if (cand / 'mpcc_acados' / '__init__.py').is_file():
            sys.path.insert(0, str(cand))
            from mpcc_acados import MPC
            from mpcc_acados.utils.mpc_yaml import configure_mpc_from_yaml
            return MPC, configure_mpc_from_yaml
    raise ImportError(
        'MpccController: cannot locate mpcc_acados. Set MPCC_REPO_ROOT or '
        'MPCC_ACADOS_EXAMPLE_DIR so the loader can find the mpcc_acados '
        'Python package.')


_MPCC_SPLINE_KNOTS = 20
_SOFT_WEIGHT_DEFAULT = 2.0
_SOFT_WEIGHT_ORIGINAL = 1.0
_SEARCH_RADIUS = 1.0      # m around the previous s_eval
_DS_COARSE = 0.10
_DS_FINE = 0.01


@dataclass
class MpccControllerConfig:
    """Parsed configuration for :class:`MpccController`.

    Three groups of fields:
      * ``mpc_yaml_path``: file with the MPCC weights (``gains_*``,
        ``softcontraints_*``, ``lbu``, ``ubu``) consumed by
        :func:`mpcc_acados.utils.mpc_yaml.configure_mpc_from_yaml`.
      * ``ocp_json_file_path``: the acados-generated OCP json (defaults
        to the one shipped next to ``mpc_yaml_path``).
      * Mission paths + cruise tuning, shared with
        :class:`CircuitGenerator` (the controller cooks its own spline,
        independent of any generator).
    """

    mpc_yaml_path: str = ''
    ocp_json_file_path: str = ''
    mission_yaml: str = ''
    gates_yaml: str = ''
    desired_speed: float = 4.0
    origin_offset_m: float = 3.0
    closing_exit_margin_m: float = 3.0
    target_segment_length: float = 1.0
    samples_per_segment: int = 400


def _find_closest_s(drone_pos: np.ndarray, window, evaluate_fn,
                    s_min: float, s_max: float) -> float:
    """Coarse + fine search for the closest spline point to drone_pos."""
    best_s = s_min
    min_d = float('inf')
    s = s_min
    while s <= s_max:
        pt = evaluate_fn(s, window)
        d = math.dist(drone_pos, pt.position[:3])
        if d < min_d:
            min_d = d
            best_s = s
        s += _DS_COARSE
    s_fine = max(s_min, best_s - _DS_COARSE)
    s_end = min(s_max, best_s + _DS_COARSE)
    while s_fine <= s_end:
        pt = evaluate_fn(s_fine, window)
        d = math.dist(drone_pos, pt.position[:3])
        if d < min_d:
            min_d = d
            best_s = s_fine
        s_fine += _DS_FINE
    return best_s


class MpccController(IController):
    """Self-contained MPCC adapter (ignores the framework's references)."""

    def __init__(self, cfg: MpccControllerConfig) -> None:
        if not cfg.mpc_yaml_path:
            raise ValueError('MpccController: mpc_yaml_path must be provided.')
        if cfg.desired_speed <= 0.0:
            raise ValueError('MpccController: desired_speed must be > 0.')
        self._cfg = cfg
        self._mpc = None
        self._mpc_data = None
        self._traj = None
        self._evaluate_fn = None
        self._setpoint_cls = None
        self._traj_cls = None
        self._s_eval = 0.0
        self._control_period = 0.04
        self._last_solve_us = 0.0
        self._last_desired_velocity = np.zeros(3, dtype=float)
        self._name = 'MpccController'

    # ------------------------------------------------------------------
    # YAML config
    # ------------------------------------------------------------------

    @staticmethod
    def load_config_from_yaml(path: str) -> MpccControllerConfig:
        """Read ``configs/controllers/config_mpcc.yaml``.

        Top-level keys:
          * ``controller.ocp_json_file_path`` (optional) — acados OCP path.
          * ``mpcc.parameters`` / ``mpcc.constraints`` — solver weights and
            bounds. Uses the ``mpcc:`` key (NOT ``mpc:``) so the file is
            visually distinct from `config_mpc_trajectory.yaml`. The
            adapter rewrites the block to ``mpc:`` at runtime when
            handing it to the upstream ``configure_mpc_from_yaml``
            loader (which is hard-coded against ``mpc:``).
          * ``circuit`` (optional) — mission anchors mirroring
            :class:`CircuitGenerator`'s config.
        """
        if not os.path.isfile(path):
            raise ValueError(f'Config file not found at {os.path.abspath(path)}.')
        with open(path, 'r') as f:
            root = yaml.safe_load(f) or {}
        if not isinstance(root, dict):
            raise ValueError('mpcc config: root must be a mapping.')
        if 'mpcc' not in root:
            raise ValueError(
                f"mpcc config {path}: missing top-level 'mpcc:' block. "
                "(If you are porting an old config, rename `mpc:` -> `mpcc:`.)")
        cfg = MpccControllerConfig(mpc_yaml_path=path)
        ctrl = root.get('controller')
        if isinstance(ctrl, dict):
            if 'ocp_json_file_path' in ctrl:
                cfg.ocp_json_file_path = str(ctrl['ocp_json_file_path'])
        circuit = root.get('circuit')
        if isinstance(circuit, dict):
            for k in ('mission_yaml', 'gates_yaml'):
                if k in circuit:
                    setattr(cfg, k, str(circuit[k]))
            for k in ('desired_speed', 'origin_offset_m', 'closing_exit_margin_m',
                      'target_segment_length'):
                if k in circuit:
                    setattr(cfg, k, float(circuit[k]))
            if 'samples_per_segment' in circuit:
                cfg.samples_per_segment = int(circuit['samples_per_segment'])
        return cfg

    # ------------------------------------------------------------------
    # IController
    # ------------------------------------------------------------------

    def initialize(self, initial_state: State, example_cfg: ExampleConfig) -> None:
        if example_cfg.mpc_dt <= 0.0:
            raise ValueError('MpccController: example_cfg.mpc_dt must be > 0.')

        MPC, configure_mpc_from_yaml = _import_mpcc_acados()
        # MPCC solver --------------------------------------------------
        from examples_py.generators.circuit_generator import _import_spline_py
        self._traj_cls, self._setpoint_cls, self._evaluate_fn = _import_spline_py()

        ocp_json = self._cfg.ocp_json_file_path
        if not ocp_json:
            # Fall back, in order of preference:
            #   1. ``MPCC_ACADOS_EXAMPLE_DIR/mpcc_interface/...`` (env override
            #      set by the bench harness when the package is co-located
            #      with the mpcc repo).
            #   2. ``${MPCC_REPO_ROOT}/workspace/thirdparty_libs/mpcc/...``
            #      (canonical mpcc_v2 layout).
            #   3. Sibling of the mpc_yaml_path (the upstream mpcc layout).
            for cand_dir in [
                os.environ.get('MPCC_ACADOS_EXAMPLE_DIR'),
                (os.path.join(os.environ['MPCC_REPO_ROOT'],
                              'workspace/thirdparty_libs/mpcc/'
                              'mpcc_acados_example')
                 if os.environ.get('MPCC_REPO_ROOT') else None),
                str(Path(self._cfg.mpc_yaml_path).resolve().parent.parent),
            ]:
                if not cand_dir:
                    continue
                cand = os.path.join(cand_dir, 'mpcc_interface',
                                    'mpc_generated_code', 'acados_ocp.json')
                if os.path.isfile(cand):
                    ocp_json = cand
                    break
        if not ocp_json or not os.path.isfile(ocp_json):
            raise ValueError(
                f'MpccController: acados ocp_json_file_path not found '
                f'(tried explicit + MPCC_ACADOS_EXAMPLE_DIR + '
                f'MPCC_REPO_ROOT + sibling of {self._cfg.mpc_yaml_path}). '
                f'Set MPCC_ACADOS_EXAMPLE_DIR or specify the path in '
                f'configs/controllers/config_mpcc.yaml under '
                f'controller.ocp_json_file_path.')
        self._mpc = MPC(ocp_json_file=ocp_json)
        # `configure_mpc_from_yaml` upstream is hardcoded to read the
        # `mpc:` top-level key, but our config files use `mpcc:` so
        # they don't collide with `config_mpc_trajectory.yaml`. Rewrite
        # the block to a tmp YAML with the legacy key before invoking
        # the loader, then drop the temp file.
        configure_mpc_from_yaml(self._mpc, self._rekey_mpcc_to_mpc(
            self._cfg.mpc_yaml_path))
        self._mpc_data = self._mpc.get_data()
        self._control_period = float(self._mpc.get_prediction_time_step())
        if example_cfg.mpc_dt > 0:
            # Honour the framework's mpc_dt as the outer-loop step. The
            # MPCC solver still uses its own prediction step internally.
            self._control_period = float(example_cfg.mpc_dt)

        # Mission + spline --------------------------------------------
        # Path precedence: env var > config YAML > package demo defaults.
        # The env vars let downstream consumers (e.g. mpcc_v2's
        # launchers/simulation/launch/mav.sh) inject their own mission
        # without editing the generic configs shipped here.
        env_mission = os.environ.get('CIRCUIT_MISSION_YAML', '')
        env_gates = os.environ.get('CIRCUIT_GATES_YAML', '')
        ml_mission, ml_gates = ml._default_paths()  # noqa: SLF001
        mission_yaml = env_mission or self._cfg.mission_yaml or ml_mission
        gates_yaml = env_gates or self._cfg.gates_yaml or ml_gates

        # Match the CircuitGenerator's anchor logic: post-takeoff pose
        # (initial.xy, takeoff_altitude_m) so the MPCC's first solve
        # sees zero contour error throughout the takeoff phase. See
        # examples_py/generators/circuit_generator.py for rationale.
        origin_pose = np.asarray(initial_state.position, dtype=float).copy()
        takeoff_alt = float(getattr(example_cfg, 'takeoff_altitude_m', 0.0))
        if takeoff_alt > 0.0:
            origin_pose[2] = takeoff_alt

        setpoints = CircuitGenerator.build_setpoints(
            initial_position=origin_pose,
            mission_yaml=mission_yaml,
            gates_yaml=gates_yaml,
            origin_offset_m=self._cfg.origin_offset_m,
            closing_exit_margin_m=self._cfg.closing_exit_margin_m,
            setpoint_cls=self._setpoint_cls,
        )
        self._traj = self._traj_cls(
            setpoints,
            _MPCC_SPLINE_KNOTS,
            self._cfg.target_segment_length,
            self._cfg.samples_per_segment,
        )
        self._s_eval = 0.0
        self._last_solve_us = 0.0
        self._last_desired_velocity = np.zeros(3, dtype=float)

    def reference_horizon_size(self) -> int:
        # MPCC builds the horizon internally from the spline. The framework
        # still queries 1 sample per tick so the generator hook fires.
        return 1

    def reference_horizon_dt(self) -> float:
        return self._control_period

    def control_period(self) -> float:
        return self._control_period

    def required_reference_fields(self) -> ReferenceField:
        # We do not consume the reference samples — POSITION is the
        # cheapest mask that satisfies WaypointsSimulator's compatibility
        # check against any of the position / trajectory generators.
        return make_mask([ReferenceField.POSITION])

    def name(self) -> str:
        return self._name

    def last_solve_time_micros(self) -> float:
        return self._last_solve_us

    def last_desired_velocity(self) -> np.ndarray:
        return self._last_desired_velocity

    def provides_desired_velocity(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # YAML re-key shim
    # ------------------------------------------------------------------

    @staticmethod
    def _rekey_mpcc_to_mpc(source_path: str) -> str:
        """Write a temporary copy of ``source_path`` with the top-level
        ``mpcc:`` block renamed to ``mpc:`` so the upstream
        :func:`configure_mpc_from_yaml` (which is hardcoded against
        ``mpc:``) can consume it. The temp file lives in ``$TMPDIR``
        and is left for the OS to reap — `configure_mpc_from_yaml` only
        reads it once at construction time."""
        with open(source_path, 'r', encoding='utf-8') as f:
            root = yaml.safe_load(f) or {}
        if not isinstance(root, dict) or 'mpcc' not in root:
            return source_path
        rekeyed = dict(root)
        rekeyed['mpc'] = rekeyed.pop('mpcc')
        fd, tmp_path = tempfile.mkstemp(prefix='mpcc_rekey_', suffix='.yaml')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write('# AUTO-GENERATED by MpccController._rekey_mpcc_to_mpc — '
                    'do not edit by hand.\n')
            yaml.safe_dump(rekeyed, f, sort_keys=False, default_flow_style=None)
        return tmp_path

    # ------------------------------------------------------------------
    # MPCC step
    # ------------------------------------------------------------------

    def compute_command(self, state: State,
                         references: List[ReferenceSample]) -> ControlCommand:
        del references  # MPCC sources its references from the internal spline
        if self._mpc is None or self._traj is None:
            raise RuntimeError('MpccController: initialize() must be called first.')

        pos = np.asarray(state.position, dtype=float)
        q = np.asarray(state.orientation, dtype=float)
        vel = np.asarray(state.linear_velocity, dtype=float)

        # MPCC state vector: [x, y, z, qw, qx, qy, qz, vx, vy, vz, theta]
        mpc_state = np.array([pos[0], pos[1], pos[2],
                              q[0], q[1], q[2], q[3],
                              vel[0], vel[1], vel[2],
                              self._s_eval], dtype=float)
        self._mpc_data.state.vector = mpc_state

        window, s_eval = self._traj.get_trajectory_window(self._s_eval)
        self._s_eval = s_eval

        params = self._mpc_data.parameters
        params.set_spline_points(
            np.asarray(window.path.position[:3 * _MPCC_SPLINE_KNOTS], dtype=float))
        params.set_spline_tangents(
            np.asarray(window.path.tangent[:3 * _MPCC_SPLINE_KNOTS], dtype=float))
        params.set_spline_face_points(
            np.asarray(window.facing_points[:3 * _MPCC_SPLINE_KNOTS], dtype=float))
        params.set_s_lengths(
            np.asarray(window.s_values[:_MPCC_SPLINE_KNOTS], dtype=float))
        soft = np.full(_MPCC_SPLINE_KNOTS, _SOFT_WEIGHT_DEFAULT, dtype=float)
        for og in window.original_points:
            if 0 <= og.index < _MPCC_SPLINE_KNOTS:
                soft[og.index] = _SOFT_WEIGHT_ORIGINAL
        params.set_spline_softconstraints(soft)

        t0 = time.perf_counter()
        try:
            self._mpc.solve()
        except Exception as exc:
            raise RuntimeError(f'MpccController: solver failed: {exc}') from exc
        self._last_solve_us = (time.perf_counter() - t0) * 1e6

        actu = np.asarray(self._mpc_data.actuation.vector, dtype=float)
        thrust = float(actu[0])
        rates = actu[1:4].astype(float).copy()

        # Update s_eval by projecting the drone onto the spline (monotonic).
        s_top = float(window.s_values[-1])
        s_min = max(0.0, self._s_eval - _SEARCH_RADIUS)
        s_max = min(s_top, self._s_eval + _SEARCH_RADIUS)
        best_s = _find_closest_s(pos, window, self._evaluate_fn, s_min, s_max)
        best_s = max(self._s_eval, best_s)
        self._s_eval = min(max(best_s, 0.0), s_top)

        # Expose stage-1 predicted velocity (best proxy for v_des).
        self._last_desired_velocity = self._cfg.desired_speed * np.asarray(
            window.path.tangent[:3], dtype=float)

        return ControlCommand(thrust_n=thrust, angular_rate=rates)
