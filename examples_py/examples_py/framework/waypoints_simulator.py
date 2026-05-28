#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Waypoint simulator for the unified controller × generator comparison.

Python mirror of ``examples/framework/src/waypoints_simulator.cpp``.

Architecture:
  - Waypoint advancement is driven by :class:`WaypointScheduler` on a time
    basis so every controller × generator combination receives identical
    transitions at identical simulator times.
  - Controller and generator compute times are modelled as latency through
    two :class:`DelayBuffer` queues. The physics loop keeps advancing while
    the outer stage has no fresh output; logging records the delay that was
    actually applied to each tick.
  - Logging uses the ROS 2-compatible MCAP backend (:class:`UnifiedMcapLogger`).
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

import math
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from mavpy.controllers import ControlMode
from mavpy.simulator import Simulator, SimulatorParameters

from .controller_base import IController
from .delay_buffer import DelayBuffer
from .example_config import DelayMode, ExampleConfig
from .stdout_progress import print_case_summary, print_status
from .trajectory_generator_base import ITrajectoryGenerator
from mav_flight_review import TrajectoryPoint

from .types import ControlCommand, ReferenceField, ReferenceSample, has_field
# TODO: remove once MCAP pipeline validated.
# from .unified_csv_logger import LogRow, RunMetadata, UnifiedCsvLogger
from .unified_mcap_logger import LogRow, RunMetadata, UnifiedMcapLogger
from .waypoint_scheduler import WaypointScheduler


def _euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    sr = math.sin(roll * 0.5)
    cr = math.cos(roll * 0.5)
    sp = math.sin(pitch * 0.5)
    cp = math.cos(pitch * 0.5)
    sy = math.sin(yaw * 0.5)
    cy = math.cos(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    q = np.array([w, x, y, z], dtype=float)
    n = float(np.linalg.norm(q))
    return q / n if n > 0.0 else q


def _describe_fields_mask(mask: ReferenceField) -> str:
    parts = []
    if has_field(mask, ReferenceField.POSITION):
        parts.append('position')
    if has_field(mask, ReferenceField.VELOCITY):
        parts.append('velocity')
    if has_field(mask, ReferenceField.ACCELERATION):
        parts.append('acceleration')
    return ', '.join(parts) if parts else '<none>'


def _is_finite_vec(v: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(v)))


def _mean_us(seconds: List[float]) -> float:
    if not seconds:
        return 0.0
    return (sum(seconds) / len(seconds)) * 1e6


def _resolve_delay(mode: DelayMode, measured_s: float, fixed_s: float) -> float:
    if mode == DelayMode.MEASURED:
        return max(measured_s, 0.0)
    return max(fixed_s, 0.0)


@dataclass
class BenchmarkStats:
    """Aggregated timing statistics produced by a single :meth:`WaypointsSimulator.run`."""

    simulated_time_s: float = 0.0
    real_time_s: float = 0.0
    sim_speedup: float = 0.0
    controller_mean_us: float = 0.0
    generator_update_mean_us: float = 0.0
    generator_eval_mean_us: float = 0.0
    indi_mean_us: float = 0.0
    imu_mean_us: float = 0.0
    model_mean_us: float = 0.0
    tracking_rmse_m: float = 0.0
    controller_steps: int = 0
    indi_steps: int = 0


@dataclass
class _TimedCommand:
    cmd: ControlCommand = field(default_factory=ControlCommand)
    compute_time_us: float = 0.0
    delay_applied_us: float = 0.0


@dataclass
class _TimedReference:
    sample: ReferenceSample = field(default_factory=ReferenceSample)
    update_time_us: float = 0.0
    eval_time_us: float = 0.0
    delay_applied_us: float = 0.0


class WaypointsSimulator:
    """Runs a single controller × generator mission."""

    def __init__(
        self,
        controller: IController,
        traj_gen: ITrajectoryGenerator,
        example_cfg: ExampleConfig,
        simulator_params: SimulatorParameters,
        output_csv: str,
        metadata: RunMetadata,
    ) -> None:
        if controller is None:
            raise ValueError('WaypointsSimulator: controller must not be None.')
        if traj_gen is None:
            raise ValueError('WaypointsSimulator: trajectory generator must not be None.')
        if not example_cfg.waypoints:
            raise ValueError('WaypointsSimulator: example_cfg.waypoints is empty.')

        self._controller = controller
        self._traj_gen = traj_gen
        self._example_cfg = example_cfg
        self._output_csv = output_csv
        self._metadata = metadata
        self._sim = Simulator(simulator_params)
        self._stats = BenchmarkStats()

        self._check_compatibility()

    def _check_compatibility(self) -> None:
        required = self._controller.required_reference_fields()
        provided = self._traj_gen.provided_reference_fields()
        missing = ReferenceField(required & ~provided)
        if missing != ReferenceField.NONE:
            sys.stderr.write(
                f"[WaypointsSimulator] Warning: generator '{self._traj_gen.name()}' "
                f"does not produce field(s) required by controller "
                f"'{self._controller.name()}': {_describe_fields_mask(missing)}. "
                'Missing fields will be held at zero; controller may not '
                'converge as expected.\n'
            )

    @property
    def stats(self) -> BenchmarkStats:
        return self._stats

    @property
    def output_csv(self) -> str:
        return self._output_csv

    def run(self) -> None:
        sim = self._sim
        # Optional override of the simulator's initial pose. Applied
        # BEFORE arm() so the HOVER reference (seeded from getState()
        # inside arm) matches the new start pose. Mirrors the
        # `vehicle_initial_pose` consumed by the standalone acados
        # examples.
        if self._example_cfg.has_initial_state:
            from mavpy.model import State as _State
            rpy = self._example_cfg.initial_rpy
            init_state = _State(
                position=np.asarray(
                    self._example_cfg.initial_position, dtype=float),
                orientation=_euler_to_quaternion(
                    float(rpy[0]), float(rpy[1]), float(rpy[2])),
            )
            sim.set_initial_state(init_state)
        sim.arm()
        sim.set_control_mode(ControlMode.RATES)

        initial_state = sim.state
        self._controller.initialize(initial_state, self._example_cfg)
        self._traj_gen.initialize(initial_state, self._example_cfg)

        # Effective waypoint list with optional takeoff / land phases ------
        # Mirrors aerostack2's 3-phase mission. Prepend a synthetic takeoff
        # waypoint at (initial.x, initial.y, takeoff_altitude_m) when
        # takeoff_altitude_m > 0, and append a land waypoint at
        # (last.x, last.y, 0) when land_at_end=True. Phase 6 uses
        # mission_first_idx / mission_last_idx to gate experiment_active.
        init_pos_xyz = np.asarray(initial_state.position, dtype=float)
        effective_waypoints = [np.asarray(w, dtype=float).copy()
                               for w in self._example_cfg.waypoints]
        # `mission_first_idx` and `mission_last_idx` are 0-based inclusive
        # bounds around the user-supplied waypoints inside `effective_waypoints`.
        # `experiment_active` is True only inside that closed interval, so the
        # synthetic takeoff (idx < mission_first_idx) and the synthetic land
        # (idx > mission_last_idx) stay out of the paper metrics — mirroring
        # aerostack2's behavior, where takeoff_behavior and land_behavior keep
        # `experiment_active=False` outside the mission window.
        n_project_wps = len(effective_waypoints)
        mission_first_idx = 0
        if (self._example_cfg.takeoff_altitude_m > 0.0
                and n_project_wps > 0):
            takeoff_wp = init_pos_xyz.copy()
            takeoff_wp[2] = float(self._example_cfg.takeoff_altitude_m)
            effective_waypoints.insert(0, takeoff_wp)
            mission_first_idx = 1
        # Inclusive last index of the user-supplied waypoint segment.
        mission_last_idx = mission_first_idx + max(n_project_wps - 1, 0)
        if (self._example_cfg.land_at_end
                and n_project_wps > 0):
            land_wp = effective_waypoints[-1].copy()
            land_wp[2] = 0.0
            effective_waypoints.append(land_wp)

        # Scheduler --------------------------------------------------------
        # In `continuous` mission_mode, override the scheduler tunables so
        # the drone flows through every waypoint with no idle hold between
        # hops (parity with aerostack2's mission_moving_path.py).
        continuous_mode = (self._example_cfg.mission_mode == 'continuous')
        scheduler_settle_margin_s = (
            0.0 if continuous_mode else self._example_cfg.settle_margin_s)
        scheduler_speed_factor = (
            1.0 if continuous_mode else self._example_cfg.scheduler_speed_factor)

        scheduler = WaypointScheduler()
        scheduler.initialize(
            effective_waypoints,
            init_pos_xyz,
            self._example_cfg.max_speed,
            scheduler_settle_margin_s,
            scheduler_speed_factor,
        )

        self._traj_gen.on_waypoint_changed(
            np.asarray(effective_waypoints[0], dtype=float),
            initial_state, 0.0,
        )

        # follow_reference emulation (continuous_mode only) ---------------
        # Mirror of the C++ logic in waypoints_simulator.cpp: build a
        # piecewise-linear moving-target schedule through every waypoint
        # at `max_speed`, then call the local generator's
        # on_waypoint_changed() every `target_modify_period_s` whenever the
        # sampled target has shifted more than `target_modify_threshold_m`
        # from the last published goal. Active for every controller in
        # continuous_mode, including `mpc_trajectory` (parity with the
        # aerostack2 3-phase moving_path flow); the QP solver of the
        # trajectory MPC tolerates the modify cadence as long as it stays
        # below ~10 Hz, which is what `target_modify_period_s ≈ 0.1` enforces.
        controller_name = getattr(self._metadata, 'controller_name', '')
        target_plan_active = (
            continuous_mode
            and len(effective_waypoints) >= 2
        )
        # The target plan starts at the live initial pose so the moving TF
        # is anchored where the drone actually is at fly-phase entry
        # (analogous to the TF lookup in aerostack2's `drone_fly()`). If
        # the user's wp[0] is already the spawn pose we skip the prepend
        # to avoid emitting a zero-length first segment.
        raw_target_wps = np.asarray(effective_waypoints, dtype=float)
        init_pos = np.asarray(initial_state.position, dtype=float)
        if np.linalg.norm(raw_target_wps[0] - init_pos) > 1e-3:
            target_wps = np.vstack([init_pos[None, :], raw_target_wps])
        else:
            target_wps = raw_target_wps
        target_t_at_wp = np.zeros(len(target_wps))
        if target_plan_active:
            speeds = max(self._example_cfg.max_speed, 1e-9)
            for i in range(1, len(target_wps)):
                target_t_at_wp[i] = target_t_at_wp[i - 1] + \
                    float(np.linalg.norm(target_wps[i] - target_wps[i - 1])) / speeds
            if not self._example_cfg.silent:
                print(
                    f'  follow_reference emulation: linear target plan '
                    f'duration = {target_t_at_wp[-1]:.2f} s, modify_period = '
                    f'{self._example_cfg.target_modify_period_s:.3f} s, '
                    f'threshold = {self._example_cfg.target_modify_threshold_m:.2f} m, '
                    f'wps={len(target_wps)} (init pose prepended: '
                    f'{target_wps is not raw_target_wps})'
                )
        target_plan_duration = float(target_t_at_wp[-1]) if target_plan_active else 0.0
        target_last_published = np.asarray(initial_state.position, dtype=float).copy()
        target_last_modify_t = -1.0e9
        # Degenerate-hold state. Mirrors `degenerate_hold_` /
        # `degenerate_target_` / `init_yaw_angle_` in aerostack2's
        # generate_polynomial_trajectory_behavior. Active only when
        # target_plan_active (i.e. inside the follow_reference emulator).
        from examples_py.framework.degenerate_hold import (
            fill_static_horizon,
            is_degenerate_target,
            quat_to_yaw,
        )
        degenerate_hold_active = False
        degenerate_target = np.asarray(initial_state.position, dtype=float).copy()
        degenerate_yaw = quat_to_yaw(np.asarray(initial_state.orientation, dtype=float))

        def target_plan_position(t: float) -> np.ndarray:
            """Sample the piecewise-linear moving target at time t."""
            if t <= 0.0:
                return target_wps[0].copy()
            if t >= target_t_at_wp[-1]:
                return target_wps[-1].copy()
            i = 1
            while i < len(target_t_at_wp) and t > target_t_at_wp[i]:
                i += 1
            t0 = target_t_at_wp[i - 1]
            t1 = target_t_at_wp[i]
            alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return (1.0 - alpha) * target_wps[i - 1] + alpha * target_wps[i]

        # Timing params ----------------------------------------------------
        model_dt = self._example_cfg.model_dt
        controller_dt = self._example_cfg.controller_dt
        outer_dt = self._controller.control_period()
        hover_time = self._example_cfg.hover_time
        mission_end_t = (
            (self._example_cfg.target_start_delay_s + target_plan_duration)
            if target_plan_active else scheduler.final_time()
        )
        # sim_config.sim_time is a hard cap on the wall of simulated time:
        # the run ends at min(mission_end + hover, sim_time), even if that
        # truncates the hover phase or the mission itself.
        max_sim_time = min(mission_end_t + hover_time, self._example_cfg.sim_time)
        benchmark = self._example_cfg.benchmark
        silent = self._example_cfg.silent
        max_speed = self._example_cfg.max_speed

        n_samples = self._controller.reference_horizon_size()
        dt_h = self._controller.reference_horizon_dt()
        refs: List[ReferenceSample] = [ReferenceSample() for _ in range(n_samples)]

        # Logger -----------------------------------------------------------
        # TODO: remove once MCAP pipeline validated.
        # logger: Optional[UnifiedCsvLogger] = None
        # if not benchmark and self._output_csv:
        #     logger = UnifiedCsvLogger(self._output_csv, self._metadata)
        logger: Optional[UnifiedMcapLogger] = None
        if not benchmark and self._output_csv:
            logger = UnifiedMcapLogger(self._output_csv, self._metadata)

        # Delay buffers ----------------------------------------------------
        cmd_buffer: DelayBuffer = DelayBuffer()
        ref_buffer: DelayBuffer = DelayBuffer()

        # Benchmark accumulators ------------------------------------------
        controller_times: List[float] = []
        generator_update_times: List[float] = []
        generator_eval_times: List[float] = []
        indi_times: List[float] = []
        imu_times: List[float] = []
        model_times: List[float] = []

        tracking_sq_sum = 0.0
        tracking_samples = 0

        # Warm-start -------------------------------------------------------
        self._traj_gen.update(0.0, initial_state)
        for k in range(n_samples):
            refs[k] = self._traj_gen.evaluate(k * dt_h)
        warm_cmd = self._controller.compute_command(initial_state, refs)
        warm = _TimedCommand(cmd=warm_cmd)
        warm_ref = _TimedReference(sample=refs[0])
        cmd_buffer.push(warm, 0.0)
        ref_buffer.push(warm_ref, 0.0)

        # Running state
        current_cmd = ControlCommand()
        current_ref = refs[0]
        current_cmd_compute_us = 0.0
        current_cmd_delay_us = 0.0
        current_ref_update_us = 0.0
        current_ref_eval_us = 0.0
        current_ref_delay_us = 0.0

        # Detect whether the active controller consumes a velocity /
        # acceleration horizon. Trajectory-scope controllers (mpc_trajectory)
        # do; position-only ones (pid, mpc_position) don't. Mirrors
        # aerostack2's behaviour: trajectory_generation_behavior only emits
        # `motion_reference/trajectory` for trajectory-scope runs.
        required_fields = self._controller.required_reference_fields()
        emit_trajectory_horizon = (
            has_field(required_fields, ReferenceField.VELOCITY) or
            has_field(required_fields, ReferenceField.ACCELERATION))

        hover_active = False
        hover_end_time = max_sim_time
        active_index = scheduler.active_index()

        # Per-topic emission state for the mission-side topics so the MCAP
        # mirrors aerostack2's publish pattern: pose_ref rate-limited at
        # `mission_pose_ref_freq`, the latched topics emitted only on
        # value change. `t_last_mission_pose_ref_pub` is -inf so the first
        # tick inside the mission window emits the topic; the `None`
        # sentinels on the latched caches guarantee the initial value is
        # emitted exactly once (so the reviewer's ZOH resampler has data
        # to hold from).
        mission_pose_ref_period_s = (
            1.0 / self._example_cfg.mission_pose_ref_freq
            if self._example_cfg.mission_pose_ref_freq > 0.0 else 0.0)
        t_last_mission_pose_ref_pub = -1.0e30
        last_waypoint_index: Optional[int] = None
        last_max_speed: Optional[float] = None
        last_experiment_active: Optional[bool] = None
        last_hover_active: Optional[bool] = None

        wall_start = time.perf_counter()
        t = 0.0

        gen_delay_mode = self._example_cfg.generator_delay_mode
        gen_delay_fixed = self._example_cfg.generator_delay_fixed_s
        ctrl_delay_mode = self._example_cfg.controller_delay_mode
        ctrl_delay_fixed = self._example_cfg.controller_delay_fixed_s

        while t < max_sim_time + 1e-9:
            if hover_active and t >= hover_end_time - 1e-9:
                break

            state = sim.state
            position = np.asarray(state.position, dtype=float)
            orientation = np.asarray(state.orientation, dtype=float)
            linear_velocity = np.asarray(state.linear_velocity, dtype=float)
            if (not _is_finite_vec(position) or not _is_finite_vec(orientation)
                    or not _is_finite_vec(linear_velocity)):
                sys.stderr.write(f'\n[WaypointsSimulator] Non-finite state at t={t} s\n')
                break

            # Scheduler tick
            tick = scheduler.tick(t)
            if tick.waypoint_changed:
                self._traj_gen.on_waypoint_changed(
                    scheduler.waypoint(tick.active_index), state, t)
            active_index = tick.active_index

            # follow_reference emulation tick (continuous_mode only) -------
            # Mirrors the C++ logic. The degenerate-hold gate runs FIRST:
            # if the current target is within `degenerate_distance_m` of
            # the drone, we skip the local generator and publish a static
            # horizon (target, v=0, a=0, latched yaw). Otherwise the modify
            # gate (period + threshold) decides whether to replan.
            skip_generator_for_hold = False
            if target_plan_active:
                t_motion = max(0.0, t - self._example_cfg.target_start_delay_s)
                t_target = min(t_motion, target_plan_duration)
                target_now = target_plan_position(t_target)
                target_now_degenerate = is_degenerate_target(
                    target_now, position,
                    self._example_cfg.degenerate_distance_m)
                if target_now_degenerate:
                    if not degenerate_hold_active and not self._example_cfg.silent:
                        sys.stderr.write(
                            f'\n[WaypointsSimulator] Target within '
                            f'{self._example_cfg.degenerate_distance_m} m of '
                            f'vehicle at t={t} s: degenerate-hold engaged.\n')
                    if not degenerate_hold_active:
                        degenerate_yaw = quat_to_yaw(orientation)
                    degenerate_hold_active = True
                    degenerate_target = target_now.copy()
                    skip_generator_for_hold = True
                    target_last_published = target_now.copy()
                    target_last_modify_t = t
                else:
                    if degenerate_hold_active:
                        if not self._example_cfg.silent:
                            sys.stderr.write(
                                f'\n[WaypointsSimulator] Degenerate-hold '
                                f'released at t={t} s: regenerating '
                                f'trajectory.\n')
                        # Force a replan on the first tick out of the hold.
                        self._traj_gen.on_waypoint_changed(target_now, state, t)
                        target_last_published = target_now.copy()
                        target_last_modify_t = t
                        degenerate_hold_active = False
                    else:
                        dt_since_modify = t - target_last_modify_t
                        dist = float(np.linalg.norm(target_now - target_last_published))
                        if (dt_since_modify >= self._example_cfg.target_modify_period_s
                                and dist >= self._example_cfg.target_modify_threshold_m):
                            self._traj_gen.on_waypoint_changed(target_now, state, t)
                            target_last_published = target_now.copy()
                            target_last_modify_t = t

            # Generator step ------------------------------------------------
            gen_t0 = time.perf_counter()
            if skip_generator_for_hold:
                # Static horizon latched to (degenerate_target, degenerate_yaw,
                # v=0, a=0). Skip traj_gen.update entirely.
                fill_static_horizon(degenerate_target, degenerate_yaw,
                                    refs, n_samples)
                gen_t1 = time.perf_counter()
                gen_t2 = gen_t1
            else:
                self._traj_gen.update(t, state)
                gen_t1 = time.perf_counter()
                for k in range(n_samples):
                    refs[k] = self._traj_gen.evaluate(t + k * dt_h)
                gen_t2 = time.perf_counter()
            gen_update_s = gen_t1 - gen_t0
            gen_eval_s = gen_t2 - gen_t1
            generator_update_times.append(gen_update_s)
            generator_eval_times.append(gen_eval_s)
            gen_delay_s = _resolve_delay(
                gen_delay_mode, gen_update_s + gen_eval_s, gen_delay_fixed)

            ref_payload = _TimedReference(
                sample=refs[0],
                update_time_us=gen_update_s * 1e6,
                eval_time_us=gen_eval_s * 1e6,
                delay_applied_us=gen_delay_s * 1e6,
            )

            # Controller step ----------------------------------------------
            ctrl_t0 = time.perf_counter()
            cmd = self._controller.compute_command(state, refs)
            ctrl_t1 = time.perf_counter()
            ctrl_solve_s = ctrl_t1 - ctrl_t0
            controller_times.append(ctrl_solve_s)

            ctrl_delay_s = _resolve_delay(
                ctrl_delay_mode, ctrl_solve_s, ctrl_delay_fixed)

            ref_buffer.push(ref_payload, t + gen_delay_s)

            cmd_payload = _TimedCommand(
                cmd=cmd,
                compute_time_us=ctrl_solve_s * 1e6,
                delay_applied_us=ctrl_delay_s * 1e6,
            )
            cmd_buffer.push(cmd_payload, t + gen_delay_s + ctrl_delay_s)

            # Hover transition ---------------------------------------------
            mission_done = (
                (t >= mission_end_t - 1e-9)
                if target_plan_active
                else (tick.finished and t >= mission_end_t - 1e-9)
            )
            if not hover_active and mission_done:
                hover_active = True
                hover_end_time = t + hover_time
                if not silent:
                    print(f'\n  mission finished @ t={t:.2f}s · hovering '
                          f'for {hover_time:.2f}s')

            # Build the trajectory horizon (TrajectorySetpoints payload)
            # once per outer tick. Emitted on the first inner sub-step
            # below so the topic cadence matches the outer-loop rate
            # (100 Hz), comparable to aerostack2's
            # trajectory_generation_behavior publish cadence.
            trajectory_horizon_pts = []
            if emit_trajectory_horizon:
                for k in range(n_samples):
                    s = refs[k]
                    p = TrajectoryPoint()
                    p.position = np.asarray(s.position, dtype=float)
                    p.twist = np.asarray(s.velocity, dtype=float)
                    p.acceleration = np.asarray(s.acceleration, dtype=float)
                    p.yaw_angle = float(s.yaw)
                    trajectory_horizon_pts.append(p)
            first_inner_sub = True

            # Inner loop: INDI + physics at controller_dt -------------------
            t_inner = t
            while t_inner < t + outer_dt - 1e-9:
                t_sub = t_inner + controller_dt

                new_cmd = cmd_buffer.latest_available(t_sub)
                if new_cmd is not None:
                    current_cmd = new_cmd.cmd
                    current_cmd_compute_us = new_cmd.compute_time_us
                    current_cmd_delay_us = new_cmd.delay_applied_us
                new_ref = ref_buffer.latest_available(t_sub)
                if new_ref is not None:
                    current_ref = new_ref.sample
                    current_ref_update_us = new_ref.update_time_us
                    current_ref_eval_us = new_ref.eval_time_us
                    current_ref_delay_us = new_ref.delay_applied_us

                sim.set_reference_rates(
                    float(current_cmd.thrust_n),
                    np.asarray(current_cmd.angular_rate, dtype=float),
                )

                indi_t0 = time.perf_counter()
                sim.update_controller(controller_dt)
                indi_t1 = time.perf_counter()
                sim.update_imu(controller_dt)
                indi_t2 = time.perf_counter()

                t_model = t_inner
                while t_model < t_sub - 1e-9:
                    sim.update_model(model_dt)
                    t_model += model_dt
                indi_t3 = time.perf_counter()

                indi_times.append(indi_t1 - indi_t0)
                imu_times.append(indi_t2 - indi_t1)
                model_times.append(indi_t3 - indi_t2)

                t_inner = t_sub

                s = sim.state
                s_pos = np.asarray(s.position, dtype=float)
                s_ori = np.asarray(s.orientation, dtype=float)
                s_vel = np.asarray(s.linear_velocity, dtype=float)
                s_omega = np.asarray(s.angular_velocity, dtype=float)
                s_motor = np.asarray(s.motor_angular_velocity, dtype=float)

                err = s_pos - np.asarray(current_ref.position, dtype=float)
                tracking_sq_sum += float(np.dot(err, err))
                tracking_samples += 1

                if logger is not None:
                    row = LogRow(
                        time=t_sub,
                        position=s_pos,
                        orientation=s_ori,
                        linear_velocity=s_vel,
                        angular_velocity=s_omega,
                        reference_position=np.asarray(
                            scheduler.waypoint(active_index), dtype=float),
                        trajectory_position=np.asarray(
                            current_ref.position, dtype=float),
                        trajectory_velocity=np.asarray(
                            current_ref.velocity, dtype=float),
                        trajectory_orientation=_euler_to_quaternion(
                            0.0, 0.0, float(current_ref.yaw)),
                        thrust_n=float(current_cmd.thrust_n),
                        command_angular_velocity=np.asarray(
                            current_cmd.angular_rate, dtype=float),
                        motor_w=s_motor,
                        controller_compute_time_us=current_cmd_compute_us,
                        generator_update_time_us=current_ref_update_us,
                        generator_eval_time_us=current_ref_eval_us,
                        controller_delay_applied_us=current_cmd_delay_us,
                        generator_delay_applied_us=current_ref_delay_us,
                        # waypoint_index / experiment_active alignment
                        # (Phase 6). The mav_flight_review mask resolver
                        # picks `post_settling` when waypoint_index has
                        # multiple segments and `experiment_active`
                        # otherwise. For both backends to land on the
                        # SAME branch per combo:
                        #   * triangle (stepwise): publish the live
                        #     active_index (0..N-1), experiment_active=True
                        #     between takeoff and pre-land hover. Resolver
                        #     picks `post_settling`.
                        #   * moving_path (continuous): publish
                        #     waypoint_index=0 constant and
                        #     experiment_active=True during the moving-
                        #     target motion. Resolver picks
                        #     `experiment_active`.
                        waypoint_index=(0 if target_plan_active else active_index),
                        hover_active=hover_active,
                        # Stepwise gating uses mission_first_idx /
                        # mission_last_idx (computed once when the takeoff
                        # / land virtual waypoints are inserted) to keep
                        # transitions outside the paper metrics — mirrors
                        # aerostack2's experiment_active window.
                        # Continuous gating skips the synthetic takeoff
                        # and land legs of the target_plan by clamping the
                        # window to [t_at_wp[first], t_at_wp[last]].
                        experiment_active=(
                            (not hover_active
                             and t >= (
                                 self._example_cfg.target_start_delay_s
                                 + (float(target_t_at_wp[mission_first_idx])
                                    if mission_first_idx < len(target_t_at_wp)
                                    else 0.0))
                             and t <= (
                                 self._example_cfg.target_start_delay_s
                                 + (float(target_t_at_wp[mission_last_idx])
                                    if mission_last_idx < len(target_t_at_wp)
                                    else mission_end_t)) + 1e-9)
                            if target_plan_active
                            else (not hover_active
                                  and active_index >= mission_first_idx
                                  and active_index <= mission_last_idx)
                        ),
                        max_speed=max_speed,
                    )
                    if self._controller.provides_desired_velocity():
                        row.desired_velocity = np.asarray(
                            self._controller.last_desired_velocity(),
                            dtype=float).copy()
                        row.publishes_desired_velocity = True

                    # Match the C++ wrapper: keep mission-side topics
                    # silent during the synthetic takeoff/land transients
                    # so the reviewer's clip window aligns with the
                    # aerostack2 mission window.
                    row.publish_mission_signals = row.experiment_active

                    # debug/mission/reference/pose payload: as2 publishes
                    # the active waypoint (triangle) or the live moving-TF
                    # sample (moving_path). Mirror the split so the
                    # reviewer's pose_ref carries the same semantics in
                    # both backends.
                    if target_plan_active:
                        t_motion = max(0.0,
                                       t - self._example_cfg.target_start_delay_s)
                        t_target_pose = min(t_motion, target_plan_duration)
                        row.mission_pose_ref_position = target_plan_position(
                            t_target_pose)
                    else:
                        row.mission_pose_ref_position = np.asarray(
                            row.reference_position, dtype=float).copy()

                    # Mission-topic emission gates. aerostack2 publishes
                    # `debug/mission/reference/pose` at a fixed rate inside
                    # the goto / follow_reference loop and the latched
                    # mission topics only on value change. Mirror here:
                    #   * pose_ref: rate-limited and gated by the mission
                    #     window (`publish_mission_signals`).
                    #   * waypoint_index, max_speed: latched + mission
                    #     window (silent during synthetic takeoff/land).
                    #   * experiment_active, hover_active: latched
                    #     unconditionally so the False↔True transitions
                    #     are recorded outside the mission window too.
                    if (row.publish_mission_signals
                            and mission_pose_ref_period_s > 0.0
                            and (t_sub - t_last_mission_pose_ref_pub)
                            >= mission_pose_ref_period_s - 1e-9):
                        row.publish_mission_pose_ref = True
                        t_last_mission_pose_ref_pub = t_sub
                    if row.publish_mission_signals:
                        if (last_waypoint_index is None
                                or last_waypoint_index != row.waypoint_index):
                            row.publish_waypoint_index_change = True
                            last_waypoint_index = row.waypoint_index
                        if (last_max_speed is None
                                or last_max_speed != row.max_speed):
                            row.publish_max_speed_change = True
                            last_max_speed = row.max_speed
                    if (last_experiment_active is None
                            or last_experiment_active != row.experiment_active):
                        row.publish_experiment_active_change = True
                        last_experiment_active = row.experiment_active
                    if (last_hover_active is None
                            or last_hover_active != row.hover_active):
                        row.publish_hover_active_change = True
                        last_hover_active = row.hover_active

                    # `motion_reference/trajectory` mirrors aerostack2's
                    # trajectory_generation_behavior output: emit one
                    # TrajectorySetpoints per outer tick (gated to the
                    # mission window), only when the controller actually
                    # consumes a velocity/acceleration horizon.
                    if (emit_trajectory_horizon and first_inner_sub
                            and row.publish_mission_signals):
                        row.trajectory_horizon = trajectory_horizon_pts
                        row.publish_trajectory_horizon = True
                    first_inner_sub = False

                    logger.log_row(row)

            t += outer_dt
            if not silent:
                state_pos = np.asarray(sim.state.position, dtype=float)
                err_now = float(np.linalg.norm(
                    state_pos - np.asarray(current_ref.position, dtype=float)))
                print_status(t, max_sim_time, active_index, scheduler.size(),
                             err_now, current_cmd_compute_us)

        wall_end = time.perf_counter()
        real_time_s = wall_end - wall_start
        rmse_m = math.sqrt(tracking_sq_sum / tracking_samples) if tracking_samples > 0 else 0.0

        self._stats = BenchmarkStats(
            simulated_time_s=t,
            real_time_s=real_time_s,
            sim_speedup=(t / real_time_s) if real_time_s > 0.0 else 0.0,
            controller_mean_us=_mean_us(controller_times),
            generator_update_mean_us=_mean_us(generator_update_times),
            generator_eval_mean_us=_mean_us(generator_eval_times),
            indi_mean_us=_mean_us(indi_times),
            imu_mean_us=_mean_us(imu_times),
            model_mean_us=_mean_us(model_times),
            tracking_rmse_m=rmse_m,
            controller_steps=len(controller_times),
            indi_steps=len(indi_times),
        )

        if logger is not None:
            logger.close()

        if not silent:
            print_case_summary(
                self._stats.real_time_s,
                self._stats.tracking_rmse_m,
                self._stats.controller_mean_us,
                self._stats.generator_update_mean_us + self._stats.generator_eval_mean_us,
            )

    def print_benchmark(self) -> None:
        s = self._stats
        print(
            f'Simulated time      : {s.simulated_time_s} s\n'
            f'Real time           : {s.real_time_s} s\n'
            f'Sim speedup         : {s.sim_speedup}x\n'
            f'Controller compute  : {s.controller_mean_us} us (avg, '
            f'{s.controller_steps} steps)\n'
            f'Generator update    : {s.generator_update_mean_us} us (avg)\n'
            f'Generator eval      : {s.generator_eval_mean_us} us (avg)\n'
            f'INDI controller     : {s.indi_mean_us} us (avg)\n'
            f'IMU update          : {s.imu_mean_us} us (avg)\n'
            f'Physics model       : {s.model_mean_us} us (avg, per INDI step)\n'
            f'Tracking RMSE       : {s.tracking_rmse_m} m'
        )
