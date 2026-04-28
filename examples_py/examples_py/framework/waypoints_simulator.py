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
        sim.arm()
        sim.set_control_mode(ControlMode.RATES)

        initial_state = sim.state
        self._controller.initialize(initial_state, self._example_cfg)
        self._traj_gen.initialize(initial_state, self._example_cfg)

        # Scheduler --------------------------------------------------------
        scheduler = WaypointScheduler()
        scheduler.initialize(
            self._example_cfg.waypoints,
            np.asarray(initial_state.position, dtype=float),
            self._example_cfg.max_speed,
            self._example_cfg.settle_margin_s,
        )

        self._traj_gen.on_waypoint_changed(
            np.asarray(self._example_cfg.waypoints[0], dtype=float),
            initial_state, 0.0,
        )

        # Timing params ----------------------------------------------------
        model_dt = self._example_cfg.model_dt
        controller_dt = self._example_cfg.controller_dt
        outer_dt = self._controller.control_period()
        hover_time = self._example_cfg.hover_time
        mission_end_t = scheduler.final_time()
        max_sim_time = mission_end_t + hover_time
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

        hover_active = False
        hover_end_time = max_sim_time
        active_index = scheduler.active_index()

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

            # Generator step ------------------------------------------------
            gen_t0 = time.perf_counter()
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
            ref_buffer.push(ref_payload, t + gen_delay_s)

            # Controller step ----------------------------------------------
            ctrl_t0 = time.perf_counter()
            cmd = self._controller.compute_command(state, refs)
            ctrl_t1 = time.perf_counter()
            ctrl_solve_s = ctrl_t1 - ctrl_t0
            controller_times.append(ctrl_solve_s)

            ctrl_delay_s = _resolve_delay(
                ctrl_delay_mode, ctrl_solve_s, ctrl_delay_fixed)

            cmd_payload = _TimedCommand(
                cmd=cmd,
                compute_time_us=ctrl_solve_s * 1e6,
                delay_applied_us=ctrl_delay_s * 1e6,
            )
            cmd_buffer.push(cmd_payload, t + gen_delay_s + ctrl_delay_s)

            # Hover transition ---------------------------------------------
            if not hover_active and tick.finished and t >= mission_end_t - 1e-9:
                hover_active = True
                hover_end_time = t + hover_time
                if not silent:
                    print(f'\n  mission finished @ t={t:.2f}s · hovering '
                          f'for {hover_time:.2f}s')

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
                        waypoint_index=active_index,
                        hover_active=hover_active,
                        max_speed=max_speed,
                    )
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
