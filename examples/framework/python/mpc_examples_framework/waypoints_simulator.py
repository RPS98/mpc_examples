#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Triple-loop waypoint simulator driving an :class:`IController` with an
:class:`ITrajectoryGenerator`.

Python mirror of ``examples/framework/src/waypoints_simulator.cpp``. Delegates
physics, INDI and IMU updates to :class:`mavpy.simulator.Simulator` in
``RATES`` control mode and logs data via ``mpc_acados_core.logging.CsvLogger``.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from mavpy.controllers import ControlMode
from mavpy.simulator import Simulator, SimulatorParameters
from mpc_acados_core.logging.csv_logger import CsvLogger, euler_to_quaternion

from .controller_base import IController
from .example_config import ExampleConfig
from .trajectory_generator_base import ITrajectoryGenerator
from .types import ReferenceField, ReferenceSample, has_field


@dataclass
class BenchmarkStats:
    """Aggregate timing statistics produced by one :meth:`WaypointsSimulator.run`."""

    simulated_time_s: float = 0.0
    real_time_s: float = 0.0
    sim_speedup: float = 0.0
    controller_mean_us: float = 0.0
    indi_mean_us: float = 0.0
    imu_mean_us: float = 0.0
    model_mean_us: float = 0.0
    controller_steps: int = 0
    indi_steps: int = 0


def _describe_fields_mask(mask: ReferenceField) -> str:
    parts = []
    if has_field(mask, ReferenceField.POSITION):
        parts.append('position')
    if has_field(mask, ReferenceField.VELOCITY):
        parts.append('velocity')
    if has_field(mask, ReferenceField.ACCELERATION):
        parts.append('acceleration')
    return ', '.join(parts) if parts else '<none>'


def _is_finite_vec3(v: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(v)))


def _mean_of_seconds_in_us(seconds: List[float]) -> float:
    if not seconds:
        return 0.0
    return (sum(seconds) / len(seconds)) * 1e6


def _print_progress(fraction: float) -> None:
    fraction = max(0.0, min(1.0, fraction))
    bar_width = 40
    filled = int(bar_width * fraction)
    bar = '#' * filled + '-' * (bar_width - filled)
    sys.stdout.write(f'\r[{bar}] {fraction * 100.0:6.2f}%')
    sys.stdout.flush()


class WaypointsSimulator:
    """Orchestrates the triple-loop waypoint simulation.

    Construction takes ownership of the controller and the generator and pairs
    them with an ``mavpy.Simulator``. A compatibility warning is emitted once
    in the constructor if the generator does not cover every
    :class:`ReferenceField` required by the controller; the run proceeds
    anyway with missing fields held at zero (useful to study controller
    degradation under incomplete references).
    """

    def __init__(
        self,
        controller: IController,
        traj_gen: ITrajectoryGenerator,
        example_cfg: ExampleConfig,
        simulator_params: SimulatorParameters,
        output_csv: str,
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
        """Return the benchmark statistics collected by the last :meth:`run`."""
        return self._stats

    def run(self) -> None:
        """Execute the mission until ``sim_time + hover_time`` elapses."""
        sim = self._sim
        sim.arm()
        sim.set_control_mode(ControlMode.RATES)

        initial_state = sim.state
        self._controller.initialize(initial_state, self._example_cfg)
        self._traj_gen.initialize(
            self._example_cfg.waypoints, initial_state, self._example_cfg)

        model_dt = self._example_cfg.model_dt
        controller_dt = self._example_cfg.controller_dt
        outer_dt = self._controller.control_period()
        hover_time = self._example_cfg.hover_time
        max_sim_time = self._example_cfg.sim_time + hover_time
        benchmark = self._example_cfg.benchmark
        silent = self._example_cfg.silent
        max_speed = self._example_cfg.max_speed

        n_samples = self._controller.reference_horizon_size()
        dt_h = self._controller.reference_horizon_dt()
        refs: List[ReferenceSample] = [ReferenceSample() for _ in range(n_samples)]

        logger: Optional[CsvLogger] = None
        if not benchmark:
            logger = CsvLogger(self._output_csv)

        controller_times: List[float] = []
        indi_times: List[float] = []
        imu_times: List[float] = []
        model_times: List[float] = []

        prefix = '[BENCHMARK] ' if benchmark else ''
        print(
            f'{prefix}Controller   : {self._controller.name()}\n'
            f'Generator    : {self._traj_gen.name()}\n'
            f'Output file  : {self._output_csv}\n'
            f'Waypoints    : {len(self._example_cfg.waypoints)}\n'
            f'Horizon      : N={n_samples}, dt_h={dt_h} s\n'
            f'Outer period : {outer_dt} s\n'
            f'Max sim time : {max_sim_time} s\n'
        )
        print(
            f'Waypoint 1/{len(self._example_cfg.waypoints)}: '
            f'{self._example_cfg.waypoints[0]}'
        )

        zero3 = np.zeros(3)
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
        zero4 = np.zeros(4)

        if logger is not None:
            logger.save(
                0.0, zero3, identity_quat, zero3, zero3, zero3, identity_quat,
                0.0, zero3, zero4, 0.0, 0, False, max_speed,
            )

        ref0_snapshot = ReferenceSample()
        hover_active = False
        hover_end_time = max_sim_time
        last_wp_index = self._traj_gen.current_waypoint_index()

        wall_start = time.perf_counter()
        t = 0.0

        while t < max_sim_time + 1e-9:
            if hover_active and t >= hover_end_time - 1e-9:
                break

            state = sim.state
            position = np.asarray(state.position, dtype=float)
            orientation = np.asarray(state.orientation, dtype=float)
            velocity = np.asarray(state.linear_velocity, dtype=float)

            if (not _is_finite_vec3(position)
                    or not bool(np.all(np.isfinite(orientation)))
                    or not _is_finite_vec3(velocity)):
                sys.stderr.write(f'Non-finite state at t={t} s\n')
                break

            self._traj_gen.update(t, state)
            for k in range(n_samples):
                refs[k] = self._traj_gen.evaluate(t + k * dt_h)
            ref0_snapshot = refs[0]

            cmd = self._controller.compute_command(state, refs)
            solve_us = self._controller.last_solve_time_micros()
            controller_times.append(solve_us * 1e-6)

            current_wp = self._traj_gen.current_waypoint_index()
            if current_wp != last_wp_index:
                if not silent:
                    print(
                        f'\nWaypoint {current_wp + 1}/{len(self._example_cfg.waypoints)}: '
                        f'{self._example_cfg.waypoints[current_wp]}'
                    )
                last_wp_index = current_wp

            if not hover_active and self._traj_gen.is_finished(t):
                hover_active = True
                hover_end_time = t + hover_time
                if not silent:
                    print(f'\nMission finished. Hovering for {hover_time} s')

            ref_orientation = euler_to_quaternion(0.0, 0.0, ref0_snapshot.yaw)
            sim.set_reference_rates(float(cmd.thrust_n),
                                    np.asarray(cmd.angular_rate, dtype=float))

            t_ctrl = t
            while t_ctrl < t + outer_dt - 1e-9:
                indi_t0 = time.perf_counter()
                sim.update_controller(controller_dt)
                indi_t1 = time.perf_counter()

                sim.update_imu(controller_dt)
                indi_t2 = time.perf_counter()

                t_model = t_ctrl
                while t_model < t_ctrl + controller_dt - 1e-9:
                    sim.update_model(model_dt)
                    t_model += model_dt
                indi_t3 = time.perf_counter()

                indi_times.append(indi_t1 - indi_t0)
                imu_times.append(indi_t2 - indi_t1)
                model_times.append(indi_t3 - indi_t2)

                t_ctrl += controller_dt

                if logger is not None:
                    s = sim.state
                    logger.save(
                        t_ctrl,
                        np.asarray(s.position, dtype=float),
                        np.asarray(s.orientation, dtype=float),
                        np.asarray(s.linear_velocity, dtype=float),
                        np.asarray(s.angular_velocity, dtype=float),
                        np.asarray(ref0_snapshot.position, dtype=float),
                        np.asarray(ref_orientation, dtype=float),
                        float(cmd.thrust_n),
                        np.asarray(cmd.angular_rate, dtype=float),
                        np.asarray(s.motor_angular_velocity, dtype=float),
                        float(solve_us),
                        int(self._traj_gen.current_waypoint_index()),
                        bool(hover_active),
                        float(max_speed),
                    )

            t += outer_dt
            if not silent:
                _print_progress(t / max_sim_time)

        wall_end = time.perf_counter()
        real_time_s = wall_end - wall_start

        self._stats = BenchmarkStats(
            simulated_time_s=t,
            real_time_s=real_time_s,
            sim_speedup=(t / real_time_s) if real_time_s > 0.0 else 0.0,
            controller_mean_us=_mean_of_seconds_in_us(controller_times),
            indi_mean_us=_mean_of_seconds_in_us(indi_times),
            imu_mean_us=_mean_of_seconds_in_us(imu_times),
            model_mean_us=_mean_of_seconds_in_us(model_times),
            controller_steps=len(controller_times),
            indi_steps=len(indi_times),
        )

        if logger is not None:
            logger.close()

    def print_benchmark(self) -> None:
        """Pretty-print the :class:`BenchmarkStats` from the last run."""
        s = self._stats
        print(
            f'\n\nSimulated time      : {s.simulated_time_s} s\n'
            f'Real time           : {s.real_time_s} s\n'
            f'Sim speedup         : {s.sim_speedup}x\n'
            f'Controller compute  : {s.controller_mean_us} us (avg, '
            f'{s.controller_steps} steps)\n'
            f'INDI controller     : {s.indi_mean_us} us (avg)\n'
            f'IMU update          : {s.imu_mean_us} us (avg)\n'
            f'Physics model       : {s.model_mean_us} us (avg, per INDI step)'
        )
