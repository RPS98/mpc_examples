#!/usr/bin/env python3

# Copyright 2025 Universidad Politecnica de Madrid
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
#    * Neither the name of the Universidad Politecnica de Madrid nor the names
#      of its contributors may be used to endorse or promote products derived
#      from this software without specific prior written permission.
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

"""
MPC + MAV Simulator integrated example.

Simulates a quadcopter following a series of waypoints using three nested
control loops running at different rates:

  - MPC (100 Hz, mpc_dt):
      Reads position/velocity/orientation from the simulator, sets the
      position reference, and solves the OCP to get thrust + angular
      velocity commands.

  - INDI + IMU (500 Hz, controller_dt):
      Converts thrust + angular velocity into motor commands (INDI) and
      updates the IMU noise model.

  - Physics model (1000 Hz, model_dt):
      Integrates rigid-body dynamics with the current motor commands.

The MPC output is held constant (zero-order hold) across the INDI sub-steps.

Usage:
    python3 examples/run_example.py \\
      -c config_example.yaml \\
      -s config_simulator.yaml \\
      -m config_mpc.yaml \\
      -f mpc_log.csv
"""

__authors__ = 'Rafael Perez-Segui'
__copyright__ = 'Copyright (c) 2025 Universidad Politecnica de Madrid'
__license__ = 'BSD-3-Clause'

import time

import numpy as np
from tqdm import tqdm

# mavpy: Python bindings for mav_simulator (C++ via pybind11)
from mavpy.simulator import Simulator, ControlMode
# mpc_position: position MPC built on acados
from mpc_position.mpc_controller import MPC
from mpc_position import configure_mpc_from_yaml

from utils.config_utils import (
    ExampleArgs,
    load_example_config,
    load_simulator_config,
    load_mpc_runtime_config,
    parse_arguments,
)
from utils.utils import (
    CsvLogger,
    compute_path_facing,
)


# ── Speed constraint ─────────────────────────────────────────────────────────

def update_speed_constraint(
        mpc: MPC,
        lh: np.ndarray,
        soft_speed_margin: float,
        max_speed: float) -> None:
    """Override the nonlinear constraint upper bound uh = (soft_speed_margin * max_speed)².

    The soft penalty starts penalizing when ||v|| exceeds soft_speed_margin * max_speed.
    """
    uh = np.array([(soft_speed_margin * max_speed) ** 2], dtype=float)
    mpc.get_nonlinear_constraint_bounds().set_bounds(lh, uh)
    mpc.update_nonlinear_constraint_bounds()


# ── Stage-dependent references ────────────────────────────────────────────────

def set_progressive_references(
        mpc_data,
        current_position: np.ndarray,
        goal_position: np.ndarray,
        desired_orientation: np.ndarray,
        v_ref: float,
        dt_horizon: float,
        N: int) -> None:
    """Set per-stage position references interpolating from current position to goal.

    Each stage k gets a reference at current_position + min((k+1)*v_ref*dt, L) * d_hat,
    where L is the distance to the goal and d_hat the unit direction. This spreads the
    references uniformly along the path at v_ref speed, clamping to the goal when reached.
    """
    d = goal_position - current_position
    L = np.linalg.norm(d)

    if L < 1e-9:
        mpc_data.parameters.set_desired_position(goal_position)
    else:
        d_hat = d / L
        for k in range(N + 1):
            s_k = min((k + 1) * v_ref * dt_horizon, L)
            mpc_data.parameters.set_desired_position(
                current_position + s_k * d_hat, stage=k)

    mpc_data.parameters.set_desired_orientation(desired_orientation)


# ── Orientation helper ────────────────────────────────────────────────────────

def get_desired_orientation(
        waypoint: np.ndarray,
        current_position: np.ndarray,
        current_orientation: np.ndarray,
        path_facing: bool) -> np.ndarray:
    """Compute desired orientation quaternion [w, x, y, z] for the current waypoint.

    When path_facing is enabled the drone yaws to face the direction of travel.
    The heading is held if the drone is within 0.1 m of the waypoint to avoid
    discontinuities.
    """
    if not path_facing:
        # No yaw tracking: keep north-facing (identity) orientation
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    diff = waypoint[:2] - current_position[:2]
    if np.linalg.norm(diff) < 0.1:
        # Close enough to waypoint: hold current orientation to avoid spin
        return current_orientation.copy()

    return compute_path_facing(diff)


# ── Main simulation loop ──────────────────────────────────────────────────────

def run(args: ExampleArgs) -> None:
    """Run the MPC + simulator integrated example."""

    # ── Load configuration ─────────────────────────────────────────────────────
    example_cfg = load_example_config(args.example_config_path)
    sim_params = load_simulator_config(args.simulator_config_path)
    mpc_runtime_cfg = load_mpc_runtime_config(args.mpc_config_path)

    # ── Derive loop step counts ────────────────────────────────────────────────
    # All rates must satisfy: model_dt | controller_dt | mpc_dt
    model_dt = example_cfg.model_dt          # 1000 Hz
    controller_dt = example_cfg.controller_dt  # 500 Hz
    mpc_dt = example_cfg.mpc_dt              # 100 Hz
    sim_time = example_cfg.sim_time
    total_time = sim_time + example_cfg.hover_time

    controller_steps_per_mpc = round(mpc_dt / controller_dt)
    model_steps_per_controller = round(controller_dt / model_dt)

    # ── Create simulator ───────────────────────────────────────────────────────
    sim = Simulator(sim_params)
    sim.arm()
    # RATES mode: the simulator expects (thrust, angular_velocity) as input.
    # The INDI controller converts these into motor commands each controller step.
    sim.set_control_mode(ControlMode.RATES)

    # ── Create MPC ─────────────────────────────────────────────────────────────
    mpc = MPC(ocp_json_file=mpc_runtime_cfg.ocp_json_file_path)
    configure_mpc_from_yaml(mpc, args.mpc_config_path)

    # Override uh with the value derived from soft_speed_margin and max_speed
    if mpc.lh_size > 0:
        lh = np.array([0.0], dtype=float)
        soft_speed_margin = float(mpc_runtime_cfg.soft_speed_margin)
        update_speed_constraint(mpc, lh, soft_speed_margin, float(example_cfg.max_speed))

    # mpc_data gives direct read/write access to state, parameters and actuation
    mpc_data = mpc.get_data()
    N = mpc.get_prediction_steps()
    dt_horizon = mpc.get_prediction_time_step()

    # ── Logger ─────────────────────────────────────────────────────────────────
    logger = CsvLogger(args.output_file)

    # ── Waypoint tracking ──────────────────────────────────────────────────────
    waypoints = example_cfg.waypoints
    wp_index = 0
    v_ref = float(example_cfg.max_speed)

    # ── MPC output (held between MPC steps via zero-order hold) ───────────────
    mpc_thrust = 0.0
    mpc_angular_velocity = np.zeros(3, dtype=float)
    # These are logged at the controller rate using the last MPC solution
    ref_position_log = np.zeros(3, dtype=float)
    ref_orientation_log = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    # ── Print summary ──────────────────────────────────────────────────────────
    print('=== MPC + Simulator Example ===')
    print(f'Total time        : {total_time} s')
    print(f'MPC dt            : {mpc_dt} s ({1.0 / mpc_dt:.0f} Hz)')
    print(f'MPC horizon       : N={N}, dt_h={dt_horizon:.4f} s, tf={N * dt_horizon:.2f} s')
    print(f'Controller dt     : {controller_dt} s ({1.0 / controller_dt:.0f} Hz)')
    print(f'Model dt          : {model_dt} s ({1.0 / model_dt:.0f} Hz)')
    print(f'Ctrl steps / MPC  : {controller_steps_per_mpc}')
    print(f'Model steps / Ctrl: {model_steps_per_controller}')
    print(f'Waypoints         : {len(waypoints)}')

    # Initial log entry at t=0 (drone at origin, all zeros)
    logger.save(
        0.0,
        np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3), np.zeros(3),
        np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]),
        0.0, np.zeros(3), np.zeros(4))

    # Timing statistics
    mpc_times = []

    pbar = tqdm(total=total_time, desc='Simulation', unit='s',
                bar_format='{l_bar}{bar} | {n:.3f}/{total:.1f} '
                '[{elapsed}<{remaining}, {rate_fmt}]')

    # ── Simulation loop ────────────────────────────────────────────────────────
    t = 0.0
    while t < total_time + 1e-9:

        # ── MPC step (runs at mpc_dt = 100 Hz) ──────────────────────────────
        mpc_start = time.perf_counter()

        # Read current ground-truth state from the physics model
        state = sim.state
        position = np.asarray(state.position, dtype=float)
        orientation = np.asarray(state.orientation, dtype=float)  # [w, x, y, z]
        velocity = np.asarray(state.linear_velocity, dtype=float)

        # Pack state into the MPC data structure
        mpc_data.state.position = position
        mpc_data.state.orientation = orientation
        mpc_data.state.linear_velocity = velocity

        # Set progressive per-stage references toward the current waypoint
        desired_position = waypoints[wp_index]
        desired_orientation = get_desired_orientation(
            desired_position, position, orientation, example_cfg.path_facing)

        set_progressive_references(
            mpc_data, position, desired_position,
            desired_orientation, v_ref, dt_horizon, N)

        # Cache for logging (shared across the INDI sub-steps below)
        ref_position_log = desired_position.copy()
        ref_orientation_log = desired_orientation.copy()

        # Solve the OCP (SQP-RTI: one linearization + one QP per call)
        mpc_status = mpc.solve()
        mpc_end = time.perf_counter()
        mpc_times.append(mpc_end - mpc_start)

        if mpc_status != 0:
            print(f'\nMPC solver failed with status {mpc_status} at time {t:.3f}s')
            break

        # Extract the first control action from the MPC solution
        mpc_thrust = float(mpc_data.actuation.thrust)
        mpc_angular_velocity = np.asarray(mpc_data.actuation.angular_velocity, dtype=float)

        # Advance waypoint index when the drone is within 0.1 m of the target
        error = np.linalg.norm(position - desired_position)
        if error < 0.1 and wp_index < len(waypoints) - 1:
            wp_index += 1
            print(f'\n  -> Waypoint {wp_index}: {waypoints[wp_index]}')

        # ── INDI + model sub-steps (zero-order hold on MPC output) ───────────
        # The MPC command is fixed for this block; INDI recalculates motor speeds
        # at each controller step to track the angular velocity reference.
        sim.set_reference_rates(mpc_thrust, mpc_angular_velocity)

        for ctrl_step in range(controller_steps_per_mpc):
            # INDI: thrust + omega_ref → motor angular velocity commands
            sim.update_controller(controller_dt)
            # IMU: advance gyro/accelerometer noise model
            sim.update_imu(controller_dt)

            # Physics model inner loop (smaller timestep for numerical accuracy)
            for _ in range(model_steps_per_controller):
                sim.update_model(model_dt)

            # Log at controller rate (500 Hz)
            s = sim.state
            log_time = t + (ctrl_step + 1) * controller_dt
            logger.save(
                log_time,
                np.asarray(s.position, dtype=float),
                np.asarray(s.orientation, dtype=float),
                np.asarray(s.linear_velocity, dtype=float),
                np.asarray(s.angular_velocity, dtype=float),
                ref_position_log,
                ref_orientation_log,
                mpc_thrust,
                mpc_angular_velocity,
                np.asarray(s.motor_angular_velocity, dtype=float))

        t += mpc_dt
        pbar.update(mpc_dt)

    pbar.close()
    logger.close()

    # ── Print statistics ───────────────────────────────────────────────────────
    mpc_avg = np.mean(mpc_times)
    print('\n=== Simulation finished ===')
    print(f'Simulated time      : {t:.3f} s')
    print(f'MPC avg solve time  : {mpc_avg * 1000.0:.3f} ms')
    if mpc_avg > 0.0:
        print(f'MPC real-time factor: {mpc_dt / mpc_avg:.1f}')
    print(f'Output: {args.output_file}')


if __name__ == '__main__':
    cli_args = parse_arguments()
    run(cli_args)
