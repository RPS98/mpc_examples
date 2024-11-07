#!/usr/bin/env python3

# Copyright 2024 Universidad Politécnica de Madrid
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

"""Multirotor simulator example."""

__authors__ = 'Rafael Pérez Seguí'
__copyright__ = 'Copyright (c) 2022 Universidad Politécnica de Madrid'
__license__ = 'BSD-3-Clause'

from functools import wraps

import math
from mpc.mpc_controller import MPC, mpc_lib
import multirotor_simulator.multirotor_simulator as ms
import numpy as np
from utils.multirotor_utils import CsvLogger, SimParams, get_multirotor_simulator
from examples.utils.utils import get_trajectory_generator
import time
from tqdm import tqdm


def Euler_to_quaternion(roll: float, pitch: float, yaw: float) -> list:
    """
    Convert Euler angles to a quaternion.

    :param roll (float): The roll angle in radians.
    :param pitch (float): The pitch angle in radians.
    :param yaw (float): The yaw angle in radians.

    :return (Quaternion): The resulting quaternion.
    """
    # Calculate half angles
    roll_half = roll * 0.5
    pitch_half = pitch * 0.5
    yaw_half = yaw * 0.5

    # Calculate the sine and cosine of the half angles
    sr = math.sin(roll_half)
    cr = math.cos(roll_half)
    sp = math.sin(pitch_half)
    cp = math.cos(pitch_half)
    sy = math.sin(yaw_half)
    cy = math.cos(yaw_half)

    # Calculate the quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    # Create the Quaternion object and return it
    return [w, x, y, z]


def trajectory_point_to_mpc_reference(trajectory_point):
    """Convert trajectory point to MPC reference."""
    ref_position, ref_velocity, _, ref_yaw = \
        trajectory_point
    return mpc_lib.CaState.get_state(
        position=ref_position,
        orientation=Euler_to_quaternion(0.0, 0.0, ref_yaw),
        linear_velocity=ref_velocity
    )


def progress_bar(func):
    @wraps(func)
    def wrapper(logger, mpc, simulator, trajectory_generator, sim_params, *args, **kwargs):
        sim_max_t = trajectory_generator.get_max_time()

        pbar = tqdm(total=sim_max_t, desc=f'Progress {func.__name__}', unit='iter',
                    bar_format='{l_bar}{bar} | {n:.4f}/{total:.2f} '
                    '[{elapsed}<{remaining}, {rate_fmt}]')

        result = func(logger, mpc, simulator, trajectory_generator, sim_params, pbar, *args, **kwargs)

        pbar.close()
        return result
    return wrapper


@progress_bar
def test_trajectory_controller(
        logger: CsvLogger,
        mpc: MPC,
        simulator: ms.Simulator,
        trajectory_generator,
        sim_params: SimParams,
        pbar):
    """Test trajectory controller."""
    # MPC params
    prediction_steps = mpc.prediction_steps
    prediction_horizon = mpc.prediction_horizon
    tf = prediction_horizon / prediction_steps

    # Get max and min time
    max_time = trajectory_generator.get_max_time()
    min_time = trajectory_generator.get_min_time()

    # Set control mode
    simulator.set_control_mode(ms.ControlMode.ACRO)
    ref_position, ref_velocity, ref_acceleration, ref_yaw = \
        trajectory_generator.evaluate_trajectory(min_time)
    simulator.set_reference_trajectory(
        ref_position,
        ref_velocity,
        ref_acceleration)
    simulator.set_reference_yaw_angle(ref_yaw)

    # Update simulator
    for i in range(10):
        simulator.update_controller(tf)
        simulator.update_dynamics(tf)
        simulator.update_imu(tf)
        simulator.update_inertial_odometry(tf)

    mpc_solve_times = np.zeros(0)

    t = 0.0  # Sim time in seconds
    logger.save(t, simulator)
    while t < trajectory_generator.get_max_time():
        t_eval = t
        reference_trajectory = np.zeros((prediction_steps+1, mpc.x_dim))
        first_trajectory_point = None
        for i in range(prediction_steps+1):
            if t_eval >= max_time:
                t_eval = max_time - tf
            elif t_eval <= min_time:
                t_eval = min_time
            trajectory_point = trajectory_generator.evaluate_trajectory(t_eval)
            reference_trajectory[i, :] = trajectory_point_to_mpc_reference(trajectory_point)
            t_eval += tf

            if first_trajectory_point is None:
                first_trajectory_point = trajectory_point

        # Current state
        orientation = simulator.get_state().kinematics.orientation
        state = mpc_lib.CaState.get_state(
            position=simulator.get_state().kinematics.position,
            orientation=np.array(
                [orientation.w, orientation.x, orientation.y, orientation.z]),
            linear_velocity=simulator.get_state().kinematics.linear_velocity)

        current_time = time.time()
        u0 = mpc.compute_control_action(
            state=state,
            reference_trajectory_intermediate=reference_trajectory[:-1],
            reference_trajectory_final=reference_trajectory[-1][:mpc.x_dim])
        mpc_solve_times = np.append(mpc_solve_times, time.time() - current_time)

        simulator.set_reference_acro(
            thrust=u0[0],
            angular_velocity=u0[1:])

        # Update references for debugging
        ref_position, ref_velocity, ref_acceleration, ref_yaw = \
            first_trajectory_point
        simulator.set_reference_trajectory(
            reference_position=ref_position,
            reference_velocity=ref_velocity,
            reference_acceleration=ref_acceleration)
        simulator.set_reference_yaw_angle(ref_yaw)

        # Update simulator
        simulator.update_controller(tf)
        simulator.update_dynamics(tf)
        simulator.update_imu(tf)
        simulator.update_inertial_odometry(tf)

        t += tf
        logger.save(t, simulator)
        pbar.update(tf)
    logger.close()


if __name__ == '__main__':
    # Params
    simulator, sim_params, mpc_params = \
        get_multirotor_simulator('mpc_examples/simulation_config.yaml')

    # Logger
    file_name = 'ms_mpc_log.csv'
    logger = CsvLogger(file_name)

    # MPC
    mpc = MPC(
        prediction_steps=100,
        prediction_horizon=0.5,
        params=mpc_params
    )

    # Trajectory generator
    trajectory_generator = get_trajectory_generator(
        initial_position=simulator.get_state().kinematics.position,
        waypoints=sim_params.trajectory_generator_waypoints,
        speed=sim_params.trajectory_generator_max_speed
    )
    trajectory_generator.set_path_facing(sim_params.path_facing)

    test_trajectory_controller(
        logger,
        mpc,
        simulator,
        trajectory_generator,
        sim_params)
