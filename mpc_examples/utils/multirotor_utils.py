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

"""Log simulation data."""

__authors__ = 'Rafael Pérez Seguí'
__copyright__ = 'Copyright (c) 2022 Universidad Politécnica de Madrid'
__license__ = 'BSD-3-Clause'

import multirotor_simulator.multirotor_simulator as ms
import multirotor_simulator.multirotor_simulator_utils as ms_utils
import numpy as np
from dataclasses import dataclass, field
from mpc.mpc_controller import mpc_lib
from pyquaternion import Quaternion
import yaml


@dataclass
class SimParams:
    """
    Sim parameters.

    :param trajectory_generator_max_speed(float): Maximum speed of the
    trajectory generator.
    :param trajectory_generator_waypoints(np.ndarray): Waypoints of the
    trajectory generator.
    :param floor_height(float): Floor height.
    :param path_facing(bool): Path facing.
    """

    trajectory_generator_max_speed: float = 1.0
    trajectory_generator_waypoints: np.ndarray = \
        field(default_factory=lambda: np.zeros((1, 3)))
    floor_height: float = 0.0
    path_facing: bool = False


def read_simulator_params(config_file_path: str) \
        -> tuple[ms.SimulatorParams, SimParams]:
    """
    Read simulator parameters from a yaml file.

    :param config_file_path(str): Path to the yaml file.
    :return(tuple): A tuple with the simulator parameters and the simulation
    parameters.
    """
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Simulation parameters
    sim_params = config['sim_config']
    sim_params = SimParams(
        trajectory_generator_max_speed=(
            sim_params['trajectory_generator_max_speed']),
        trajectory_generator_waypoints=(
            np.array(sim_params['trajectory_generator_waypoints'])),
        floor_height=sim_params['floor_height'],
        path_facing=sim_params['path_facing']
    )

    # Dynamics state
    config_state = config['dynamics']['state']
    kinematics = ms.Kinematics(
        position=np.array(config_state['position']),
        orientation=Quaternion(config_state['orientation']),
        linear_velocity=np.array(config_state['linear_velocity']),
        angular_velocity=np.array(config_state['angular_velocity']),
        linear_acceleration=np.array(config_state['linear_acceleration']),
        angular_acceleration=np.array(config_state['angular_acceleration'])
    )
    actuators = ms.Actuators(
        motor_angular_velocity=\
            np.array(config_state['actuators_angular_speed'])
    )

    # Dynamics model
    config_model = config['dynamics']['model']
    config_motors = config_model['motors_params']
    model = ms.ModelParams(
        gravity=np.array(config_model['gravity']),
        vehicle_mass=config_model['vehicle_mass'],
        vehicle_inertia=np.diag(np.array(config_model['vehicle_inertia'])),
        vehicle_drag_coefficient=config_model['vehicle_drag_coefficient'],
        vehicle_aero_moment_coefficient=(
            np.diag(np.array(config_model['vehicle_aero_moment_coefficient']))),
        force_process_noise_auto_correlation=(
            config_model['force_process_noise_auto_correlation']),
        moment_process_noise_auto_correlation=(
            config_model['moment_process_noise_auto_correlation']),
        motors_params=ms.Model.create_quadrotor_x_config(
            thrust_coefficient=config_motors['thrust_coefficient'],
            torque_coefficient=config_motors['torque_coefficient'],
            x_dist=config_motors['x_dist'],
            y_dist=config_motors['y_dist'],
            min_speed=config_motors['min_speed'],
            max_speed=config_motors['max_speed'],
            time_constant=config_motors['time_constant'],
            rotational_inertia=config_motors['rotational_inertia'],
        )
    )

    # Controller params INDI
    config_controller = config['controller']
    config_indi = config_controller['indi']

    mixing_matrix_6D_4rotors = ms.Model.compute_mixer_matrix_6D(
        model.motors_params)
    matrix_inverse = ms.INDIController.compute_quadrotor_mixer_matrix_inverse(
        mixing_matrix_6D_4rotors)
    inertia = model.vehicle_inertia

    indi = ms.INDIControllerParams(
        inertia=inertia,
        mixer_matrix_inverse=matrix_inverse,
        pid_params=ms.PIDParams(
            kp=np.array(config_indi['Kp']),
            ki=np.array(config_indi['Ki']),
            kd=np.array(config_indi['Kd']),
            alpha=np.array(config_indi['alpha']),
            antiwindup_cte=np.array(config_indi['antiwindup_cte']),
            upper_output_saturation=(
                np.array(config_indi['angular_acceleration_limit'])),
            lower_output_saturation=(
                -np.array(config_indi['angular_acceleration_limit']))
        )
    )

    # IMU params
    config_imu = config['imu']

    imu = ms.IMUParams(
        gyro_noise_var=config_imu['gyro_noise_var'],
        accel_noise_var=config_imu['accel_noise_var'],
        gyro_bias_noise_autocorr_time=(
            config_imu['gyro_bias_noise_autocorr_time']),
        accel_bias_noise_autocorr_time=(
            config_imu['accel_bias_noise_autocorr_time'])
    )

    # Inertial Odometer params
    config_inertial_odometer = config['inertial_odometry']

    initial_world_orientation = kinematics.orientation

    inertial_odometer = ms.InertialOdometryParams(
        alpha=config_inertial_odometer['alpha'],
        initial_world_orientation=initial_world_orientation)

    # Simulation params
    dynamics_params = ms.DynamicsParams(
        model_params=model,
        state=ms.State(
            kinematics=kinematics,
            actuators=actuators
        ))

    controller_params = ms.ControllerParams(
        indi_controller_params=indi
    )

    # Multirotor simulator params
    simulation_params = ms.SimulatorParams(
        dynamics_params=dynamics_params,
        controller_params=controller_params,
        imu_params=imu,
        inertial_odometry_params=inertial_odometer,
    )

    # MPC params
    mpc_params = mpc_lib.AcadosMPCParams(
        Q=np.diag(np.array(config_controller['mpc']['Q'], dtype=np.float64)),
        Qe=np.diag(np.array(config_controller['mpc']['Qe'], dtype=np.float64)),
        R=np.diag(np.array(config_controller['mpc']['R'], dtype=np.float64)),
        lbu=np.array(config_controller['mpc']['lbu'], dtype=np.float64),
        ubu=np.array(config_controller['mpc']['ubu'], dtype=np.float64),
        p=np.array(config_controller['mpc']['p'], dtype=np.float64)
    )

    return simulation_params, sim_params, mpc_params


def get_multirotor_simulator(config_file_path: str) \
        -> tuple[ms.Simulator, SimParams, mpc_lib.AcadosMPCParams]:
    """
    Get a multirotor simulator with the parameters from a yaml file.

    :param config_file_path(str): Path to the yaml file.
    :return(tuple): A tuple with the multirotor simulator and the
    simulation
    """
    simulation_params, sim_params, mpc_params = read_simulator_params(config_file_path)
    simulator = ms.Simulator(simulation_params)

    # Enable floor collision
    simulator.enable_floor_collision(sim_params.floor_height)

    # Arm
    simulator.arm()
    return simulator, sim_params, mpc_params


class CsvLogger:
    """Log simulation data to a csv file."""

    def __init__(self, file_name: str) -> None:
        """
        Log simulation data to a csv file.

        :param file_name(str): Name of the file to save the data.
        """
        self.file_name = file_name
        print(f'Saving to file: {self.file_name}')
        self.file = open(self.file_name, 'w')
        self.file.write(
            'time,x,y,z,qw,qx,qy,qz,roll,pitch,yaw,vx,vy,vz,wx,wy,wz,ax,ay,az,dwx,dwy,dwz,'
            'fx,fy,fz,tx,ty,tz,mw1,mw2,mw3,mw4,mdw1,mdw2,mdw3,mdw4,x_ref,y_ref,'
            'z_ref,yaw_ref,vx_ref,vy_ref,vz_ref,wx_ref,wy_ref,'
            'wz_ref,ax_ref,ay_ref,az_ref,dwx_ref,dwy_ref,dwz_ref,fx_ref,fy_ref,fz_ref,'
            'tx_ref,ty_ref,tz_ref,mw1_ref,mw2_ref,mw3_ref,mw4_ref,'
            'x_io,y_io,z_io,roll_io,pitch_io,yaw_io,vx_io,vy_io,vz_io,wx_io,wy_io,'
            'wz_io,ax_io,ay_io,az_io\n')

    def add_double(self, data: float) -> None:
        """
        Add a double data to the csv file.

        :param data(float): Double data to add.
        """
        if data is None:
            raise ValueError('Data is None')
        self.file.write(f'{data},')

    def add_string(self, data: str, add_final_comma: bool = True) -> None:
        """
        Add a string data to the csv file.

        :param data(str): String data to add.
        :param add_final_comma(bool): Add a final comma to the string.
        """
        self.file.write(f'{data}')
        if add_final_comma:
            self.file.write(',')

    def add_vector_row(self, data: np.array, add_final_comma: bool = True) -> None:
        """
        Add a vector data to the csv file.

        :param data(np.array): Vector data to add.
        :param add_final_comma(bool): Add a final comma to the string.
        """
        for i in range(data.size):
            if data[i] is None:
                raise ValueError('Data is None')
            self.file.write(f'{data[i]}')
            if i < data.size - 1:
                self.file.write(',')
            elif add_final_comma:
                self.file.write(',')

    def save(self, time: float, simulator: ms.Simulator) -> None:
        """
        Save the simulation data to the csv file.

        :param time(float): Current simulation time.
        :param simulator(ms.Simulator): Simulator object.
        """
        self.add_double(time)

        # State ground truth
        state = simulator.get_state()
        state_string = f'{state}'
        self.add_string(state_string)

        # References
        # Reference trajectory generator
        self.add_vector_row(simulator.reference_position)  # Position
        self.add_double(simulator.reference_yaw)  # Yaw
        self.add_vector_row(simulator.reference_velocity)  # Velocity

        # Reference multirotor_controller
        # Angular velocity
        self.add_vector_row(simulator.controller.desired_angular_velocity)
        # Linear acceleration
        self.add_vector_row(simulator.reference_acceleration)
        # Angular acceleration
        self.add_vector_row(
            simulator.controller.indi_controller.desired_angular_acceleration)

        # Force reference compensated with the gravity and in earth frame
        force_ref = \
            state.kinematics.orientation.rotate(
                simulator.controller.indi_controller.desired_thrust +
                simulator.dynamics.model.gravity * simulator.dynamics.model.mass)
        # Force
        self.add_vector_row(force_ref)
        # Torque
        self.add_vector_row(simulator.controller.indi_controller.desired_torque)

        # Motor angular velocity
        self.add_vector_row(simulator.actuation_motors_angular_velocity)

        # State inertial odometry
        inertial_odometry = simulator.get_odometry_dict()
        io_pos = inertial_odometry['position']
        io_ori_q = inertial_odometry['orientation']
        io_ori = ms_utils.quaternion_to_Euler(io_ori_q)
        io_vel = inertial_odometry['linear_velocity']
        io_ang = inertial_odometry['angular_velocity']
        io_acc = inertial_odometry['linear_acceleration']
        io_data = \
            f'{io_pos[0]},{io_pos[1]},{io_pos[2]},' \
            f'{io_ori[0]},{io_ori[1]},{io_ori[2]},' \
            f'{io_vel[0]},{io_vel[1]},{io_vel[2]},' \
            f'{io_ang[0]},{io_ang[1]},{io_ang[2]},' \
            f'{io_acc[0]},{io_acc[1]},{io_acc[2]}'
        self.add_string(io_data, False)

        # End line
        self.file.write('\n')

    def close(self) -> None:
        """Close the csv file."""
        self.file.close()


if __name__ == '__main__':
    filename = 'multirotor_log.csv'
    logger = CsvLogger(filename)

    CONFIG_FILE = 'examples/simulation_config.yaml'
    simulator, sim_params = get_multirotor_simulator(CONFIG_FILE)

    t = 0.0  # Sim time in seconds
    dt = 0.001  # Time step in seconds
    for _ in range(1000):
        logger.save(t, simulator)
        simulator.update_controller(dt)
        simulator.update_dynamics(dt)
        simulator.update_imu(dt)
        simulator.update_inertial_odometry(dt)
        t += dt
