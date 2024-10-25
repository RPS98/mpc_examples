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

"""Example utils."""

__authors__ = 'Rafael Pérez Seguí'
__copyright__ = 'Copyright (c) 2022 Universidad Politécnica de Madrid'
__license__ = 'BSD-3-Clause'

import math
import yaml
import os
import numpy as np
from pyquaternion import Quaternion
import utils.trajectory_generator as dtb

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
    """
    Convert Euler angles to a quaternion.

    :param roll: Roll angle in radians.
    :param pitch: Pitch angle in radians.
    :param yaw: Yaw angle in radians.
    :return: Quaternion object.
    """
    # Calculate half angles
    roll_half = roll * 0.5
    pitch_half = pitch * 0.5
    yaw_half = yaw * 0.5

    # Calculate sine and cosine of the half angles
    sr = math.sin(roll_half)
    cr = math.cos(roll_half)
    sp = math.sin(pitch_half)
    cp = math.cos(pitch_half)
    sy = math.sin(yaw_half)
    cy = math.cos(yaw_half)

    # Calculate quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    # Return the quaternion
    q = Quaternion(w, x, y, z).normalised
    return np.array([q.w, q.x, q.y, q.z])

def compute_path_facing(velocity: np.ndarray) -> list:
    """
    Compute the quaternion facing based on velocity.
    
    :param velocity: 3D velocity vector.
    :return: Quaternion as a list [w, x, y, z].
    """
    yaw = math.atan2(velocity[1], velocity[0])
    pitch, roll = 0.0, 0.0

    q = euler_to_quaternion(roll, pitch, yaw)
    return np.array([q.w, q.x, q.y, q.z])

class YamlData:
    def __init__(self):
        self.sim_time = 0.0
        self.dt = 0.0
        self.trajectory_generator_max_speed = 0.0
        self.waypoints = []
        self.path_facing = False

def read_yaml_params(file_path: str):
    """
    Read YAML configuration file and populate YamlData object.
    
    :param file_path: Path to the YAML file.
    """
    if not os.path.exists(file_path):
        absolute_simulation_config_path = os.path.abspath(file_path)
        print(f"File {absolute_simulation_config_path} does not exist.")
        raise ValueError("File does not exist")

    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)

    data = YamlData()

    # Read simulation parameters
    data.sim_time = config["sim_config"]["sim_time"]
    data.dt = config["sim_config"]["dt"]
    data.trajectory_generator_max_speed = config["sim_config"]["trajectory_generator_max_speed"]

    for waypoint in config["sim_config"]["trajectory_generator_waypoints"]:
        data.waypoints.append(np.array([waypoint[0], waypoint[1], waypoint[2]]))

    data.path_facing = config["sim_config"]["path_facing"]

    return data

def get_trajectory_generator(initial_position: np.ndarray, waypoints: list, speed: float):
    """
    Initialize and generate a dynamic trajectory.

    :param initial_position: Initial position of the vehicle.
    :param waypoints: List of waypoints.
    :param speed: Speed of the trajectory generator.
    :return: Generated trajectory.
    """
    # Initialize dynamic trajectory generator
    trajectory_generator = dtb.DynamicTrajectory()

    # Set waypoints
    trajectory_generator.set_path_facing(False)
    trajectory_generator.generate_trajectory(
        initial_position,
        0.0,
        waypoints,
        speed)

    # Generate trajectory
    max_time = trajectory_generator.get_max_time()

    print(f"Trajectory generated with max time: {max_time}")
    return trajectory_generator


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
            'time,'
            'x,y,z,qw,qx,qy,qz,vx,vy,vz,'
            'x_ref,y_ref,z_ref,qw_ref,qx_ref,qy_ref,qz_ref,vx_ref,vy_ref,vz_ref'
            'thrust_ref,wx_ref,wy_ref,wz_ref\n')

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

    def save(self, time: float, x: np.ndarray, y: np.ndarray, u: np.ndarray) -> None:
        """
        Save the simulation data to the csv file.

        :param time(float): Current simulation time.
        :param simulator(ms.Simulator): Simulator object.
        """
        self.add_double(time)

        # State
        self.add_vector_row(x)

        # Reference
        self.add_vector_row(y)

        # Control
        self.add_vector_row(u, False)

        # End line
        self.file.write('\n')

    def close(self) -> None:
        """Close the csv file."""
        self.file.close()
