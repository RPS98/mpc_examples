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

"""Plot results from the MPC + MAV Simulator integrated example CSV log."""

__authors__ = 'Rafael Perez-Segui'
__copyright__ = 'Copyright (c) 2025 Universidad Politecnica de Madrid'
__license__ = 'BSD-3-Clause'

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

PRINT_ERROR = True


def get_series(data, key, fallback_keys=None):
    """Return a data series using fallback keys when needed."""
    if fallback_keys is None:
        fallback_keys = []
    for candidate in [key] + fallback_keys:
        if candidate in data and len(data[candidate]) > 0:
            return data[candidate], candidate
    return [], key


def compute_mean_error(value_gt, value_ref):
    """Compute the mean absolute error."""
    return np.mean(np.abs(np.array(value_gt) - np.array(value_ref)))


def plot_values(data, values, title, axs):
    """Plot values vs time with optional reference overlay."""
    for i, value in enumerate(values):
        series_gt, label_gt = get_series(data, value)
        if not series_gt:
            print(f"Warn: no data found for '{value}'")
            continue
        axs[i].plot(data['time'], series_gt, linestyle='solid', label=label_gt)

        # Check for reference values
        fallback_ref = []
        if value in ['thrust', 'wx', 'wy', 'wz']:
            fallback_ref = [value + '_cmd', value + '_ref']
        series_ref, label_ref = get_series(data, value + '_ref', fallback_ref)
        if series_ref:
            axs[i].plot(data['time'], series_ref, linestyle='dotted', label=label_ref)
            if PRINT_ERROR:
                print(f'  Mean error {value}: {compute_mean_error(series_gt, series_ref):.6f}')

        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel(value)
        axs[i].set_title(f'{title} - {value}')
        axs[i].legend()
        axs[i].grid(True, alpha=0.3)


def plot_speed_magnitude(data, ax):
    """Plot speed magnitude |v| vs time."""
    vx, _ = get_series(data, 'vx')
    vy, _ = get_series(data, 'vy')
    vz, _ = get_series(data, 'vz')
    if not vx or not vy or not vz:
        return

    speed = np.sqrt(np.array(vx)**2 + np.array(vy)**2 + np.array(vz)**2)
    ax.plot(data['time'], speed, linestyle='solid', label='|v|')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('m/s')
    ax.set_title('Speed Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_drone_3d(position, orientation, axs):
    """Plot a 3D drone representation at a given pose."""
    arm_len = 0.1
    rotor_len = 0.05

    x, y, z = position
    qw, qx, qy, qz = orientation

    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])

    pos = np.array([x, y, z])
    heading = pos + R @ np.array([rotor_len, 0, 0])
    q1 = pos + R @ np.array([arm_len, arm_len, 0])
    q2 = pos + R @ np.array([-arm_len, -arm_len, 0])
    q3 = pos + R @ np.array([arm_len, -arm_len, 0])
    q4 = pos + R @ np.array([-arm_len, arm_len, 0])

    rotors = [q + R @ np.array([0, 0, rotor_len]) for q in [q1, q2, q3, q4]]

    axs.plot3D([q1[0], q2[0]], [q1[1], q2[1]], [q1[2], q2[2]], 'k')
    axs.plot3D([q3[0], q4[0]], [q3[1], q4[1]], [q3[2], q4[2]], 'k')
    for q, r in zip([q1, q2, q3, q4], rotors):
        axs.plot3D([q[0], r[0]], [q[1], r[1]], [q[2], r[2]], 'r')
    axs.plot3D([x, heading[0]], [y, heading[1]], [z, heading[2]], '-', color='orange')


def plot_trajectory_3d(data, axs):
    """Plot 3D trajectory with drone visualizations."""
    x_vals = data['x']
    y_vals = data['y']
    z_vals = data['z']

    axs.plot(x_vals, y_vals, z_vals, linestyle='solid', label='trajectory')

    x_ref, _ = get_series(data, 'x_ref')
    y_ref, _ = get_series(data, 'y_ref')
    z_ref, _ = get_series(data, 'z_ref')
    if x_ref and y_ref and z_ref:
        axs.plot(x_ref, y_ref, z_ref, linestyle='dashed', label='reference')

    # Plot drone at intervals
    num_steps = len(x_vals)
    interval = max(1, num_steps // 20)
    for step in range(0, num_steps, interval):
        position = np.array([x_vals[step], y_vals[step], z_vals[step]])
        orientation = np.array([
            data['qw'][step], data['qx'][step],
            data['qy'][step], data['qz'][step]])
        plot_drone_3d(position, orientation, axs)

    axs.set_xlabel('x (m)')
    axs.set_ylabel('y (m)')
    axs.set_zlabel('z (m)')
    axs.set_title('3D Trajectory')
    axs.legend()

    # Equal aspect ratio
    all_vals = np.concatenate([x_vals, y_vals, z_vals])
    max_range = max(abs(np.min(all_vals)), abs(np.max(all_vals)), 0.5)
    axs.set_xlim(-max_range, max_range)
    axs.set_ylim(-max_range, max_range)
    axs.set_zlim(0.0, max_range)


def read_csv(file_path):
    """Read the CSV file and return a dictionary with column data."""
    data = defaultdict(list)
    with open(file_path, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if isinstance(value, str) and value.strip():
                    data[key].append(float(value))

    if not data:
        return data

    # Validate consistent lengths
    time_len = len(data['time'])
    for key in data:
        if len(data[key]) != time_len:
            print(f'ERROR: key {key} has length {len(data[key])}, expected {time_len}')

    return data


def main():
    parser = argparse.ArgumentParser(description='Plot MPC + Simulator results')
    parser.add_argument(
        '-f', '--file_name', type=str, default='mpc_sim_log.csv',
        help='CSV log file (default: mpc_sim_log.csv)')
    args = parser.parse_args()

    file_path = os.path.abspath(args.file_name)
    print(f'Reading results from: {file_path}')

    data = read_csv(file_path)
    if not data or not data['time']:
        print('No data to plot')
        return

    # Figure 0: 3D trajectory
    fig0 = plt.figure(figsize=(8, 6))
    ax0 = fig0.add_subplot(projection='3d')
    plot_trajectory_3d(data, ax0)

    # Figure 1: Position, orientation, velocity
    fig1, axs1 = plt.subplots(3, 3, figsize=(14, 10))
    fig1.suptitle('State Tracking')
    print('Position errors:')
    plot_values(data, ['x', 'y', 'z'], 'Position', axs1[0, :])
    print('Orientation errors:')
    plot_values(data, ['roll', 'pitch', 'yaw'], 'Orientation', axs1[1, :])
    print('Velocity:')
    plot_values(data, ['vx', 'vy', 'vz'], 'Velocity', axs1[2, :])
    fig1.tight_layout()

    # Figure 2: Control inputs + angular velocity + speed
    fig2, axs2 = plt.subplots(2, 3, figsize=(14, 7))
    fig2.suptitle('Control & Angular Velocity')
    print('Angular velocity:')
    plot_values(data, ['wx', 'wy', 'wz'], 'Angular Velocity', axs2[0, :])

    # Thrust
    thrust_data, _ = get_series(data, 'thrust')
    if thrust_data:
        axs2[1, 0].plot(data['time'], thrust_data, label='thrust')
        axs2[1, 0].set_xlabel('Time (s)')
        axs2[1, 0].set_ylabel('N')
        axs2[1, 0].set_title('Thrust')
        axs2[1, 0].legend()
        axs2[1, 0].grid(True, alpha=0.3)

    plot_speed_magnitude(data, axs2[1, 1])

    # Motor angular velocities
    motor_keys = ['motor_w0', 'motor_w1', 'motor_w2', 'motor_w3']
    for mk in motor_keys:
        series, label = get_series(data, mk)
        if series:
            axs2[1, 2].plot(data['time'], series, label=label)
    axs2[1, 2].set_xlabel('Time (s)')
    axs2[1, 2].set_ylabel('rad/s')
    axs2[1, 2].set_title('Motor Angular Velocities')
    axs2[1, 2].legend()
    axs2[1, 2].grid(True, alpha=0.3)

    fig2.tight_layout()

    plt.show(block=False)
    plt.pause(0.001)

    if sys.stdin.isatty():
        print('\nPress [Enter] to close.')
        try:
            input()
        except KeyboardInterrupt:
            pass

    plt.close('all')
    print('Plotting finished')


if __name__ == '__main__':
    main()
