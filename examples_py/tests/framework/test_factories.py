#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :mod:`examples_py.framework.factories`."""

import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from examples_py.framework.factories import (  # noqa: E402
    ControllerKeys,
    GeneratorKeys,
    default_controller_config_path,
    default_generator_config_path,
    make_controller,
    make_generator,
)
from _helpers import repo_path  # noqa: E402


class FactoriesTest(unittest.TestCase):

    def test_default_paths_for_each_key(self) -> None:
        self.assertEqual(default_controller_config_path(ControllerKeys.PID),
                         'configs/controllers/config_pid.yaml')
        self.assertEqual(default_controller_config_path(ControllerKeys.MPC_POSITION),
                         'configs/controllers/config_mpc.yaml')
        self.assertEqual(default_controller_config_path(ControllerKeys.MPC_TRAJECTORY),
                         'configs/controllers/config_mpc_trajectory.yaml')

        for key, expected in [
            (GeneratorKeys.WAYPOINTS, 'configs/generators/config_waypoints.yaml'),
            (GeneratorKeys.JERK_LIMITED, 'configs/generators/config_jerk_limited.yaml'),
            (GeneratorKeys.GCOPTER, 'configs/generators/config_gcopter.yaml'),
            (GeneratorKeys.DYNAMIC, 'configs/generators/config_dynamic.yaml'),
            (GeneratorKeys.MAV_TRAJ_GEN, 'configs/generators/config_mav_traj_gen.yaml'),
        ]:
            self.assertEqual(default_generator_config_path(key), expected)

    def test_unknown_keys_raise(self) -> None:
        with self.assertRaises(ValueError):
            default_controller_config_path('does_not_exist')
        with self.assertRaises(ValueError):
            default_generator_config_path('does_not_exist')
        with self.assertRaises(ValueError):
            make_controller('does_not_exist')
        with self.assertRaises(ValueError):
            make_generator('does_not_exist')

    def test_makes_pid_and_waypoints_from_yaml(self) -> None:
        pid = make_controller(ControllerKeys.PID,
                              str(repo_path('configs/controllers/config_pid.yaml')))
        self.assertIsNotNone(pid)
        self.assertNotEqual(pid.name(), '')

        wp = make_generator(GeneratorKeys.WAYPOINTS,
                            str(repo_path('configs/generators/config_waypoints.yaml')))
        self.assertIsNotNone(wp)

    def test_makes_all_five_trajectory_generators(self) -> None:
        for key, path in [
            (GeneratorKeys.WAYPOINTS, 'configs/generators/config_waypoints.yaml'),
            (GeneratorKeys.JERK_LIMITED, 'configs/generators/config_jerk_limited.yaml'),
            (GeneratorKeys.GCOPTER, 'configs/generators/config_gcopter.yaml'),
            (GeneratorKeys.DYNAMIC, 'configs/generators/config_dynamic.yaml'),
            (GeneratorKeys.MAV_TRAJ_GEN, 'configs/generators/config_mav_traj_gen.yaml'),
        ]:
            gen = make_generator(key, str(repo_path(path)))
            self.assertIsNotNone(gen, msg=f'key={key}')
            self.assertNotEqual(gen.name(), '', msg=f'key={key}')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
