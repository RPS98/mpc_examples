#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Unit test for :class:`examples_py.controllers.PidTrajectoryGeometricController`."""

import math
import unittest

import numpy as np

from examples_py.controllers import PidTrajectoryGeometricController
from examples_py.framework import ReferenceField, ReferenceSample, has_field
from _helpers import (
    hover_state_at,
    load_test_sim_config,
    repo_path,
)


class PidTrajectoryGeometricControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sim_cfg = load_test_sim_config()
        self.state = hover_state_at([0.0, 0.0, 10.0])
        cfg = PidTrajectoryGeometricController.load_config_from_yaml(
            str(repo_path('configs/controllers/config_pid_trajectory.yaml')))
        self.ctrl = PidTrajectoryGeometricController(cfg)
        self.ctrl.initialize(self.state, self.sim_cfg)

    def test_horizon_is_single_step(self) -> None:
        self.assertEqual(self.ctrl.reference_horizon_size(), 1)
        self.assertGreater(self.ctrl.control_period(), 0.0)

    def test_requires_position_and_velocity_reference(self) -> None:
        mask = self.ctrl.required_reference_fields()
        self.assertTrue(has_field(mask, ReferenceField.POSITION))
        self.assertTrue(has_field(mask, ReferenceField.VELOCITY))

    def test_produces_finite_command_over_brief_simulation(self) -> None:
        s = ReferenceSample()
        s.position = np.array([5.0, 0.0, 10.0])
        s.velocity = np.zeros(3)
        s.acceleration = np.zeros(3)
        s.yaw = 0.0
        refs = [s]
        for k in range(50):
            cmd = self.ctrl.compute_command(self.state, refs)
            self.assertTrue(math.isfinite(cmd.thrust_n), f'step {k}: thrust non-finite')
            self.assertTrue(np.all(np.isfinite(cmd.angular_rate)), f'step {k}: rates non-finite')
            self.assertGreater(cmd.thrust_n, 0.0, f'step {k}: thrust not positive')

    def test_finite_command_at_waypoint_transition_high_velocity_error(self) -> None:
        """Replicate the worst-case seen in mav_traj_gen run.

        Drone flying diagonally at ~3 m/s; new segment starts with ref_vel = 0.
        Velocity error ≈ [0, 2.8, 2.8] → Kd * error ≈ [0, 16.8, 16.8] m/s².
        """
        from mavpy.model import State
        moving = State()
        moving.position = [0.0, 10.0, 20.0]
        moving.orientation = [1.0, 0.0, 0.0, 0.0]
        moving.linear_velocity = [0.0, -2.8, -2.8]

        s = ReferenceSample()
        s.position = np.array([0.0, 10.0, 20.0])
        s.velocity = np.zeros(3)
        s.acceleration = np.zeros(3)
        s.yaw = 0.0
        refs = [s]

        for k in range(50):
            cmd = self.ctrl.compute_command(moving, refs)
            self.assertTrue(math.isfinite(cmd.thrust_n),
                            f'step {k}: thrust non-finite (acc_des ~[0,16.8,16.8] m/s²)')
            self.assertTrue(np.all(np.isfinite(cmd.angular_rate)),
                            f'step {k}: rates non-finite')
            self.assertGreater(cmd.thrust_n, 0.0, f'step {k}: thrust not positive')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
