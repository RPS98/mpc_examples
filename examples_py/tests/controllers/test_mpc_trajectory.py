#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Unit test for :class:`examples_py.controllers.MpcTrajectoryController`."""

import math
import unittest

import numpy as np

from examples_py.controllers import MpcTrajectoryController
from _helpers import (
    hover_state_at,
    horizon_at_position,
    load_test_sim_config,
    repo_path,
)


class MpcTrajectoryControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sim_cfg = load_test_sim_config()
        self.state = hover_state_at([0.0, 0.0, 10.0])
        cfg = MpcTrajectoryController.load_config_from_yaml(
            str(repo_path('configs/controllers/config_mpc_trajectory.yaml')))
        self.ctrl = MpcTrajectoryController(cfg)
        self.ctrl.initialize(self.state, self.sim_cfg)

    def test_horizon_is_multi_step(self) -> None:
        self.assertGreater(self.ctrl.reference_horizon_size(), 1)
        self.assertGreater(self.ctrl.reference_horizon_dt(), 0.0)

    def test_produces_finite_command_over_brief_simulation(self) -> None:
        refs = horizon_at_position([5.0, 0.0, 10.0],
                                   self.ctrl.reference_horizon_size())
        for k in range(5):
            cmd = self.ctrl.compute_command(self.state, refs)
            self.assertTrue(math.isfinite(cmd.thrust_n), f'step {k}')
            self.assertTrue(np.all(np.isfinite(cmd.angular_rate)), f'step {k}')
            self.assertGreater(cmd.thrust_n, 0.0, f'step {k}')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
