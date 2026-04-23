#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Unit test for :class:`examples_py.controllers.PidGeometricController`."""

import math
import unittest

import numpy as np

from examples_py.controllers import PidGeometricController
from examples_py.framework import ReferenceField, has_field
from _helpers import (
    hover_state_at,
    horizon_at_position,
    load_test_sim_config,
    repo_path,
)


class PidGeometricControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sim_cfg = load_test_sim_config()
        self.state = hover_state_at([0.0, 0.0, 10.0])
        cfg = PidGeometricController.load_config_from_yaml(
            str(repo_path('configs/controllers/config_pid.yaml')))
        self.ctrl = PidGeometricController(cfg)
        self.ctrl.initialize(self.state, self.sim_cfg)

    def test_horizon_is_single_step(self) -> None:
        self.assertEqual(self.ctrl.reference_horizon_size(), 1)
        self.assertGreater(self.ctrl.control_period(), 0.0)

    def test_produces_finite_command_over_brief_simulation(self) -> None:
        refs = horizon_at_position([5.0, 0.0, 10.0],
                                   self.ctrl.reference_horizon_size())
        for k in range(50):
            cmd = self.ctrl.compute_command(self.state, refs)
            self.assertTrue(math.isfinite(cmd.thrust_n), f'step {k}')
            self.assertTrue(np.all(np.isfinite(cmd.angular_rate)), f'step {k}')
            self.assertGreater(cmd.thrust_n, 0.0, f'step {k}')

    def test_requires_position_reference(self) -> None:
        mask = self.ctrl.required_reference_fields()
        self.assertTrue(has_field(mask, ReferenceField.POSITION))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
