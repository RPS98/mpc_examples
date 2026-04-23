#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Unit test for :class:`examples_py.generators.JerkLimitedGenerator`."""

import unittest

import numpy as np

from examples_py.framework import ReferenceField, has_field
from examples_py.generators import JerkLimitedGenerator
from _helpers import (
    hover_state_at,
    load_test_sim_config,
    repo_path,
)


class JerkLimitedGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sim_cfg = load_test_sim_config()
        self.state = hover_state_at([0.0, 0.0, 10.0])
        cfg = JerkLimitedGenerator.load_config_from_yaml(
            str(repo_path('configs/generators/config_jerk_limited.yaml')))
        self.gen = JerkLimitedGenerator(cfg)
        self.gen.initialize(self.state, self.sim_cfg)

    def test_produces_finite_references_over_50_steps(self) -> None:
        self.gen.on_waypoint_changed([5.0, 0.0, 10.0], self.state, 0.0)
        for k in range(50):
            t = 0.01 * k
            self.gen.update(t, self.state)
            s = self.gen.evaluate(t)
            self.assertTrue(np.all(np.isfinite(s.position)), f'step {k}')
            self.assertTrue(np.all(np.isfinite(s.velocity)), f'step {k}')
            self.assertTrue(np.all(np.isfinite(s.acceleration)), f'step {k}')

    def test_advertises_trajectory_fields(self) -> None:
        mask = self.gen.provided_reference_fields()
        self.assertTrue(has_field(mask, ReferenceField.POSITION))
        self.assertTrue(has_field(mask, ReferenceField.VELOCITY))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
