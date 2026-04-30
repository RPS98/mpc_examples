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

    def test_starts_at_initial_position(self) -> None:
        p0 = np.asarray(self.state.position, dtype=float)
        target = p0 + np.array([5.0, 0.0, 0.0])
        self.gen.on_waypoint_changed(target, self.state, 0.0)
        s = self.gen.evaluate(0.0)
        self.assertLess(float(np.linalg.norm(s.position - p0)), 1e-3)
        self.assertLess(float(np.linalg.norm(s.velocity)), 1e-3)
        self.assertLess(float(np.linalg.norm(s.acceleration)), 1e-3)

    def test_reaches_target_past_duration(self) -> None:
        target = np.array([5.0, 0.0, 10.0])
        self.gen.on_waypoint_changed(target, self.state, 0.0)
        s = self.gen.evaluate(1000.0)
        self.assertLess(float(np.linalg.norm(s.position - target)), 1e-3)
        self.assertLess(float(np.linalg.norm(s.velocity)), 1e-9)
        self.assertLess(float(np.linalg.norm(s.acceleration)), 1e-9)

    def test_no_plan_when_waypoint_equals_current(self) -> None:
        p0 = np.asarray(self.state.position, dtype=float)
        self.gen.on_waypoint_changed(p0.copy(), self.state, 0.0)
        for t in (0.0, 1.0, 100.0):
            s = self.gen.evaluate(t)
            self.assertLess(float(np.linalg.norm(s.position - p0)), 1e-9, f't={t}')
            self.assertLess(float(np.linalg.norm(s.velocity)), 1e-9, f't={t}')
            self.assertLess(float(np.linalg.norm(s.acceleration)), 1e-9, f't={t}')

    def test_replan_resets_time_origin(self) -> None:
        p0 = np.asarray(self.state.position, dtype=float)

        self.gen.on_waypoint_changed([5.0, 0.0, 10.0], self.state, 0.0)
        s0 = self.gen.evaluate(0.0)
        self.assertLess(float(np.linalg.norm(s0.position - p0)), 1e-3)

        # Second segment starts at t=5; state still at p0 (scheduler may
        # fire before settle, same as the rest of p2p adapters).
        self.gen.on_waypoint_changed([-3.0, 4.0, 10.0], self.state, 5.0)
        s1 = self.gen.evaluate(5.0)
        self.assertLess(float(np.linalg.norm(s1.position - p0)), 1e-3)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
