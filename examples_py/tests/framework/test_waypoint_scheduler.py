#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :class:`examples_py.framework.WaypointScheduler`."""

import unittest

import numpy as np

from examples_py.framework.waypoint_scheduler import WaypointScheduler


_TOLERANCE = 1e-9


class WaypointSchedulerTest(unittest.TestCase):

    def test_computes_switch_times_from_distance_and_speed(self) -> None:
        wps = [np.array([10.0, 0.0, 0.0]),
               np.array([10.0, 5.0, 0.0]),
               np.array([10.0, 5.0, 5.0])]
        scheduler = WaypointScheduler()
        scheduler.initialize(wps, np.zeros(3), max_speed=5.0, settle_margin_s=1.0)

        # First hop covers 10 m at 5 m/s + 1 s margin.
        self.assertAlmostEqual(scheduler.switch_time(0), 10.0 / 5.0 + 1.0, delta=_TOLERANCE)
        self.assertAlmostEqual(scheduler.switch_time(1),
                               scheduler.switch_time(0) + 5.0 / 5.0 + 1.0, delta=_TOLERANCE)
        self.assertAlmostEqual(scheduler.switch_time(2),
                               scheduler.switch_time(1) + 5.0 / 5.0 + 1.0, delta=_TOLERANCE)
        self.assertEqual(scheduler.size(), len(wps))

    def test_tick_advances_index_and_flags_transitions_exactly_once(self) -> None:
        wps = [np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])]
        scheduler = WaypointScheduler()
        scheduler.initialize(wps, np.zeros(3), max_speed=1.0, settle_margin_s=0.0)

        r0 = scheduler.tick(0.5)
        self.assertFalse(r0.waypoint_changed)
        self.assertEqual(r0.active_index, 0)
        self.assertFalse(r0.finished)

        r1 = scheduler.tick(scheduler.switch_time(0) + 1e-3)
        self.assertTrue(r1.waypoint_changed)
        self.assertEqual(r1.active_index, 1)

        r2 = scheduler.tick(scheduler.switch_time(0) + 2e-3)
        self.assertFalse(r2.waypoint_changed)
        self.assertEqual(r2.active_index, 1)

        r3 = scheduler.tick(scheduler.switch_time(1) + 1e-3)
        self.assertTrue(r3.finished)

    def test_rejects_invalid_inputs(self) -> None:
        scheduler = WaypointScheduler()
        with self.assertRaises(ValueError):
            scheduler.initialize([], np.zeros(3), max_speed=1.0, settle_margin_s=0.0)
        with self.assertRaises(ValueError):
            scheduler.initialize([np.array([1.0, 0.0, 0.0])], np.zeros(3),
                                 max_speed=0.0, settle_margin_s=0.0)
        with self.assertRaises(ValueError):
            scheduler.initialize([np.array([1.0, 0.0, 0.0])], np.zeros(3),
                                 max_speed=1.0, settle_margin_s=-1.0)
        # scheduler_speed_factor must lie in (0, 1].
        for bad_factor in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                scheduler.initialize([np.array([1.0, 0.0, 0.0])], np.zeros(3),
                                     max_speed=1.0, settle_margin_s=0.0,
                                     scheduler_speed_factor=bad_factor)

    def test_speed_factor_inflates_segment_durations(self) -> None:
        wps = [np.array([10.0, 0.0, 0.0]), np.array([10.0, 5.0, 0.0])]
        scheduler = WaypointScheduler()
        scheduler.initialize(wps, np.zeros(3), max_speed=5.0, settle_margin_s=1.0,
                             scheduler_speed_factor=0.5)
        # First hop: 10 m / (5 m/s * 0.5) + 1 s = 5 s.
        self.assertAlmostEqual(scheduler.switch_time(0),
                               10.0 / (5.0 * 0.5) + 1.0, delta=_TOLERANCE)
        # Second hop adds 5 m / (5 m/s * 0.5) + 1 s = 3 s.
        self.assertAlmostEqual(scheduler.switch_time(1),
                               scheduler.switch_time(0) + 5.0 / (5.0 * 0.5) + 1.0,
                               delta=_TOLERANCE)

    def test_default_factor_matches_legacy_heuristic(self) -> None:
        s_default = WaypointScheduler()
        s_one = WaypointScheduler()
        wps = [np.array([3.0, 4.0, 0.0])]
        s_default.initialize(wps, np.zeros(3), max_speed=1.0, settle_margin_s=0.5)
        s_one.initialize(wps, np.zeros(3), max_speed=1.0, settle_margin_s=0.5,
                         scheduler_speed_factor=1.0)
        self.assertAlmostEqual(s_default.switch_time(0), s_one.switch_time(0),
                               delta=_TOLERANCE)
        self.assertAlmostEqual(s_default.switch_time(0), 5.0 + 0.5, delta=_TOLERANCE)

    def test_single_waypoint_mission_finishes_after_first_switch_time(self) -> None:
        scheduler = WaypointScheduler()
        scheduler.initialize([np.array([2.0, 0.0, 0.0])], np.zeros(3),
                             max_speed=1.0, settle_margin_s=0.5)

        r_before = scheduler.tick(0.0)
        self.assertFalse(r_before.finished)
        self.assertEqual(r_before.active_index, 0)

        r_after = scheduler.tick(scheduler.switch_time(0) + 1e-3)
        self.assertTrue(r_after.finished)
        self.assertEqual(r_after.active_index, 0)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
