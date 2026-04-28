#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for :class:`examples_py.framework.WaypointsSimulator`."""

import os
import sys
import unittest

from mavpy.simulator.simulator_yaml import load_simulator_parameters_from_yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from examples_py.framework.factories import (  # noqa: E402
    ControllerKeys,
    GeneratorKeys,
    make_controller,
    make_generator,
)
from examples_py.framework.unified_mcap_logger import RunMetadata  # noqa: E402
from examples_py.framework.waypoints_simulator import WaypointsSimulator  # noqa: E402
from _helpers import load_test_sim_config, repo_path  # noqa: E402


def _make_meta() -> RunMetadata:
    m = RunMetadata()
    m.controller_name = 'pid'
    m.generator_name = 'waypoints'
    m.run_id = 'unit_test'
    m.language = 'py'
    return m


class WaypointsSimulatorTest(unittest.TestCase):

    def setUp(self) -> None:
        self.cfg = load_test_sim_config()
        self.sim_params = load_simulator_parameters_from_yaml(
            str(repo_path('configs/simulation/config_simulator.yaml')))

    def _make_pid(self):
        return make_controller(ControllerKeys.PID,
                               str(repo_path('configs/controllers/config_pid.yaml')))

    def _make_waypoints(self):
        return make_generator(GeneratorKeys.WAYPOINTS,
                              str(repo_path('configs/generators/config_waypoints.yaml')))

    def test_rejects_none_controller(self) -> None:
        with self.assertRaises(ValueError):
            WaypointsSimulator(None, self._make_waypoints(), self.cfg, self.sim_params,
                               '', _make_meta())

    def test_rejects_none_generator(self) -> None:
        with self.assertRaises(ValueError):
            WaypointsSimulator(self._make_pid(), None, self.cfg, self.sim_params,
                               '', _make_meta())

    def test_rejects_empty_waypoint_list(self) -> None:
        cfg = self.cfg
        cfg.waypoints = []
        with self.assertRaises(ValueError):
            WaypointsSimulator(self._make_pid(), self._make_waypoints(), cfg, self.sim_params,
                               '', _make_meta())

    def test_run_completes_and_reports_benchmark_stats(self) -> None:
        cfg = self.cfg
        cfg.sim_time = 0.5
        cfg.hover_time = 0.0
        cfg.silent = True
        cfg.benchmark = True
        cfg.output_format = 'mcap'

        sim = WaypointsSimulator(self._make_pid(), self._make_waypoints(), cfg,
                                 self.sim_params, '', _make_meta())
        sim.run()

        stats = sim.stats
        self.assertGreater(stats.simulated_time_s, 0.0)
        self.assertGreater(stats.controller_steps, 0)
        self.assertGreater(stats.indi_steps, 0)
        self.assertGreaterEqual(stats.controller_mean_us, 0.0)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
