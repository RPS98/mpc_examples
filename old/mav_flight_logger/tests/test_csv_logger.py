#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ``mav_flight_logger.csv_logger`` (Python side).

Written with :mod:`unittest` so they can run either via
``python -m unittest discover`` or through pytest's auto-discovery of
``unittest.TestCase``.
"""

import os
import tempfile
import unittest

import numpy as np

from mav_flight_logger import (
    COLUMN_COUNT,
    COLUMN_HEADER,
    CsvLogger,
    LogRow,
    RunMetadata,
    quaternion_to_euler,
)


class ColumnHeaderTest(unittest.TestCase):
    def test_has_45_columns(self) -> None:
        self.assertEqual(COLUMN_COUNT, 45)
        self.assertEqual(COLUMN_HEADER.count(','), 44)


class QuaternionToEulerTest(unittest.TestCase):
    def test_identity_returns_zero(self) -> None:
        euler = quaternion_to_euler(np.array([1.0, 0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(euler, 0.0))


class CsvLoggerTest(unittest.TestCase):
    def test_writes_header_and_single_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'run.csv')
            md = RunMetadata(controller_name='pid',
                             generator_name='waypoints',
                             run_id='20260422_120000',
                             language='py')
            with CsvLogger(path, md) as logger:
                row = LogRow(time=0.01, thrust_n=9.81,
                             waypoint_index=2, hover_active=True, max_speed=3.0)
                row.position = np.array([1.0, 2.0, 3.0])
                logger.write_row(row)

            with open(path) as f:
                lines = f.read().splitlines()

            self.assertGreaterEqual(len(lines), 6)
            self.assertEqual(lines[0], '# controller: pid')
            self.assertEqual(lines[1], '# generator: waypoints')
            self.assertEqual(lines[2], '# run_id: 20260422_120000')
            self.assertEqual(lines[3], '# language: py')
            self.assertEqual(lines[4], COLUMN_HEADER)
            # Data row has exactly 44 commas -> 45 fields.
            self.assertEqual(lines[5].count(','), 44)

    def test_creates_nested_directory_if_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'nested', 'deep', 'run.csv')
            md = RunMetadata(controller_name='pid', generator_name='waypoints',
                             run_id='r', language='py')
            with CsvLogger(path, md):
                pass
            self.assertTrue(os.path.exists(path))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
