#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Self-contained MAV flight telemetry logger / metrics / plotter package.

The package only depends on numpy (logger + metrics) and matplotlib (plots).
No other thirdparty submodule is referenced so it can be reused as-is from
other projects.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'
__version__ = '0.1.0'

from mav_flight_logger.csv_logger import (
    COLUMN_HEADER,
    COLUMN_COUNT,
    CsvLogger,
    LogRow,
    RunMetadata,
    quaternion_to_euler,
)

__all__ = [
    'COLUMN_HEADER',
    'COLUMN_COUNT',
    'CsvLogger',
    'LogRow',
    'RunMetadata',
    'quaternion_to_euler',
]
