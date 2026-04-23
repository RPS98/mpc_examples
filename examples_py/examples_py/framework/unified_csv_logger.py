#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Facade over :mod:`mav_flight_logger.csv_logger` kept under the framework
namespace for backwards compatibility with existing callers.

All the implementation lives in ``thirdparty/mav_flight_logger/`` and is
re-exported here so pure-Python mirrors can continue to import
``from examples_py.framework import UnifiedCsvLogger, LogRow, RunMetadata``
while higher-level code migrates to the thirdparty package directly.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

# TODO: remove once MCAP pipeline validated. Superseded by
#       ``examples_py.examples_py.framework.unified_mcap_logger``. The body
#       below is kept disabled via ``if False:`` so the file stays readable if
#       we need to re-enable the CSV backend during the transition.
if False:
    from mav_flight_logger.csv_logger import (
        COLUMN_COUNT,
        COLUMN_HEADER,
        CsvLogger as UnifiedCsvLogger,
        LogRow,
        RunMetadata,
        quaternion_to_euler,
    )

    _COLUMN_HEADER = COLUMN_HEADER

    __all__ = [
        'COLUMN_COUNT',
        'COLUMN_HEADER',
        'LogRow',
        'RunMetadata',
        'UnifiedCsvLogger',
        'quaternion_to_euler',
    ]
