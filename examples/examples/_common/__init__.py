#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for the 12 Python example mains."""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

from .run_cli import (
    CommonArgs,
    SharedCfgArgs,
    ensure_examples_import_path,
    parse_full_args,
    parse_shared_args,
)

__all__ = [
    'CommonArgs',
    'SharedCfgArgs',
    'ensure_examples_import_path',
    'parse_full_args',
    'parse_shared_args',
]
