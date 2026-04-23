"""Pytest conftest for :mod:`examples_py` adapter tests.

Inserts the ``examples_py/tests/`` directory at the front of ``sys.path`` so
individual test files can use ``import _helpers`` as a sibling module
without having to turn ``examples_py/tests`` into a nested sub-package of
``examples_py`` (which is mirrored under ``build/python/`` and would
otherwise not see the tests directory).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
