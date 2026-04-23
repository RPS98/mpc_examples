"""Smoke tests for MCAPViewer.

The viewer's ``save`` path exercises every entity log used by the interactive
``show`` path without requiring a GUI. A successful run produces a non-empty
``.rrd`` file.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from mav_flight_mcap import MCAPRecorder, euler_to_quaternion
from mav_flight_viewer import MCAPViewer, default_blueprint


@pytest.fixture()
def synthetic_log(tmp_path: Path) -> Path:
    path = tmp_path / "flight.mcap"
    with MCAPRecorder(path, n_motors=4,
                      extra_fields=["solve_time_us"]) as rec:
        for k in range(50):
            t = k * 0.02
            theta = 0.3 * t
            pos = np.array([math.cos(theta), math.sin(theta), 1.0 + 0.1 * t])
            vel = np.array([-math.sin(theta), math.cos(theta), 0.1])
            w = np.array([0.0, 0.0, 0.3])
            q = euler_to_quaternion(0.0, 0.0, theta)
            rec.save(
                t, pos, q, vel, w, pos, vel, q, w,
                9.81, w, np.full(4, 500.0),
                extras={"solve_time_us": 800.0 + k},
            )
    return path


def test_save_rrd_from_python_log(synthetic_log: Path, tmp_path: Path) -> None:
    out = tmp_path / "flight.rrd"
    viewer = MCAPViewer(synthetic_log, label="unit-test")
    result = viewer.save(out)
    assert result == out
    assert out.is_file()
    assert out.stat().st_size > 0


def test_default_blueprint_builds() -> None:
    bp = default_blueprint()
    assert bp is not None


def test_viewer_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MCAPViewer(tmp_path / "does_not_exist.mcap")
