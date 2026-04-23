"""Roundtrip tests for MCAPRecorder.

These verify:

* That every declared topic appears in the output.
* That values written match what is read back (floating-point bitwise close).
* That missing extras are written as NaN.
* That invalid shapes / unknown extras / closed-file writes raise.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from mav_flight_mcap import MCAPRecorder, euler_to_quaternion


def _decode(path: Path) -> dict[str, list]:
    topics: dict[str, list] = {}
    with open(path, "rb") as fh:
        reader = make_reader(fh, decoder_factories=[DecoderFactory()])
        for _schema, channel, _msg, proto in reader.iter_decoded_messages():
            topics.setdefault(channel.topic, []).append(proto)
    return topics


@pytest.fixture()
def simple_log(tmp_path: Path) -> Path:
    path = tmp_path / "simple.mcap"
    with MCAPRecorder(path, n_motors=4,
                      extra_fields=["solve_time_us"]) as rec:
        for k in range(20):
            t = k * 0.05
            pos = np.array([float(k), 0.0, 1.0 + 0.1 * k])
            vel = np.array([1.0, 0.0, 0.1])
            w = np.array([0.0, 0.0, 0.2])
            q = euler_to_quaternion(0.0, 0.0, 0.1 * k)
            rec.save(
                t, pos, q, vel, w, pos, vel, q, w,
                9.81, w, np.full(4, 500.0),
                extras={"solve_time_us": 800.0 + k},
            )
    return path


def test_all_topics_present(simple_log: Path) -> None:
    topics = _decode(simple_log)
    assert set(topics) == {
        "/drone/state",
        "/drone/reference",
        "/drone/actuation",
        "/drone/extras/solve_time_us",
    }
    assert all(len(msgs) == 20 for msgs in topics.values())


def test_state_roundtrip(simple_log: Path) -> None:
    topics = _decode(simple_log)
    msg = topics["/drone/state"][10]
    assert msg.time == pytest.approx(0.5)
    assert msg.position.x == pytest.approx(10.0)
    assert msg.position.z == pytest.approx(2.0)
    assert msg.linear_velocity.x == pytest.approx(1.0)
    assert msg.angular_velocity.z == pytest.approx(0.2)


def test_extras_roundtrip(simple_log: Path) -> None:
    topics = _decode(simple_log)
    extras = topics["/drone/extras/solve_time_us"]
    assert extras[0].value == pytest.approx(800.0)
    assert extras[5].value == pytest.approx(805.0)


def test_missing_extras_become_nan(tmp_path: Path) -> None:
    path = tmp_path / "nan.mcap"
    with MCAPRecorder(path, n_motors=2, extra_fields=["foo"]) as rec:
        rec.save(
            0.0,
            np.zeros(3), euler_to_quaternion(0, 0, 0),
            np.zeros(3), np.zeros(3),
            np.zeros(3), np.zeros(3),
            euler_to_quaternion(0, 0, 0), np.zeros(3),
            0.0, np.zeros(3), np.zeros(2),
            extras=None,
        )
    topics = _decode(path)
    assert math.isnan(topics["/drone/extras/foo"][0].value)


def test_motor_size_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.mcap"
    with MCAPRecorder(path, n_motors=4) as rec:
        with pytest.raises(ValueError):
            rec.save(
                0.0, np.zeros(3), euler_to_quaternion(0, 0, 0),
                np.zeros(3), np.zeros(3),
                np.zeros(3), np.zeros(3),
                euler_to_quaternion(0, 0, 0), np.zeros(3),
                0.0, np.zeros(3), np.zeros(3),   # wrong size
            )


def test_unknown_extra_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.mcap"
    with MCAPRecorder(path, n_motors=0, extra_fields=["a"]) as rec:
        with pytest.raises(ValueError):
            rec.save(
                0.0, np.zeros(3), euler_to_quaternion(0, 0, 0),
                np.zeros(3), np.zeros(3),
                np.zeros(3), np.zeros(3),
                euler_to_quaternion(0, 0, 0), np.zeros(3),
                0.0, np.zeros(3), np.zeros(0),
                extras={"b": 1.0},
            )


def test_write_after_close_raises(tmp_path: Path) -> None:
    path = tmp_path / "closed.mcap"
    rec = MCAPRecorder(path, n_motors=0)
    rec.close()
    with pytest.raises(RuntimeError):
        rec.save(
            0.0, np.zeros(3), euler_to_quaternion(0, 0, 0),
            np.zeros(3), np.zeros(3),
            np.zeros(3), np.zeros(3),
            euler_to_quaternion(0, 0, 0), np.zeros(3),
            0.0, np.zeros(3), np.zeros(0),
        )


def test_context_manager_closes_file(tmp_path: Path) -> None:
    path = tmp_path / "ctx.mcap"
    with MCAPRecorder(path, n_motors=0) as rec:
        rec.save(
            0.0, np.zeros(3), euler_to_quaternion(0, 0, 0),
            np.zeros(3), np.zeros(3),
            np.zeros(3), np.zeros(3),
            euler_to_quaternion(0, 0, 0), np.zeros(3),
            0.0, np.zeros(3), np.zeros(0),
        )
    assert path.is_file() and path.stat().st_size > 0
