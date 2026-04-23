# Copyright 2025 Universidad Politécnica de Madrid
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Cross-stack golden test: pybind writer -> mcap library -> viewer decoder.

The bytes written by the C++ logger through Fast-CDR must decode to the exact
field values when parsed with the viewer's pure-Python CdrReader. This locks
both sides against silent alignment regressions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('mcap')
pytest.importorskip('mav_flight_mcap._logger_cpp')

from mcap.reader import make_reader  # noqa: E402

import mav_flight_mcap as mfm  # noqa: E402
from mav_flight_viewer.ros2_schemas import decode  # noqa: E402


def _write(path: Path) -> None:
    cfg = mfm.LoggerConfig()
    cfg.file_path = str(path)
    cfg.compress_zstd = False

    logger = mfm.MCAPLogger(cfg)
    logger.add_float64_topic('/ext/f')
    logger.add_string_topic('/ext/s')
    logger.add_int32_topic('/ext/i')
    logger.add_vector3_topic('/ext/v')
    logger.add_float64_multi_array_topic('/ext/m')
    logger.start()

    logger.save_pose_state(0.0,
                           np.array([1.5, -2.25, 3.125]),
                           np.array([0.927362, 0.1, 0.2, 0.3]))  # w,x,y,z
    logger.save_odom_state(0.0,
                           np.array([1.0, 2.0, 10.0]),
                           np.array([1.0, 0.0, 0.0, 0.0]),
                           np.array([1.0, 0.0, 0.0]),
                           np.array([0.0, 0.0, 0.5]))
    logger.save_thrust_command(0.0, 7.25)
    logger.save_float64('/ext/f', 0.0, 3.14159265)
    logger.save_string('/ext/s', 0.0, 'hola mundo')
    logger.save_int32('/ext/i', 0.0, -42)
    logger.save_vector3('/ext/v', 0.0, np.array([0.1, 0.2, 0.3]))
    logger.save_float64_multi_array('/ext/m', 0.0, [1.0, 2.0, 3.0, 4.0])
    logger.close()


def _first_payload(path: Path, topic: str) -> tuple[str, bytes]:
    with open(path, 'rb') as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=[topic]):
            return schema.name, bytes(message.data)
    raise AssertionError(f'no messages on topic {topic!r}')


def test_pose_stamped_decodes(tmp_path):
    path = tmp_path / 'golden.mcap'
    _write(path)
    schema, data = _first_payload(path, '/drone0/self_localization/pose')
    out = decode(schema, data)
    assert out['header']['frame_id'] == 'earth'
    assert out['pose']['position'] == {'x': 1.5, 'y': -2.25, 'z': 3.125}
    # Quaternion stored in CDR order x, y, z, w.
    assert out['pose']['orientation']['w'] == pytest.approx(0.927362)
    assert out['pose']['orientation']['x'] == pytest.approx(0.1)
    assert out['pose']['orientation']['y'] == pytest.approx(0.2)
    assert out['pose']['orientation']['z'] == pytest.approx(0.3)


def test_odometry_covariance_zero(tmp_path):
    path = tmp_path / 'golden.mcap'
    _write(path)
    schema, data = _first_payload(path, '/drone0/sensor_measurements/odom')
    out = decode(schema, data)
    assert out['child_frame_id'] == 'drone0/base_link'
    assert len(out['pose']['covariance']) == 36
    assert all(c == 0.0 for c in out['pose']['covariance'])
    assert out['twist']['twist']['angular']['z'] == pytest.approx(0.5)


def test_thrust_float32_exact(tmp_path):
    path = tmp_path / 'golden.mcap'
    _write(path)
    schema, data = _first_payload(path, '/drone0/actuator_command/thrust')
    out = decode(schema, data)
    # 7.25 is exactly representable in float32.
    assert out['thrust'] == 7.25
    assert out['thrust_normalized'] == 0.0


def test_extras_roundtrip(tmp_path):
    path = tmp_path / 'golden.mcap'
    _write(path)

    _, data = _first_payload(path, '/ext/f')
    assert decode('std_msgs/msg/Float64', data)['data'] == pytest.approx(3.14159265)

    _, data = _first_payload(path, '/ext/s')
    assert decode('std_msgs/msg/String', data)['data'] == 'hola mundo'

    _, data = _first_payload(path, '/ext/i')
    assert decode('std_msgs/msg/Int32', data)['data'] == -42

    _, data = _first_payload(path, '/ext/v')
    v = decode('geometry_msgs/msg/Vector3', data)
    assert v == {'x': pytest.approx(0.1), 'y': pytest.approx(0.2), 'z': pytest.approx(0.3)}

    _, data = _first_payload(path, '/ext/m')
    m = decode('std_msgs/msg/Float64MultiArray', data)
    assert m['data'] == [1.0, 2.0, 3.0, 4.0]
    assert m['layout']['dim'][0]['size'] == 4
