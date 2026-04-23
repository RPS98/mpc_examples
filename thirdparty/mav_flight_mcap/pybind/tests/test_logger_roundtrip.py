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

"""Roundtrip test: write from Python, read back with the `mcap` library."""

import numpy as np
import pytest

mcap_lib = pytest.importorskip('mcap')

import mav_flight_mcap as mfm  # noqa: E402

from mcap.reader import make_reader  # noqa: E402


def _write_minimal(path: str) -> None:
    cfg = mfm.LoggerConfig()
    cfg.file_path = path
    cfg.compress_zstd = False

    logger = mfm.MCAPLogger(cfg)
    logger.add_float64_topic('/drone0/debug/solve_time')
    logger.start()
    logger.save_pose_state(0.0, np.array([1.0, 2.0, 3.0]),
                           np.array([1.0, 0.0, 0.0, 0.0]))
    logger.save_twist_state(0.0, np.array([1.0, 0.0, 0.0]),
                            np.array([0.0, 0.0, 0.5]))
    logger.save_thrust_command(0.0, 9.81)
    logger.save_twist_command(0.0, np.array([0.1, 0.1, 0.1]))
    logger.save_float64('/drone0/debug/solve_time', 0.0, 1.4e-3)
    logger.close()


def test_schemas_and_channels_match_spec(tmp_path):
    """Read back with the `mcap` library and verify ROS 2 metadata."""
    path = str(tmp_path / 'roundtrip.mcap')
    _write_minimal(path)

    with open(path, 'rb') as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        schema_by_channel = {
            c.topic: summary.schemas[c.schema_id].name
            for c in summary.channels.values()
        }
        encodings_msg = {c.message_encoding for c in summary.channels.values()}
        encodings_sch = {s.encoding for s in summary.schemas.values()}

    assert encodings_msg == {'cdr'}
    assert encodings_sch == {'ros2msg'}
    assert schema_by_channel['/drone0/self_localization/pose'] == \
        'geometry_msgs/msg/PoseStamped'
    assert schema_by_channel['/drone0/self_localization/twist'] == \
        'geometry_msgs/msg/TwistStamped'
    assert schema_by_channel['/drone0/actuator_command/thrust'] == \
        'as2_msgs/msg/Thrust'
    assert schema_by_channel['/drone0/debug/solve_time'] == \
        'std_msgs/msg/Float64'
    assert schema_by_channel['/clock'] == 'rosgraph_msgs/msg/Clock'
