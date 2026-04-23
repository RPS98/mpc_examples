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

"""Smoke tests: module import, default config, and lifecycle invariants."""

import numpy as np
import pytest

import mav_flight_mcap as mfm


def test_module_exports():
    """The package must expose LoggerConfig, MCAPLogger, TimeMode, TrajectoryPoint."""
    assert mfm.LoggerConfig is not None
    assert mfm.MCAPLogger is not None
    assert mfm.TimeMode.SIMULATION is not None
    assert mfm.TimeMode.GLOBAL is not None
    assert mfm.TrajectoryPoint is not None


def test_default_config_matches_spec():
    """Defaults in LoggerConfig follow the aerostack2 drone0 spec."""
    c = mfm.LoggerConfig()
    assert c.pose_state_topic == '/drone0/self_localization/pose'
    assert c.twist_state_topic == '/drone0/self_localization/twist'
    assert c.odom_state_topic == '/drone0/sensor_measurements/odom'
    assert c.pose_reference_topic == '/drone0/motion_reference/pose'
    assert c.trajectory_reference_topic == '/drone0/motion_reference/trajectory'
    assert c.thrust_command_topic == '/drone0/actuator_command/thrust'
    assert c.twist_command_topic == '/drone0/actuator_command/twist'
    assert c.clock_topic == '/clock'
    assert c.frame_earth == 'earth'
    assert c.frame_body == 'drone0/base_link'


def test_set_topic_after_start_raises(tmp_path):
    """Topic setters must be rejected after start()."""
    c = mfm.LoggerConfig()
    c.file_path = str(tmp_path / 'lock.mcap')
    logger = mfm.MCAPLogger(c)
    logger.start()
    with pytest.raises(Exception):
        logger.set_pose_state_topic('/nope')
    with pytest.raises(Exception):
        logger.add_int32_topic('/nope')
    logger.close()


def test_shape_validation():
    """Shape mismatches in numpy inputs must produce a clear ValueError."""
    c = mfm.LoggerConfig()
    c.file_path = '/tmp/mfm_shape_test.mcap'
    logger = mfm.MCAPLogger(c)
    logger.start()
    with pytest.raises(ValueError):
        logger.save_pose_state(0.0, np.zeros(2), np.array([1, 0, 0, 0]))
    with pytest.raises(ValueError):
        logger.save_pose_state(0.0, np.zeros(3), np.array([1, 0, 0]))
    logger.close()


def test_context_manager(tmp_path):
    """The context manager starts on enter and closes on exit."""
    c = mfm.LoggerConfig()
    c.file_path = str(tmp_path / 'ctx.mcap')
    with mfm.MCAPLogger(c) as logger:
        assert logger.is_running is True
        logger.save_pose_state(0.0, np.zeros(3), np.array([1, 0, 0, 0]))
    assert logger.is_running is False
