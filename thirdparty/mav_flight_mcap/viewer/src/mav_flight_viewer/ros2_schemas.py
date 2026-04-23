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

"""Per-type decoders keyed by MCAP schema.name (ROS 2 Humble canonical form)."""

from __future__ import annotations

from typing import Any, Callable, Dict

from .cdr_decoder import CdrReader


def _read_time(r: CdrReader) -> Dict[str, int]:
    return {'sec': r.read_i32(), 'nanosec': r.read_u32()}


def _read_header(r: CdrReader) -> Dict[str, Any]:
    return {'stamp': _read_time(r), 'frame_id': r.read_string()}


def _read_vector3(r: CdrReader) -> Dict[str, float]:
    return {'x': r.read_f64(), 'y': r.read_f64(), 'z': r.read_f64()}


def _read_point(r: CdrReader) -> Dict[str, float]:
    return {'x': r.read_f64(), 'y': r.read_f64(), 'z': r.read_f64()}


def _read_quaternion(r: CdrReader) -> Dict[str, float]:
    return {'x': r.read_f64(), 'y': r.read_f64(),
            'z': r.read_f64(), 'w': r.read_f64()}


def _read_pose(r: CdrReader) -> Dict[str, Any]:
    return {'position': _read_point(r), 'orientation': _read_quaternion(r)}


def _read_twist(r: CdrReader) -> Dict[str, Any]:
    return {'linear': _read_vector3(r), 'angular': _read_vector3(r)}


def _decode_pose_stamped(r: CdrReader) -> Dict[str, Any]:
    return {'header': _read_header(r), 'pose': _read_pose(r)}


def _decode_twist_stamped(r: CdrReader) -> Dict[str, Any]:
    return {'header': _read_header(r), 'twist': _read_twist(r)}


def _decode_vector3(r: CdrReader) -> Dict[str, Any]:
    return _read_vector3(r)


def _decode_odometry(r: CdrReader) -> Dict[str, Any]:
    header = _read_header(r)
    child = r.read_string()
    pose = _read_pose(r)
    pose_cov = r.read_array(36, CdrReader.read_f64)
    twist = _read_twist(r)
    twist_cov = r.read_array(36, CdrReader.read_f64)
    return {
        'header': header,
        'child_frame_id': child,
        'pose': {'pose': pose, 'covariance': pose_cov},
        'twist': {'twist': twist, 'covariance': twist_cov},
    }


def _decode_thrust(r: CdrReader) -> Dict[str, Any]:
    return {
        'header': _read_header(r),
        'thrust': r.read_f32(),
        'thrust_normalized': r.read_f32(),
    }


def _read_trajectory_point(r: CdrReader) -> Dict[str, Any]:
    return {
        'id': r.read_string(),
        'position': _read_vector3(r),
        'twist': _read_vector3(r),
        'acceleration': _read_vector3(r),
        'yaw_angle': r.read_f32(),
    }


def _decode_trajectory_setpoints(r: CdrReader) -> Dict[str, Any]:
    return {
        'header': _read_header(r),
        'setpoints': r.read_sequence(_read_trajectory_point),
    }


def _decode_clock(r: CdrReader) -> Dict[str, Any]:
    return {'clock': _read_time(r)}


def _decode_int32(r: CdrReader) -> Dict[str, Any]:
    return {'data': r.read_i32()}


def _decode_string(r: CdrReader) -> Dict[str, Any]:
    return {'data': r.read_string()}


def _decode_float64(r: CdrReader) -> Dict[str, Any]:
    return {'data': r.read_f64()}


def _read_multi_array_dim(r: CdrReader) -> Dict[str, Any]:
    return {'label': r.read_string(), 'size': r.read_u32(), 'stride': r.read_u32()}


def _decode_float64_multi_array(r: CdrReader) -> Dict[str, Any]:
    layout = {
        'dim': r.read_sequence(_read_multi_array_dim),
        'data_offset': r.read_u32(),
    }
    data = r.read_sequence(CdrReader.read_f64)
    return {'layout': layout, 'data': data}


DECODERS: Dict[str, Callable[[CdrReader], Dict[str, Any]]] = {
    'geometry_msgs/msg/PoseStamped': _decode_pose_stamped,
    'geometry_msgs/msg/TwistStamped': _decode_twist_stamped,
    'geometry_msgs/msg/Vector3': _decode_vector3,
    'nav_msgs/msg/Odometry': _decode_odometry,
    'as2_msgs/msg/Thrust': _decode_thrust,
    'as2_msgs/msg/TrajectorySetpoints': _decode_trajectory_setpoints,
    'rosgraph_msgs/msg/Clock': _decode_clock,
    'std_msgs/msg/Int32': _decode_int32,
    'std_msgs/msg/String': _decode_string,
    'std_msgs/msg/Float64': _decode_float64,
    'std_msgs/msg/Float64MultiArray': _decode_float64_multi_array,
}


def decode(schema_name: str, payload: bytes) -> Dict[str, Any]:
    """Decode a CDR-encoded payload into a nested dict of primitives."""
    fn = DECODERS.get(schema_name)
    if fn is None:
        raise KeyError(f'No decoder registered for schema {schema_name!r}')
    return fn(CdrReader(payload))
