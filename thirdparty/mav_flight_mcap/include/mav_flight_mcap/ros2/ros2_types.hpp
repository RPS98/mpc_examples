// Copyright 2025 Universidad Politécnica de Madrid
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

/**
 * @file ros2_types.hpp
 *
 * Plain-old-data C++ structs mirroring the ROS 2 Humble .msg types serialized
 * by this logger. Field names match the .msg definitions verbatim.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MAV_FLIGHT_MCAP__ROS2__ROS2_TYPES_HPP_
#define MAV_FLIGHT_MCAP__ROS2__ROS2_TYPES_HPP_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace mav_flight_mcap {
namespace ros2 {

/** @brief builtin_interfaces/msg/Time. */
struct Time {
  int32_t  sec     = 0;
  uint32_t nanosec = 0;
};

/** @brief std_msgs/msg/Header. */
struct Header {
  Time        stamp;
  std::string frame_id;
};

/** @brief geometry_msgs/msg/Vector3 (also used for Point via casting). */
struct Vector3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

/** @brief geometry_msgs/msg/Point. */
struct Point {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

/** @brief geometry_msgs/msg/Quaternion (CDR order: x, y, z, w). */
struct Quaternion {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double w = 1.0;
};

/** @brief geometry_msgs/msg/Pose. */
struct Pose {
  Point      position;
  Quaternion orientation;
};

/** @brief geometry_msgs/msg/PoseStamped. */
struct PoseStamped {
  Header header;
  Pose   pose;
};

/** @brief geometry_msgs/msg/Twist. */
struct Twist {
  Vector3 linear;
  Vector3 angular;
};

/** @brief geometry_msgs/msg/TwistStamped. */
struct TwistStamped {
  Header header;
  Twist  twist;
};

/** @brief geometry_msgs/msg/PoseWithCovariance. */
struct PoseWithCovariance {
  Pose                   pose;
  std::array<double, 36> covariance = {};
};

/** @brief geometry_msgs/msg/TwistWithCovariance. */
struct TwistWithCovariance {
  Twist                  twist;
  std::array<double, 36> covariance = {};
};

/** @brief nav_msgs/msg/Odometry. */
struct Odometry {
  Header              header;
  std::string         child_frame_id;
  PoseWithCovariance  pose;
  TwistWithCovariance twist;
};

/** @brief as2_msgs/msg/Thrust. */
struct Thrust {
  Header header;
  float  thrust            = 0.0f;
  float  thrust_normalized = 0.0f;
};

/** @brief as2_msgs/msg/TrajectoryPoint (internal 1:1 mirror). */
struct TrajectoryPointMsg {
  std::string id;
  Vector3     position;
  Vector3     twist;
  Vector3     acceleration;
  float       yaw_angle = 0.0f;
};

/** @brief as2_msgs/msg/TrajectorySetpoints. */
struct TrajectorySetpoints {
  Header                          header;
  std::vector<TrajectoryPointMsg> setpoints;
};

/** @brief rosgraph_msgs/msg/Clock. */
struct Clock {
  Time clock;
};

/** @brief std_msgs/msg/Int32. */
struct Int32 {
  int32_t data = 0;
};

/** @brief std_msgs/msg/String. */
struct String {
  std::string data;
};

/** @brief std_msgs/msg/Float64. */
struct Float64 {
  double data = 0.0;
};

/** @brief std_msgs/msg/MultiArrayDimension. */
struct MultiArrayDimension {
  std::string label;
  uint32_t    size   = 0;
  uint32_t    stride = 0;
};

/** @brief std_msgs/msg/MultiArrayLayout. */
struct MultiArrayLayout {
  std::vector<MultiArrayDimension> dim;
  uint32_t                         data_offset = 0;
};

/** @brief std_msgs/msg/Float64MultiArray. */
struct Float64MultiArray {
  MultiArrayLayout    layout;
  std::vector<double> data;
};

}  // namespace ros2
}  // namespace mav_flight_mcap

#endif  // MAV_FLIGHT_MCAP__ROS2__ROS2_TYPES_HPP_
