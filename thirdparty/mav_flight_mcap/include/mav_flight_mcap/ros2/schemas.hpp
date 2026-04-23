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
 * @file schemas.hpp
 *
 * ros2msg schema text for every message type written by the logger, formatted
 * exactly as rosbag2 Humble embeds schemas in MCAP files:
 *   <root .msg text>
 *   ================================================================================
 *   MSG: <pkg>/<Type>
 *   <dependency .msg text>
 *   ...
 *
 * Each composite type (e.g. PoseStamped) includes all its transitive
 * dependencies, topologically ordered.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MAV_FLIGHT_MCAP__ROS2__SCHEMAS_HPP_
#define MAV_FLIGHT_MCAP__ROS2__SCHEMAS_HPP_

#include <string_view>

namespace mav_flight_mcap {
namespace ros2 {
namespace schemas {

/**
 * @brief Canonical separator between a root message and its dependencies in
 * the multi-part ros2msg schema format used by rosbag2.
 */
inline constexpr std::string_view kSeparator =
    "================================================================================\n";

// --- Type names (used as MCAP schema.name) ------------------------------------

inline constexpr std::string_view kTypePoseStamped       = "geometry_msgs/msg/PoseStamped";
inline constexpr std::string_view kTypeTwistStamped      = "geometry_msgs/msg/TwistStamped";
inline constexpr std::string_view kTypeVector3           = "geometry_msgs/msg/Vector3";
inline constexpr std::string_view kTypeOdometry          = "nav_msgs/msg/Odometry";
inline constexpr std::string_view kTypeThrust            = "as2_msgs/msg/Thrust";
inline constexpr std::string_view kTypeTrajectorySetpoints =
    "as2_msgs/msg/TrajectorySetpoints";
inline constexpr std::string_view kTypeClock             = "rosgraph_msgs/msg/Clock";
inline constexpr std::string_view kTypeInt32             = "std_msgs/msg/Int32";
inline constexpr std::string_view kTypeString            = "std_msgs/msg/String";
inline constexpr std::string_view kTypeFloat64           = "std_msgs/msg/Float64";
inline constexpr std::string_view kTypeFloat64MultiArray = "std_msgs/msg/Float64MultiArray";

// --- Schema text (root + dependencies), verbatim .msg content ------------------

/** @brief geometry_msgs/msg/PoseStamped multi-part schema. */
inline constexpr std::string_view kSchemaPoseStamped =
    "# A Pose with reference coordinate frame and timestamp\n"
    "Header header\n"
    "Pose pose\n"
    "\n"
    "================================================================================\n"
    "MSG: std_msgs/Header\n"
    "# Standard metadata for higher-level stamped data types.\n"
    "builtin_interfaces/Time stamp\n"
    "string frame_id\n"
    "\n"
    "================================================================================\n"
    "MSG: builtin_interfaces/Time\n"
    "int32 sec\n"
    "uint32 nanosec\n"
    "\n"
    "================================================================================\n"
    "MSG: geometry_msgs/Pose\n"
    "# A representation of pose in free space, composed of position and orientation.\n"
    "Point position\n"
    "Quaternion orientation\n"
    "\n"
    "================================================================================\n"
    "MSG: geometry_msgs/Point\n"
    "# This contains the position of a point in free space\n"
    "float64 x\n"
    "float64 y\n"
    "float64 z\n"
    "\n"
    "================================================================================\n"
    "MSG: geometry_msgs/Quaternion\n"
    "# This represents an orientation in free space in quaternion form.\n"
    "float64 x 0\n"
    "float64 y 0\n"
    "float64 z 0\n"
    "float64 w 1\n";

/** @brief geometry_msgs/msg/TwistStamped multi-part schema. */
inline constexpr std::string_view kSchemaTwistStamped =
    "# A twist with reference coordinate frame and timestamp\n"
    "Header header\n"
    "Twist twist\n"
    "\n"
    "================================================================================\n"
    "MSG: std_msgs/Header\n"
    "builtin_interfaces/Time stamp\n"
    "string frame_id\n"
    "\n"
    "================================================================================\n"
    "MSG: builtin_interfaces/Time\n"
    "int32 sec\n"
    "uint32 nanosec\n"
    "\n"
    "================================================================================\n"
    "MSG: geometry_msgs/Twist\n"
    "# This expresses velocity in free space broken into its linear and angular parts.\n"
    "Vector3 linear\n"
    "Vector3 angular\n"
    "\n"
    "================================================================================\n"
    "MSG: geometry_msgs/Vector3\n"
    "# This represents a vector in free space.\n"
    "float64 x\n"
    "float64 y\n"
    "float64 z\n";

/** @brief geometry_msgs/msg/Vector3 (standalone, no dependencies). */
inline constexpr std::string_view kSchemaVector3 =
    "# This represents a vector in free space.\n"
    "float64 x\n"
    "float64 y\n"
    "float64 z\n";

/** @brief nav_msgs/msg/Odometry multi-part schema. */
inline constexpr std::string_view kSchemaOdometry =
    "# This represents an estimate of a position and velocity in free space.\n"
    "# The pose in this message should be specified in the coordinate frame given by header.frame_id\n"
    "# The twist in this message should be specified in the coordinate frame given by the child_frame_id\n"
    "Header header\n"
    "string child_frame_id\n"
    "PoseWithCovariance pose\n"
    "TwistWithCovariance twist\n"
    "\n"
    "================================================================================\n"
    "MSG: std_msgs/Header\n"
    "builtin_interfaces/Time stamp\n"
    "string frame_id\n"
    "\n"
    "================================================================================\n"
    "MSG: builtin_interfaces/Time\n"
    "int32 sec\n"
    "uint32 nanosec\n"
    "\n"
    "================================================================================\n"
    "MSG: geometry_msgs/PoseWithCovariance\n"
    "# This represents a pose in free space with uncertainty.\n"
    "Pose pose\n"
    "# Row-major representation of the 6x6 covariance matrix.\n"
    "float64[36] covariance\n"
    "\n"
    "================================================================================\n"
    "MSG: geometry_msgs/Pose\n"
    "Point position\n"
    "Quaternion orientation\n"
    "\n"
    "================================================================================\n"
    "MSG: geometry_msgs/Point\n"
    "float64 x\n"
    "float64 y\n"
    "float64 z\n"
    "\n"
    "================================================================================\n"
    "MSG: geometry_msgs/Quaternion\n"
    "float64 x 0\n"
    "float64 y 0\n"
    "float64 z 0\n"
    "float64 w 1\n"
    "\n"
    "================================================================================\n"
    "MSG: geometry_msgs/TwistWithCovariance\n"
    "# This expresses velocity in free space with uncertainty.\n"
    "Twist twist\n"
    "# Row-major representation of the 6x6 covariance matrix.\n"
    "float64[36] covariance\n"
    "\n"
    "================================================================================\n"
    "MSG: geometry_msgs/Twist\n"
    "Vector3 linear\n"
    "Vector3 angular\n"
    "\n"
    "================================================================================\n"
    "MSG: geometry_msgs/Vector3\n"
    "float64 x\n"
    "float64 y\n"
    "float64 z\n";

/** @brief as2_msgs/msg/Thrust multi-part schema. */
inline constexpr std::string_view kSchemaThrust =
    "# Message for encoding the desired thrust value\n"
    "std_msgs/Header header\n"
    "float32 thrust\n"
    "float32 thrust_normalized\n"
    "\n"
    "================================================================================\n"
    "MSG: std_msgs/Header\n"
    "builtin_interfaces/Time stamp\n"
    "string frame_id\n"
    "\n"
    "================================================================================\n"
    "MSG: builtin_interfaces/Time\n"
    "int32 sec\n"
    "uint32 nanosec\n";

/** @brief as2_msgs/msg/TrajectorySetpoints multi-part schema. */
inline constexpr std::string_view kSchemaTrajectorySetpoints =
    "# Definition of a trajectory as an array of setpoints\n"
    "std_msgs/Header header\n"
    "as2_msgs/TrajectoryPoint[] setpoints\n"
    "\n"
    "================================================================================\n"
    "MSG: std_msgs/Header\n"
    "builtin_interfaces/Time stamp\n"
    "string frame_id\n"
    "\n"
    "================================================================================\n"
    "MSG: builtin_interfaces/Time\n"
    "int32 sec\n"
    "uint32 nanosec\n"
    "\n"
    "================================================================================\n"
    "MSG: as2_msgs/TrajectoryPoint\n"
    "# Definition of a point of a trajectory\n"
    "string id\n"
    "geometry_msgs/Vector3 position\n"
    "geometry_msgs/Vector3 twist\n"
    "geometry_msgs/Vector3 acceleration\n"
    "float32 yaw_angle\n"
    "\n"
    "================================================================================\n"
    "MSG: geometry_msgs/Vector3\n"
    "float64 x\n"
    "float64 y\n"
    "float64 z\n";

/** @brief rosgraph_msgs/msg/Clock multi-part schema. */
inline constexpr std::string_view kSchemaClock =
    "# This message communicates the current time.\n"
    "builtin_interfaces/Time clock\n"
    "\n"
    "================================================================================\n"
    "MSG: builtin_interfaces/Time\n"
    "int32 sec\n"
    "uint32 nanosec\n";

/** @brief std_msgs/msg/Int32 (standalone). */
inline constexpr std::string_view kSchemaInt32 = "int32 data\n";

/** @brief std_msgs/msg/String (standalone). */
inline constexpr std::string_view kSchemaString = "string data\n";

/** @brief std_msgs/msg/Float64 (standalone). */
inline constexpr std::string_view kSchemaFloat64 = "float64 data\n";

/** @brief std_msgs/msg/Float64MultiArray multi-part schema. */
inline constexpr std::string_view kSchemaFloat64MultiArray =
    "# Please look at the MultiArrayLayout message definition for\n"
    "# documentation on all multiarrays.\n"
    "MultiArrayLayout layout\n"
    "float64[] data\n"
    "\n"
    "================================================================================\n"
    "MSG: std_msgs/MultiArrayLayout\n"
    "MultiArrayDimension[] dim\n"
    "uint32 data_offset\n"
    "\n"
    "================================================================================\n"
    "MSG: std_msgs/MultiArrayDimension\n"
    "string label\n"
    "uint32 size\n"
    "uint32 stride\n";

}  // namespace schemas
}  // namespace ros2
}  // namespace mav_flight_mcap

#endif  // MAV_FLIGHT_MCAP__ROS2__SCHEMAS_HPP_
