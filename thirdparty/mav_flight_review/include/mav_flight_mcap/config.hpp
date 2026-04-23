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
 * @file config.hpp
 *
 * Logger configuration: topic defaults, time mode, frame names.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MAV_FLIGHT_MCAP__CONFIG_HPP_
#define MAV_FLIGHT_MCAP__CONFIG_HPP_

#include <string>

namespace mav_flight_mcap {

/**
 * @brief Time source used when writing headers and MCAP message timestamps.
 *
 * - SIMULATION: the first save() call defines t = 0; subsequent timestamps are
 *   relative to that origin. This yields short-valued header stamps that are
 *   easy to interpret when replaying in a simulator.
 * - GLOBAL: the supplied time (seconds) is interpreted as an absolute POSIX
 *   time (wall clock); the logger passes it through as-is.
 */
enum class TimeMode {
  SIMULATION,
  GLOBAL,
};

/**
 * @brief Logger configuration.
 *
 * All topics default to the aerostack2 drone0 namespace. They can be changed
 * with the setter methods on MCAPLogger before start() is called.
 *
 * Frames follow the aerostack2 convention:
 * - `frame_earth` is used for global data (position, pose, odom pose,
 *   trajectory points).
 * - `frame_body`  is used for local data (body angular velocities,
 *   odom twist).
 */
struct LoggerConfig {
  /** @brief Output MCAP file path (must be writable). */
  std::string file_path;

  /** @brief How to resolve timestamps (SIMULATION or GLOBAL). */
  TimeMode time_mode = TimeMode::SIMULATION;

  /** @brief Frame ID used for global-frame topics (earth). */
  std::string frame_earth = "earth";

  /** @brief Frame ID used for body-frame topics. */
  std::string frame_body = "drone0/base_link";

  // --- Default topic names (aerostack2 spec) ---

  /** @brief geometry_msgs/PoseStamped desired position and yaw. */
  std::string pose_reference_topic = "/drone0/motion_reference/pose";

  /** @brief geometry_msgs/TwistStamped linear = max desired linear speed. */
  std::string twist_reference_topic = "/drone0/motion_reference/twist";

  /** @brief as2_msgs/TrajectorySetpoints desired trajectory with yaw. */
  std::string trajectory_reference_topic = "/drone0/motion_reference/trajectory";

  /** @brief as2_msgs/Thrust commanded thrust (N and normalized). */
  std::string thrust_command_topic = "/drone0/actuator_command/thrust";

  /** @brief geometry_msgs/TwistStamped angular = commanded body rates. */
  std::string twist_command_topic = "/drone0/actuator_command/twist";

  /** @brief geometry_msgs/PoseStamped current pose in earth. */
  std::string pose_state_topic = "/drone0/self_localization/pose";

  /** @brief geometry_msgs/TwistStamped current linear (earth) + angular (body). */
  std::string twist_state_topic = "/drone0/self_localization/twist";

  /** @brief nav_msgs/Odometry current state with zero covariance. */
  std::string odom_state_topic = "/drone0/sensor_measurements/odom";

  /** @brief rosgraph_msgs/Clock simulated clock. */
  std::string clock_topic = "/clock";

  // --- Clock policy ---

  /** @brief If true, emit a /clock message next to every save_*() call. */
  bool emit_clock_on_every_save = true;

  /**
   * @brief Minimum period between two consecutive /clock emissions (seconds).
   *
   * If 0, no throttling. Only takes effect when `emit_clock_on_every_save`.
   */
  double clock_min_period_s = 0.0;

  // --- MCAP writer policy ---

  /** @brief If true, write MCAP chunks with Zstandard compression. */
  bool compress_zstd = true;
};

}  // namespace mav_flight_mcap

#endif  // MAV_FLIGHT_MCAP__CONFIG_HPP_
