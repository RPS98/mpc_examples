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
 * @file mcap_logger.hpp
 *
 * MCAPLogger: public class that writes ROS 2 Humble-compatible MCAP files.
 *
 * Produces bags with `message_encoding="cdr"` and `schema_encoding="ros2msg"`,
 * byte-compatible with rosbag2 Humble on x86_64. No dependency on the ROS 2
 * runtime: Fast-CDR is used purely as a serialization library.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MAV_FLIGHT_MCAP__MCAP_LOGGER_HPP_
#define MAV_FLIGHT_MCAP__MCAP_LOGGER_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "mav_flight_mcap/config.hpp"
#include "mav_flight_mcap/trajectory_point.hpp"

namespace mav_flight_mcap {

/**
 * @brief ROS 2-compatible MCAP logger with typed per-topic save() methods.
 *
 * Lifecycle:
 *   1. Construct with a LoggerConfig. The MCAP file is NOT opened yet.
 *   2. (Optional) Change topic names via set_*_topic(...) and declare extra
 *      topics with add_*_topic(...).
 *   3. Call start(): opens the file, registers schemas/channels, locks the
 *      topic list (further set_/add_ calls throw std::logic_error).
 *   4. Call save_*(t, ...) as needed. Each call writes the message in CDR and
 *      (unless throttled) an accompanying /clock message with the same stamp.
 *   5. Call close() to flush and finalize. close() is idempotent.
 *
 * Not thread-safe.
 */
class MCAPLogger {
 public:
  /** @brief Construct the logger (does not open the file). */
  explicit MCAPLogger(const LoggerConfig& cfg);

  /** @brief Destructor: closes the file if still open. */
  ~MCAPLogger();

  MCAPLogger(const MCAPLogger&)            = delete;
  MCAPLogger& operator=(const MCAPLogger&) = delete;
  MCAPLogger(MCAPLogger&&)                 = delete;
  MCAPLogger& operator=(MCAPLogger&&)      = delete;

  // --- Topic setters (must be called before start()) ---

  /** @brief Override the pose_reference topic (PoseStamped). */
  void set_pose_reference_topic(const std::string& topic);
  /** @brief Override the twist_reference topic (TwistStamped, linear only). */
  void set_twist_reference_topic(const std::string& topic);
  /** @brief Override the trajectory_reference topic (TrajectorySetpoints). */
  void set_trajectory_reference_topic(const std::string& topic);
  /** @brief Override the thrust command topic (Thrust). */
  void set_thrust_command_topic(const std::string& topic);
  /** @brief Override the twist command topic (TwistStamped, angular only). */
  void set_twist_command_topic(const std::string& topic);
  /** @brief Override the pose state topic (PoseStamped). */
  void set_pose_state_topic(const std::string& topic);
  /** @brief Override the twist state topic (TwistStamped). */
  void set_twist_state_topic(const std::string& topic);
  /** @brief Override the odometry state topic (Odometry). */
  void set_odom_state_topic(const std::string& topic);

  // --- Extra typed topic registration (before start()) ---

  /** @brief Register an extra std_msgs/Int32 topic. */
  void add_int32_topic(const std::string& topic);
  /** @brief Register an extra std_msgs/String topic. */
  void add_string_topic(const std::string& topic);
  /** @brief Register an extra std_msgs/Float64 topic. */
  void add_float64_topic(const std::string& topic);
  /** @brief Register an extra std_msgs/Float64MultiArray topic. */
  void add_float64_multi_array_topic(const std::string& topic);
  /** @brief Register an extra geometry_msgs/Vector3 topic. */
  void add_vector3_topic(const std::string& topic);

  // --- Lifecycle ---

  /** @brief Open MCAP, register schemas/channels, lock topics. */
  void start();
  /** @brief Flush and close the MCAP file (idempotent). */
  void close();
  /** @brief True after start() and before close(). */
  bool isRunning() const;

  // --- Single-topic save methods ---

  /** @brief Save a PoseStamped on the pose_reference topic. */
  void save_pose_reference(double t,
                           const Eigen::Vector3d& pos,
                           const Eigen::Vector4d& quat_wxyz);
  /** @brief Save a TwistStamped (linear only) on the twist_reference topic. */
  void save_twist_reference(double t, const Eigen::Vector3d& linear);
  /** @brief Save a TrajectorySetpoints on the trajectory_reference topic. */
  void save_trajectory_reference(double t,
                                 const std::vector<TrajectoryPoint>& points);
  /** @brief Save a Thrust on the thrust_command topic. */
  void save_thrust_command(double t, double thrust);
  /** @brief Save a TwistStamped (angular only) on the twist_command topic. */
  void save_twist_command(double t, const Eigen::Vector3d& angular);
  /** @brief Save a PoseStamped on the pose_state topic. */
  void save_pose_state(double t,
                       const Eigen::Vector3d& pos,
                       const Eigen::Vector4d& quat_wxyz);
  /** @brief Save a TwistStamped on the twist_state topic. */
  void save_twist_state(double t,
                        const Eigen::Vector3d& linear,
                        const Eigen::Vector3d& angular);
  /** @brief Save an Odometry on the odom_state topic. */
  void save_odom_state(double t,
                       const Eigen::Vector3d& pos_earth,
                       const Eigen::Vector4d& quat_wxyz,
                       const Eigen::Vector3d& linear_body,
                       const Eigen::Vector3d& angular_body);

  // --- Aggregate saves (one call -> multiple topics) ---

  /**
   * @brief Save complete drone state in one call.
   *
   * Emits PoseStamped (pose_state), TwistStamped (twist_state) and Odometry
   * (odom_state) sharing the same timestamp. linear_earth is rotated into
   * body frame for the TwistStamped/Odometry twist field using quat_wxyz.
   *
   * @param t             Timestamp (s).
   * @param pos_earth     Position in earth frame.
   * @param quat_wxyz     Orientation quaternion [w, x, y, z] (earth -> body).
   * @param linear_earth  Linear velocity in earth frame.
   * @param angular_body  Angular velocity in body frame.
   */
  void save_state(double t,
                  const Eigen::Vector3d& pos_earth,
                  const Eigen::Vector4d& quat_wxyz,
                  const Eigen::Vector3d& linear_earth,
                  const Eigen::Vector3d& angular_body);

  /**
   * @brief Save complete position reference in one call.
   *
   * Emits PoseStamped (pose_reference) and TwistStamped (twist_reference,
   * linear only) with the same timestamp.
   */
  void save_position_reference(double t,
                               const Eigen::Vector3d& pos_ref,
                               const Eigen::Vector4d& quat_wxyz_ref,
                               const Eigen::Vector3d& max_linear_speed);

  /**
   * @brief Save complete trajectory reference.
   *
   * Always emits TrajectorySetpoints. If `also_emit_pose_and_twist` is true
   * and `points` is non-empty, also emits the first setpoint as PoseStamped
   * (position + yaw) and TwistStamped (linear speed) on the reference topics.
   */
  void save_trajectory_reference_full(double t,
                                      const std::vector<TrajectoryPoint>& points,
                                      bool also_emit_pose_and_twist = false);

  /**
   * @brief Save complete actuation (Thrust + angular command) in one call.
   */
  void save_actuation(double t,
                      double thrust,
                      const Eigen::Vector3d& angular_command_body);

  // --- Extras (typed) ---

  /** @brief Save a std_msgs/Int32 on a previously registered topic. */
  void save_int32(const std::string& topic, double t, int32_t value);
  /** @brief Save a std_msgs/String on a previously registered topic. */
  void save_string(const std::string& topic, double t, const std::string& value);
  /** @brief Save a std_msgs/Float64 on a previously registered topic. */
  void save_float64(const std::string& topic, double t, double value);
  /** @brief Save a geometry_msgs/Vector3 on a previously registered topic. */
  void save_vector3(const std::string& topic, double t,
                    const Eigen::Vector3d& value);
  /**
   * @brief Save a std_msgs/Float64MultiArray on a previously registered topic.
   *
   * @param dims Optional list of dimension sizes; if empty, a single dim of
   *   size=value.size() with stride=value.size() is used.
   */
  void save_float64_multi_array(const std::string& topic, double t,
                                const std::vector<double>& value,
                                const std::vector<uint32_t>& dims = {});

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mav_flight_mcap

#endif  // MAV_FLIGHT_MCAP__MCAP_LOGGER_HPP_
