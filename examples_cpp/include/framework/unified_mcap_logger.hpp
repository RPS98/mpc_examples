// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file unified_mcap_logger.hpp
 *
 * Thin facade over ``mav_flight_review::MCAPLogger`` that exposes the same
 * Eigen-native ``LogRow`` API previously served by the CSV facade so every
 * call-site (WaypointsSimulator, unit tests, etc.) can switch backends
 * without touching its row-building code. Each ``logRow`` call is mapped to
 * the matching ``save_*`` invocations on the underlying MCAP logger.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_FRAMEWORK_UNIFIED_MCAP_LOGGER_HPP_
#define MPC_EXAMPLES_FRAMEWORK_UNIFIED_MCAP_LOGGER_HPP_

#include <Eigen/Dense>

#include <memory>
#include <string>

#include "mav_flight_review/mcap_logger.hpp"

namespace mpc_examples::framework {

/**
 * @brief Run-level metadata emitted once as std_msgs/String messages under
 * ``/drone0/debug/mission/metadata/*`` at the beginning of the MCAP.
 *
 * Field semantics match the previous CSV backend so consumers that parsed the
 * ``# controller:`` / ``# generator:`` comment block can be ported 1:1 by
 * reading the matching string topics.
 */
struct RunMetadata {
  std::string controller_name;
  std::string generator_name;
  std::string run_id;            ///< e.g. "20260421_153012".
  std::string language = "cpp";  ///< "cpp" | "py".
};

/**
 * @brief Single telemetry row exposed to the framework in Eigen-native types.
 *
 * Mirrors the schema of the retired CSV facade so ``WaypointsSimulator`` does
 * not need to change the way it assembles rows. The facade translates these
 * fields into the ROS 2 topics registered by ``mav_flight_review::MCAPLogger``.
 */
struct LogRow {
  double time = 0.0;

  // State (earth frame) -------------------------------------------------------
  Eigen::Vector3d position         = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation   = Eigen::Quaterniond::Identity();
  Eigen::Vector3d linear_velocity  = Eigen::Vector3d::Zero();  ///< earth frame.
  Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();  ///< body frame.

  // Reference (earth frame) ---------------------------------------------------
  /// Active waypoint target (stepwise, no delay). Identical across every
  /// combination of controller/generator since the scheduler is deterministic
  /// and shares the same waypoints + max_speed + settle_margin_s. This is the
  /// **position reference** the vehicle is ultimately driven toward.
  Eigen::Vector3d reference_position = Eigen::Vector3d::Zero();

  /// Trajectory sample the controller consumes at ``t``: smooth generator
  /// output with the computation delay applied, or a virtual carrot that the
  /// controller advances along the waypoint path. Differs across generators.
  Eigen::Vector3d trajectory_position       = Eigen::Vector3d::Zero();
  Eigen::Vector3d trajectory_velocity       = Eigen::Vector3d::Zero();  ///< earth frame.
  Eigen::Quaterniond trajectory_orientation = Eigen::Quaterniond::Identity();

  // Actuation -----------------------------------------------------------------
  double thrust_n                          = 0.0;
  Eigen::Vector3d command_angular_velocity = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 4, 1> motor_w      = Eigen::Matrix<double, 4, 1>::Zero();

  // Compute times + delays (microseconds) -------------------------------------
  double controller_compute_time_us  = 0.0;
  double generator_update_time_us    = 0.0;
  double generator_eval_time_us      = 0.0;
  double controller_delay_applied_us = 0.0;
  double generator_delay_applied_us  = 0.0;

  // Scheduler state -----------------------------------------------------------
  int waypoint_index = 0;
  bool hover_active  = false;
  double max_speed   = 0.0;
};

/**
 * @brief Facade MCAP logger producing the same information the CSV backend
 * used to dump as 45 flat columns, now split across ROS 2 topics.
 *
 * Topic layout:
 *   - ``/drone0/self_localization/pose``        : state pose (PoseStamped).
 *   - ``/drone0/self_localization/twist``       : state twist (TwistStamped).
 *   - ``/drone0/sensor_measurements/odom``      : full odometry (Odometry).
 *   - ``/drone0/motion_reference/trajectory``   : generator trajectory sample
 *                                                 with delay applied (PoseStamped).
 *   - ``/drone0/motion_reference/twist``        : per-axis velocity setpoint
 *                                                 from the generator with the
 *                                                 same delay (TwistStamped,
 *                                                 linear only).
 *   - ``/drone0/motion_reference/position``     : active waypoint target,
 *                                                 stepwise, no delay (Vector3).
 *   - ``/drone0/actuator_command/thrust``       : commanded thrust (Thrust).
 *   - ``/drone0/actuator_command/twist``    : commanded body rates (TwistStamped).
 *   - ``/drone0/actuator_command/motor_speeds`` : per-rotor speeds (Float64MultiArray).
 *   - ``/drone0/debug/controller/{compute_output_time,delay_applied}``  (Float64).
 *   - ``/drone0/debug/behaviors/trajectory_generation/{generation_time,eval_time,delay_applied}`` (Float64).
 *   - ``/drone0/debug/mission/waypoint_index`` / ``/drone0/debug/mission/hover_active`` (Int32).
 *   - ``/drone0/debug/mission/max_speed`` (Float64).
 *   - ``/drone0/debug/mission/metadata/{controller_name,generator_name,run_id,language}``
 *     (String, emitted once at t=0).
 */
class UnifiedMcapLogger {
public:
  UnifiedMcapLogger(const std::string& output_path, const RunMetadata& metadata);
  ~UnifiedMcapLogger();

  UnifiedMcapLogger(const UnifiedMcapLogger&)            = delete;
  UnifiedMcapLogger& operator=(const UnifiedMcapLogger&) = delete;

  /// Append a single row to the MCAP (multiple topic writes under the hood).
  void logRow(const LogRow& row);

  /// Close the file (idempotent). Automatically called by the destructor.
  void close();

  /// Canonical MCAP path this logger writes to.
  const std::string& path() const { return file_path_; }

  /// Run metadata bound to this logger.
  const RunMetadata& metadata() const { return metadata_; }

private:
  std::string file_path_;
  RunMetadata metadata_;
  std::unique_ptr<mav_flight_review::MCAPLogger> impl_;
  bool closed_ = false;
};

}  // namespace mpc_examples::framework

#endif  // MPC_EXAMPLES_FRAMEWORK_UNIFIED_MCAP_LOGGER_HPP_
