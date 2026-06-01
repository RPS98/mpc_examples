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
#include <vector>

#include "mav_flight_review/mcap_logger.hpp"
#include "mav_flight_review/trajectory_point.hpp"

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

  /// Position payload of `debug/mission/reference/pose` (PoseStamped). The
  /// caller decides what this carries so that the topic matches the
  /// aerostack2 semantics per mission mode:
  ///   * triangle (stepwise): active waypoint target (== reference_position).
  ///   * moving_path (continuous): the moving-target sample emitted by
  ///     the follow_reference broadcaster (mission_moving_path.py).
  /// Orientation is fixed to identity, matching aerostack2's behaviour.
  Eigen::Vector3d mission_pose_ref_position = Eigen::Vector3d::Zero();

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

  // Controller debug ----------------------------------------------------------
  /// Saturated linear velocity the active controller is tracking — output of
  /// the position-PID first stage for cascaded PIDs, stage-1 predicted state
  /// for MPC adapters. Earth frame, m/s. Only emitted when
  /// `publishes_desired_velocity` is true.
  Eigen::Vector3d desired_velocity = Eigen::Vector3d::Zero();
  bool publishes_desired_velocity  = false;

  // Compute times + delays (microseconds) -------------------------------------
  double controller_compute_time_us       = 0.0;
  /// Pure acados time_tot (μs) reported by `ocp_nlp_get(..., "time_tot", ...)`
  /// right after the solve call. Excludes the C++ pre/post overhead of the
  /// wrapper. Zero for non-MPC controllers.
  double controller_acados_solver_time_us = 0.0;
  double generator_update_time_us         = 0.0;
  double generator_eval_time_us           = 0.0;
  double controller_delay_applied_us      = 0.0;
  double generator_delay_applied_us       = 0.0;

  // Scheduler state -----------------------------------------------------------
  int waypoint_index = 0;
  bool hover_active  = false;
  /// When `false`, suppresses the publication of the mission-side topics
  /// (`pose_reference`, `twist_reference`, `waypoint_index`, `max_speed`)
  /// for this row. Used during the synthetic takeoff and landing phases
  /// so the reviewer's segment detector and `clip_to_pose_ref_window`
  /// limit the analysis exactly to the mission window — the same way
  /// aerostack2's mission scripts only publish those topics inside the
  /// `goto` waypoint loop. State, command and motor topics keep being
  /// logged so the MCAP captures the full flight envelope.
  bool publish_mission_signals = true;
  /// True only during the **mission-active** window: the first waypoint acts
  /// as an implicit takeoff (drone starts at (0, 0, 0)) so we mark it
  /// `experiment_active = false`; the bool flips to `true` once the
  /// scheduler advances past it (`waypoint_index >= 1`) and back to `false`
  /// when the final hover phase begins. Matches the latched topic of the
  /// same name in aerostack2's `mission.py` / `mission_moving_path.py`.
  bool experiment_active = false;
  double max_speed   = 0.0;

  // Per-topic emission gates so the mav MCAP mirrors aerostack2's mission
  // publish pattern: `debug/mission/reference/pose` rate-limited (10 Hz
  // for triangle, broadcaster_rate_hz for moving_path); the three latched
  // topics (`waypoint_index`, `max_speed`, `experiment_active`) emitted
  // only when their value changes; `hover_active` likewise latched.
  // Defaults are false so the caller (waypoints_simulator) decides which
  // mission-side save_* invocations run on each row.
  bool publish_mission_pose_ref           = false;
  bool publish_waypoint_index_change      = false;
  bool publish_max_speed_change           = false;
  bool publish_experiment_active_change   = false;
  bool publish_hover_active_change        = false;

  /// Horizon of trajectory setpoints (position + linear velocity +
  /// acceleration + yaw) the controller is consuming. Mirrors
  /// aerostack2's `motion_reference/trajectory` (`as2_msgs/msg/
  /// TrajectorySetpoints`). Empty by default; the caller (waypoints_simulator)
  /// populates it from the current `refs[]` for trajectory-scope runs.
  std::vector<mav_flight_review::TrajectoryPoint> trajectory_horizon;
  /// Gate for `motion_reference/trajectory`. The outer loop sets this to
  /// true on the first inner sub-step of each cycle, so the horizon
  /// follows the outer-loop cadence (100 Hz) instead of the INDI 500 Hz.
  bool publish_trajectory_horizon = false;
};

/**
 * @brief Facade MCAP logger producing the same information the CSV backend
 * used to dump as 45 flat columns, now split across ROS 2 topics.
 *
 * Topic layout:
 *   - ``/drone0/self_localization/pose``        : state pose (PoseStamped).
 *   - ``/drone0/self_localization/twist``       : state twist (TwistStamped).
 *   - ``/drone0/sensor_measurements/odom``      : full odometry (Odometry).
 *   - ``/drone0/debug/mission/reference/pose``  : generator trajectory sample
 *                                                 with delay applied (PoseStamped).
 *                                                 Aerostack2-native naming so
 *                                                 ``mav_flight_review.flight_frame``
 *                                                 picks it up automatically.
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
 *   - ``/drone0/debug/mission/experiment_active`` (Int32, 0/1) — mirror of the
 *     latched Bool topic published by aerostack2's mission script. Aligned
 *     with the reviewer's ``_TOPIC_EXPERIMENT_ACTIVE`` constant.
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
