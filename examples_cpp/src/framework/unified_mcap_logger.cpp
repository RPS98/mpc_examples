// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file unified_mcap_logger.cpp
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "framework/unified_mcap_logger.hpp"

#include <vector>

namespace mpc_examples::framework {

namespace {

// Custom scalar / vector topics (ROS 2 std_msgs).
constexpr const char* kTopicControllerCompute    = "/mpc_examples/controller_compute_time_us";
constexpr const char* kTopicGeneratorUpdate      = "/mpc_examples/generator_update_time_us";
constexpr const char* kTopicGeneratorEval        = "/mpc_examples/generator_eval_time_us";
constexpr const char* kTopicControllerDelay      = "/mpc_examples/controller_delay_applied_us";
constexpr const char* kTopicGeneratorDelay       = "/mpc_examples/generator_delay_applied_us";
constexpr const char* kTopicMaxSpeed             = "/mpc_examples/max_speed";
constexpr const char* kTopicWaypointIndex        = "/mpc_examples/waypoint_index";
constexpr const char* kTopicHoverActive          = "/mpc_examples/hover_active";
constexpr const char* kTopicMotorSpeeds          = "/drone0/actuator_command/motor_speeds";

// Metadata topics (emitted once at t=0).
constexpr const char* kTopicMetaController       = "/mpc_examples/metadata/controller_name";
constexpr const char* kTopicMetaGenerator        = "/mpc_examples/metadata/generator_name";
constexpr const char* kTopicMetaRunId            = "/mpc_examples/metadata/run_id";
constexpr const char* kTopicMetaLanguage         = "/mpc_examples/metadata/language";

Eigen::Vector4d toQuatWxyz(const Eigen::Quaterniond& q) {
  return {q.w(), q.x(), q.y(), q.z()};
}

}  // namespace

UnifiedMcapLogger::UnifiedMcapLogger(const std::string& output_path,
                                     const RunMetadata& metadata)
    : file_path_(output_path), metadata_(metadata) {
  mav_flight_mcap::LoggerConfig cfg;
  cfg.file_path      = output_path;
  // SIMULATION keeps the timestamps the caller passes (model seconds) instead
  // of treating them as absolute POSIX time. Matches the CSV column ``time``.
  cfg.time_mode      = mav_flight_mcap::TimeMode::SIMULATION;

  impl_ = std::make_unique<mav_flight_mcap::MCAPLogger>(cfg);

  // Custom topics must be declared before start().
  impl_->add_float64_topic(kTopicControllerCompute);
  impl_->add_float64_topic(kTopicGeneratorUpdate);
  impl_->add_float64_topic(kTopicGeneratorEval);
  impl_->add_float64_topic(kTopicControllerDelay);
  impl_->add_float64_topic(kTopicGeneratorDelay);
  impl_->add_float64_topic(kTopicMaxSpeed);
  impl_->add_int32_topic(kTopicWaypointIndex);
  impl_->add_int32_topic(kTopicHoverActive);
  impl_->add_float64_multi_array_topic(kTopicMotorSpeeds);
  impl_->add_string_topic(kTopicMetaController);
  impl_->add_string_topic(kTopicMetaGenerator);
  impl_->add_string_topic(kTopicMetaRunId);
  impl_->add_string_topic(kTopicMetaLanguage);

  impl_->start();

  // Emit the metadata block once at t=0 so the MCAP is self-describing.
  impl_->save_string(kTopicMetaController, 0.0, metadata_.controller_name);
  impl_->save_string(kTopicMetaGenerator,  0.0, metadata_.generator_name);
  impl_->save_string(kTopicMetaRunId,      0.0, metadata_.run_id);
  impl_->save_string(kTopicMetaLanguage,   0.0, metadata_.language);
}

UnifiedMcapLogger::~UnifiedMcapLogger() { close(); }

void UnifiedMcapLogger::close() {
  if (impl_ && !closed_) {
    impl_->close();
    closed_ = true;
  }
}

void UnifiedMcapLogger::logRow(const LogRow& row) {
  const double t = row.time;

  // State: pose + twist + full odometry under earth/body frame conventions.
  const Eigen::Vector4d quat = toQuatWxyz(row.orientation);
  impl_->save_state(t, row.position, quat, row.linear_velocity, row.angular_velocity);

  // Reference pose (yaw target + waypoint position).
  impl_->save_pose_reference(t, row.reference_position, toQuatWxyz(row.reference_orientation));

  // Actuation: thrust + body rate command.
  impl_->save_actuation(t, row.thrust_n, row.command_angular_velocity);

  // Motor speeds as Float64MultiArray{dim=[4]}.
  const std::vector<double> motors{
      row.motor_w(0), row.motor_w(1), row.motor_w(2), row.motor_w(3)};
  impl_->save_float64_multi_array(kTopicMotorSpeeds, t, motors, {4u});

  // Compute times and delays (all in microseconds, already).
  impl_->save_float64(kTopicControllerCompute, t, row.controller_compute_time_us);
  impl_->save_float64(kTopicGeneratorUpdate,   t, row.generator_update_time_us);
  impl_->save_float64(kTopicGeneratorEval,     t, row.generator_eval_time_us);
  impl_->save_float64(kTopicControllerDelay,   t, row.controller_delay_applied_us);
  impl_->save_float64(kTopicGeneratorDelay,    t, row.generator_delay_applied_us);

  // Scheduler state.
  impl_->save_int32(kTopicWaypointIndex, t, row.waypoint_index);
  impl_->save_int32(kTopicHoverActive,   t, row.hover_active ? 1 : 0);
  impl_->save_float64(kTopicMaxSpeed,    t, row.max_speed);
}

}  // namespace mpc_examples::framework
