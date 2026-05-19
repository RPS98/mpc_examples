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

// Aerostack2-native topic names. Aligned with
// mav_flight_review/pybind/python/mav_flight_review/data_model.py so the
// reviewer surfaces timing / mission extras from the same logger.
constexpr const char* kTopicControllerCompute   = "/drone0/debug/controller/compute_output_time";
constexpr const char* kTopicGeneratorUpdate     = "/drone0/debug/behaviors/trajectory_generation/generation_time";
constexpr const char* kTopicGeneratorEval       = "/drone0/debug/behaviors/trajectory_generation/eval_time";
constexpr const char* kTopicControllerDelay     = "/drone0/debug/controller/delay_applied";
constexpr const char* kTopicGeneratorDelay      = "/drone0/debug/behaviors/trajectory_generation/delay_applied";
constexpr const char* kTopicMaxSpeed            = "/drone0/debug/mission/max_speed";
constexpr const char* kTopicWaypointIndex       = "/drone0/debug/mission/waypoint_index";
constexpr const char* kTopicHoverActive         = "/drone0/debug/mission/hover_active";
// Mirror of the latched std_msgs/Bool topic published by aerostack2's
// mission scripts. Published here as Int32 (0/1) because the MCAP writer
// doesn't expose a Bool API; mav_flight_review's flight_frame ingests
// either schema (it reads only the `data` field) and the metrics-side
// masking does `> 0.5` so the int representation works transparently.
constexpr const char* kTopicExperimentActive    = "/drone0/debug/mission/experiment_active";
constexpr const char* kTopicMotorSpeeds         = "/drone0/actuator_command/motor_speeds";
// Aerostack2-native pose reference topic. Aligned with the reviewer's
// hardcoded constant in mav_flight_review/flight_frame.py
// (``_TOPIC_POSE_REF``) so ``load_flight_frame`` picks up the smooth
// reference automatically and ``clip_to_pose_ref_window`` has data to
// work with. The stepwise active waypoint is published separately
// under ``motion_reference/position`` (Vector3) below.
constexpr const char* kTopicMissionRefPose      = "/drone0/debug/mission/reference/pose";
constexpr const char* kTopicMotionRefPosition   = "/drone0/motion_reference/position";
// Velocity the active controller is tracking (post-saturation). Mirrors
// the aerostack2 plugins' `debug/controller/desired_velocity` topic.
constexpr const char* kTopicDesiredVelocity     = "/drone0/debug/controller/desired_velocity";

// Metadata topics (emitted once at t=0).
constexpr const char* kTopicMetaController = "/drone0/debug/mission/metadata/controller_name";
constexpr const char* kTopicMetaGenerator  = "/drone0/debug/mission/metadata/generator_name";
constexpr const char* kTopicMetaRunId      = "/drone0/debug/mission/metadata/run_id";
constexpr const char* kTopicMetaLanguage   = "/drone0/debug/mission/metadata/language";

Eigen::Vector4d toQuatWxyz(const Eigen::Quaterniond& q) { return {q.w(), q.x(), q.y(), q.z()}; }

}  // namespace

UnifiedMcapLogger::UnifiedMcapLogger(const std::string& output_path, const RunMetadata& metadata)
    : file_path_(output_path), metadata_(metadata) {
  mav_flight_review::LoggerConfig cfg;
  cfg.file_path = output_path;
  // SIMULATION keeps the timestamps the caller passes (model seconds) instead
  // of treating them as absolute POSIX time. Matches the CSV column ``time``.
  cfg.time_mode = mav_flight_review::TimeMode::SIMULATION;
  // Route the built-in pose reference channel to the aerostack2-native
  // mission-reference topic so the reviewer (mav_flight_review's
  // ``flight_frame.py::_TOPIC_POSE_REF``) picks up the controller-visible
  // smooth reference. The stepwise waypoint target is published
  // separately under ``motion_reference/position`` (Vector3) below.
  cfg.pose_reference_topic = kTopicMissionRefPose;

  impl_ = std::make_unique<mav_flight_review::MCAPLogger>(cfg);

  // Custom topics must be declared before start().
  impl_->add_float64_topic(kTopicControllerCompute);
  impl_->add_float64_topic(kTopicGeneratorUpdate);
  impl_->add_float64_topic(kTopicGeneratorEval);
  impl_->add_float64_topic(kTopicControllerDelay);
  impl_->add_float64_topic(kTopicGeneratorDelay);
  impl_->add_float64_topic(kTopicMaxSpeed);
  impl_->add_int32_topic(kTopicWaypointIndex);
  impl_->add_int32_topic(kTopicHoverActive);
  impl_->add_int32_topic(kTopicExperimentActive);
  impl_->add_float64_multi_array_topic(kTopicMotorSpeeds);
  impl_->add_vector3_topic(kTopicMotionRefPosition);
  impl_->add_twist_stamped_topic(kTopicDesiredVelocity);
  impl_->add_string_topic(kTopicMetaController);
  impl_->add_string_topic(kTopicMetaGenerator);
  impl_->add_string_topic(kTopicMetaRunId);
  impl_->add_string_topic(kTopicMetaLanguage);

  impl_->start();

  // Emit the metadata block once at t=0 so the MCAP is self-describing.
  impl_->save_string(kTopicMetaController, 0.0, metadata_.controller_name);
  impl_->save_string(kTopicMetaGenerator, 0.0, metadata_.generator_name);
  impl_->save_string(kTopicMetaRunId, 0.0, metadata_.run_id);
  impl_->save_string(kTopicMetaLanguage, 0.0, metadata_.language);
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

  if (row.publish_mission_pose_ref) {
    // Mirror aerostack2's `mission.py::_publish_reference` (triangle)
    // and `mission_moving_path.py::_broadcast_tick` (continuous): this
    // topic carries the **target the drone is currently asked to reach**
    // — the active waypoint in triangle, the moving TF sample in
    // moving_path. The caller picks the right value through
    // `row.mission_pose_ref_position`; orientation is identity in both
    // backends.
    impl_->save_pose_reference(t, row.mission_pose_ref_position,
                               Eigen::Vector4d(1.0, 0.0, 0.0, 0.0));
  }
  if (row.publish_mission_signals) {
    // Per-axis linear velocity reference from the generator (same delay applied).
    impl_->save_twist_reference(t, row.trajectory_velocity);
    // Position reference (waypoint target, stepwise — identical across cases).
    impl_->save_vector3(kTopicMotionRefPosition, t, row.reference_position);
  }

  // Actuation: thrust + body rate command.
  impl_->save_actuation(t, row.thrust_n, row.command_angular_velocity);

  // Motor speeds as Float64MultiArray{dim=[4]}.
  const std::vector<double> motors{row.motor_w(0), row.motor_w(1), row.motor_w(2), row.motor_w(3)};
  impl_->save_float64_multi_array(kTopicMotorSpeeds, t, motors, {4u});

  // Compute times and delays. The topic carries the value in **seconds**
  // (aerostack2 convention — see
  // ``as2_motion_controller/src/controller_handler.cpp:727`` and the
  // matching ``_TIMING_SECONDS_TO_US`` factor in
  // ``mav_flight_review.flight_frame``). The LogRow fields stay in
  // microseconds for in-process consumers, so we divide by 1e6 right
  // before each ``save_float64`` call.
  constexpr double kUsToSec = 1.0e-6;
  impl_->save_float64(kTopicControllerCompute, t, row.controller_compute_time_us * kUsToSec);
  impl_->save_float64(kTopicGeneratorUpdate, t, row.generator_update_time_us * kUsToSec);
  impl_->save_float64(kTopicGeneratorEval, t, row.generator_eval_time_us * kUsToSec);
  impl_->save_float64(kTopicControllerDelay, t, row.controller_delay_applied_us * kUsToSec);
  impl_->save_float64(kTopicGeneratorDelay, t, row.generator_delay_applied_us * kUsToSec);

  // Scheduler state. The three mission-only topics (`waypoint_index`,
  // `max_speed`, `experiment_active`) and `hover_active` mirror
  // aerostack2's latched semantics: each one is emitted only when its
  // value changes since the last log row. The decision is computed in
  // WaypointsSimulator and surfaced through the per-topic gate flags
  // below so the resulting MCAP carries the same sparse-but-monotonic
  // pattern the aerostack2 mission scripts produce.
  if (row.publish_waypoint_index_change) {
    impl_->save_int32(kTopicWaypointIndex, t, row.waypoint_index);
  }
  if (row.publish_max_speed_change) {
    impl_->save_float64(kTopicMaxSpeed, t, row.max_speed);
  }
  if (row.publish_hover_active_change) {
    impl_->save_int32(kTopicHoverActive, t, row.hover_active ? 1 : 0);
  }
  if (row.publish_experiment_active_change) {
    impl_->save_int32(kTopicExperimentActive, t, row.experiment_active ? 1 : 0);
  }

  if (row.publishes_desired_velocity) {
    impl_->save_twist_stamped(kTopicDesiredVelocity, t, row.desired_velocity);
  }

  if (row.publish_trajectory_horizon && !row.trajectory_horizon.empty()) {
    // Mirror aerostack2's `motion_reference/trajectory` (TrajectorySetpoints).
    // The default trajectory_reference_topic in mav_flight_review's MCAPLogger
    // is already `/drone0/motion_reference/trajectory`, so no override is
    // needed.
    impl_->save_trajectory_reference(t, row.trajectory_horizon);
  }
}

}  // namespace mpc_examples::framework
