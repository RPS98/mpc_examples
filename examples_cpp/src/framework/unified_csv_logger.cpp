// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file unified_csv_logger.cpp
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

// TODO: remove once MCAP pipeline validated. Superseded by
//       "framework/unified_mcap_logger.cpp". Body disabled via `#if 0`.
#if 0

#  include "framework/unified_csv_logger.hpp"

#  include <array>

namespace mpc_examples::framework {

namespace {

mav_flight_logger::Vec3 toVec3(const Eigen::Vector3d& v) {
  return {v.x(), v.y(), v.z()};
}

mav_flight_logger::Quat toQuat(const Eigen::Quaterniond& q) {
  return {q.w(), q.x(), q.y(), q.z()};
}

mav_flight_logger::Motor4 toMotor4(const Eigen::Matrix<double, 4, 1>& m) {
  return {m(0), m(1), m(2), m(3)};
}

}  // namespace

UnifiedCsvLogger::UnifiedCsvLogger(const std::string& output_path,
                                   const RunMetadata& metadata)
    : impl_(std::make_unique<mav_flight_logger::CsvLogger>(output_path, metadata)) {}

UnifiedCsvLogger::~UnifiedCsvLogger() { close(); }

void UnifiedCsvLogger::close() {
  if (impl_) {
    impl_->close();
  }
}

void UnifiedCsvLogger::logRow(const LogRow& row) {
  mav_flight_logger::LogRow out;
  out.time = row.time;

  out.position          = toVec3(row.position);
  out.orientation       = toQuat(row.orientation);
  out.linear_velocity   = toVec3(row.linear_velocity);
  out.angular_velocity  = toVec3(row.angular_velocity);

  out.reference_position    = toVec3(row.reference_position);
  out.reference_orientation = toQuat(row.reference_orientation);

  out.thrust_n                 = row.thrust_n;
  out.command_angular_velocity = toVec3(row.command_angular_velocity);
  out.motor_w                  = toMotor4(row.motor_w);

  out.controller_compute_time_us  = row.controller_compute_time_us;
  out.generator_update_time_us    = row.generator_update_time_us;
  out.generator_eval_time_us      = row.generator_eval_time_us;
  out.controller_delay_applied_us = row.controller_delay_applied_us;
  out.generator_delay_applied_us  = row.generator_delay_applied_us;

  out.waypoint_index  = row.waypoint_index;
  out.hover_active    = row.hover_active;
  out.max_speed       = row.max_speed;

  impl_->writeRow(out);
}

}  // namespace mpc_examples::framework

#endif  // #if 0 (TODO: remove once MCAP pipeline validated)
