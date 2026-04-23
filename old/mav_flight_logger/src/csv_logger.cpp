// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file csv_logger.cpp
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "mav_flight_logger/csv_logger.hpp"

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <system_error>

#include "mav_flight_logger/log_row.hpp"

namespace mav_flight_logger {

Vec3 quaternionToEuler(const Quat& q) {
  const double w = q[0];
  const double x = q[1];
  const double y = q[2];
  const double z = q[3];

  const double sinr_cosp = 2.0 * (w * x + y * z);
  const double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
  const double roll      = std::atan2(sinr_cosp, cosr_cosp);

  const double sinp  = 2.0 * (w * y - z * x);
  const double pitch = std::abs(sinp) >= 1.0
                           ? std::copysign(M_PI / 2.0, sinp)
                           : std::asin(sinp);

  const double siny_cosp = 2.0 * (w * z + x * y);
  const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
  const double yaw       = std::atan2(siny_cosp, cosy_cosp);

  return {roll, pitch, yaw};
}

CsvLogger::CsvLogger(const std::string& output_path, const RunMetadata& metadata)
    : file_path_(output_path), metadata_(metadata) {
  const std::filesystem::path path(output_path);
  if (path.has_parent_path()) {
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) {
      throw std::runtime_error("CsvLogger: could not create log directory '" +
                               path.parent_path().string() + "': " + error.message());
    }
  }

  file_.open(file_path_, std::ofstream::out | std::ofstream::trunc);
  if (!file_.is_open()) {
    throw std::runtime_error("CsvLogger: could not open '" + file_path_ + "'.");
  }

  writeHeader_();
}

CsvLogger::~CsvLogger() { close(); }

void CsvLogger::close() {
  if (file_.is_open()) {
    file_.close();
  }
}

void CsvLogger::writeHeader_() {
  file_ << "# controller: " << metadata_.controller_name << "\n"
        << "# generator: "  << metadata_.generator_name  << "\n"
        << "# run_id: "     << metadata_.run_id          << "\n"
        << "# language: "   << metadata_.language        << "\n"
        << kColumnHeader    << "\n";
}

void CsvLogger::writeRow(const LogRow& row) {
  const Vec3 euler     = quaternionToEuler(row.orientation);
  const Vec3 euler_ref = quaternionToEuler(row.reference_orientation);

  file_ << row.time << ',';
  file_ << row.position[0] << ',' << row.position[1] << ',' << row.position[2] << ',';
  file_ << row.orientation[0] << ',' << row.orientation[1] << ','
        << row.orientation[2] << ',' << row.orientation[3] << ',';
  file_ << euler[0] << ',' << euler[1] << ',' << euler[2] << ',';
  file_ << row.linear_velocity[0] << ',' << row.linear_velocity[1] << ','
        << row.linear_velocity[2] << ',';
  file_ << row.angular_velocity[0] << ',' << row.angular_velocity[1] << ','
        << row.angular_velocity[2] << ',';

  file_ << row.reference_position[0] << ',' << row.reference_position[1] << ','
        << row.reference_position[2] << ',';
  file_ << row.reference_orientation[0] << ',' << row.reference_orientation[1] << ','
        << row.reference_orientation[2] << ',' << row.reference_orientation[3] << ',';
  file_ << euler_ref[0] << ',' << euler_ref[1] << ',' << euler_ref[2] << ',';

  file_ << row.thrust_n << ',' << row.command_angular_velocity[0] << ','
        << row.command_angular_velocity[1] << ',' << row.command_angular_velocity[2] << ',';
  file_ << row.motor_w[0] << ',' << row.motor_w[1] << ',' << row.motor_w[2] << ','
        << row.motor_w[3] << ',';

  // Controller/generator names as CSV-safe strings (no commas expected).
  file_ << metadata_.controller_name << ',' << metadata_.generator_name << ',';

  file_ << row.controller_compute_time_us << ',' << row.generator_update_time_us << ','
        << row.generator_eval_time_us << ',';
  file_ << row.controller_delay_applied_us << ',' << row.generator_delay_applied_us << ',';

  file_ << row.waypoint_index << ',' << (row.hover_active ? 1 : 0) << ','
        << row.max_speed << '\n';
}

}  // namespace mav_flight_logger
