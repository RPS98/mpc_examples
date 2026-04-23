// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file log_row.hpp
 *
 * Canonical 45-column schema for MAV flight telemetry CSV logs. The package
 * is STL-only on purpose: no Eigen, no yaml-cpp, no ROS. Callers wrapping
 * higher-level types (Eigen::Vector3d, ros msgs, ...) must convert to the
 * plain std::array fields below when filling a LogRow.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MAV_FLIGHT_LOGGER_LOG_ROW_HPP_
#define MAV_FLIGHT_LOGGER_LOG_ROW_HPP_

#include <array>
#include <string>

namespace mav_flight_logger {

/// 3D vector in the world frame (SI units).
using Vec3 = std::array<double, 3>;

/// Quaternion in [w, x, y, z] order.
using Quat = std::array<double, 4>;

/// Per-rotor values (e.g. angular velocities or PWM duties) for a quadrotor.
using Motor4 = std::array<double, 4>;

/// Total number of columns written per row by CsvLogger.
inline constexpr int kColumnCount = 45;

/// Comma-separated header describing the 45 columns written by CsvLogger,
/// without a trailing newline.
inline constexpr const char* kColumnHeader =
    "time,"
    "x,y,z,qw,qx,qy,qz,roll,pitch,yaw,"
    "vx,vy,vz,wx,wy,wz,"
    "x_ref,y_ref,z_ref,qw_ref,qx_ref,qy_ref,qz_ref,roll_ref,pitch_ref,yaw_ref,"
    "thrust,wx_cmd,wy_cmd,wz_cmd,"
    "motor_w0,motor_w1,motor_w2,motor_w3,"
    "controller_name,generator_name,"
    "controller_compute_time_us,generator_update_time_us,generator_eval_time_us,"
    "controller_delay_applied_us,generator_delay_applied_us,"
    "waypoint_index,hover_active,max_speed";

/**
 * @brief Run-level metadata written as header comments in the CSV.
 *
 * Written once at file creation (e.g. "# controller: pid") so each CSV is
 * self-describing without relying on filename conventions.
 */
struct RunMetadata {
  std::string controller_name;
  std::string generator_name;
  std::string run_id;                 ///< e.g. "20260421_153012"
  std::string language = "cpp";       ///< cpp | py
};

/**
 * @brief Single CSV row. All units SI; angles in rad; positions in world frame.
 *
 * Quaternions are always stored as [w, x, y, z] (matching Eigen::Quaterniond's
 * coefficient order when read with .w(), .x(), .y(), .z()).
 */
struct LogRow {
  double time = 0.0;

  // State (world frame) -------------------------------------------------------
  Vec3 position           = {0.0, 0.0, 0.0};
  Quat orientation        = {1.0, 0.0, 0.0, 0.0};
  Vec3 linear_velocity    = {0.0, 0.0, 0.0};
  Vec3 angular_velocity   = {0.0, 0.0, 0.0};  ///< body rates

  // Reference (world frame) ---------------------------------------------------
  Vec3 reference_position    = {0.0, 0.0, 0.0};
  Quat reference_orientation = {1.0, 0.0, 0.0, 0.0};

  // Actuation -----------------------------------------------------------------
  double thrust_n                = 0.0;
  Vec3   command_angular_velocity = {0.0, 0.0, 0.0};
  Motor4 motor_w                  = {0.0, 0.0, 0.0, 0.0};

  // Compute times + delays (microseconds) -------------------------------------
  double controller_compute_time_us   = 0.0;
  double generator_update_time_us     = 0.0;
  double generator_eval_time_us       = 0.0;
  double controller_delay_applied_us  = 0.0;
  double generator_delay_applied_us   = 0.0;

  // Scheduler state -----------------------------------------------------------
  int    waypoint_index = 0;
  bool   hover_active   = false;
  double max_speed      = 0.0;
};

/**
 * @brief Convert a [w, x, y, z] quaternion to intrinsic ZYX Euler angles.
 * @return [roll, pitch, yaw] in radians.
 */
Vec3 quaternionToEuler(const Quat& q);

}  // namespace mav_flight_logger

#endif  // MAV_FLIGHT_LOGGER_LOG_ROW_HPP_
