// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file unified_csv_logger.hpp
 *
 * Thin facade over ``mav_flight_logger::CsvLogger`` that accepts Eigen
 * types directly and mirrors the 45-column schema of the thirdparty logger.
 * Keeping this facade localises the Eigen <-> std::array conversion so the
 * rest of the framework (and the unified runners) can pass Eigen objects
 * unchanged.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_FRAMEWORK_UNIFIED_CSV_LOGGER_HPP_
#define MPC_EXAMPLES_FRAMEWORK_UNIFIED_CSV_LOGGER_HPP_

// TODO: remove once MCAP pipeline validated. Superseded by
//       "framework/unified_mcap_logger.hpp". The body below is kept disabled
//       via `#if 0` so the file stays readable if we need to re-enable the
//       CSV backend during the transition.
#if 0

#  include <Eigen/Dense>

#  include <memory>
#  include <string>

#  include "mav_flight_logger/csv_logger.hpp"

namespace mpc_examples::framework {

/// Run-level metadata, identical to ``mav_flight_logger::RunMetadata``.
using RunMetadata = mav_flight_logger::RunMetadata;

/**
 * @brief Single CSV row exposed to the framework in Eigen-native types.
 *
 * The facade logger converts every field to plain ``std::array`` values
 * before delegating to ``mav_flight_logger::CsvLogger::writeRow``.
 */
struct LogRow {
  double time = 0.0;

  // State (world frame) -------------------------------------------------------
  Eigen::Vector3d position            = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation      = Eigen::Quaterniond::Identity();
  Eigen::Vector3d linear_velocity     = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_velocity    = Eigen::Vector3d::Zero();

  // Reference (world frame) ---------------------------------------------------
  Eigen::Vector3d reference_position           = Eigen::Vector3d::Zero();
  Eigen::Quaterniond reference_orientation     = Eigen::Quaterniond::Identity();

  // Actuation -----------------------------------------------------------------
  double thrust_n                              = 0.0;
  Eigen::Vector3d command_angular_velocity     = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 4, 1> motor_w          = Eigen::Matrix<double, 4, 1>::Zero();

  // Compute times + delays (microseconds) -------------------------------------
  double controller_compute_time_us            = 0.0;
  double generator_update_time_us              = 0.0;
  double generator_eval_time_us                = 0.0;
  double controller_delay_applied_us           = 0.0;
  double generator_delay_applied_us            = 0.0;

  // Scheduler state -----------------------------------------------------------
  int    waypoint_index   = 0;
  bool   hover_active     = false;
  double max_speed        = 0.0;
};

/**
 * @brief Facade CSV logger producing the canonical 45-column schema.
 *
 * Delegates I/O to ``mav_flight_logger::CsvLogger`` and only handles the
 * Eigen <-> std::array conversion at the ``logRow`` boundary.
 */
class UnifiedCsvLogger {
public:
  UnifiedCsvLogger(const std::string& output_path, const RunMetadata& metadata);
  ~UnifiedCsvLogger();

  UnifiedCsvLogger(const UnifiedCsvLogger&)            = delete;
  UnifiedCsvLogger& operator=(const UnifiedCsvLogger&) = delete;

  void logRow(const LogRow& row);

  void close();

  const std::string& path() const { return impl_->path(); }

private:
  std::unique_ptr<mav_flight_logger::CsvLogger> impl_;
};

}  // namespace mpc_examples::framework

#endif  // #if 0 (TODO: remove once MCAP pipeline validated)

#endif  // MPC_EXAMPLES_FRAMEWORK_UNIFIED_CSV_LOGGER_HPP_
