// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file csv_logger.hpp
 *
 * Self-contained CSV writer with a fixed 45-column flight telemetry schema.
 * Depends only on the C++ standard library (<fstream>, <filesystem>, STL).
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MAV_FLIGHT_LOGGER_CSV_LOGGER_HPP_
#define MAV_FLIGHT_LOGGER_CSV_LOGGER_HPP_

#include <fstream>
#include <string>

#include "mav_flight_logger/log_row.hpp"

namespace mav_flight_logger {

/**
 * @brief CSV writer with the fixed 45-column schema defined in log_row.hpp.
 *
 * The file is opened (creating parent directories as needed) in the
 * constructor and closed in the destructor. A four-line comment block at
 * the top records the run metadata ("# controller: ...", "# generator: ...",
 * "# run_id: ...", "# language: ...") for human inspection before the
 * canonical column header line.
 */
class CsvLogger {
public:
  /**
   * @brief Open @p output_path for writing. Throws on IO failure.
   * @param output_path Target CSV path. Parent directory is created if missing.
   * @param metadata    Header metadata (controller/generator names, run id).
   */
  CsvLogger(const std::string& output_path, const RunMetadata& metadata);

  ~CsvLogger();

  CsvLogger(const CsvLogger&)            = delete;
  CsvLogger& operator=(const CsvLogger&) = delete;

  /// Append a single row to the CSV.
  void writeRow(const LogRow& row);

  /// Close the file (idempotent). Automatically called by the destructor.
  void close();

  /// Canonical CSV path this logger writes to.
  const std::string& path() const { return file_path_; }

  /// Run metadata bound to this logger (controller/generator names, run id).
  const RunMetadata& metadata() const { return metadata_; }

private:
  void writeHeader_();

  std::string   file_path_;
  std::ofstream file_;
  RunMetadata   metadata_;
};

}  // namespace mav_flight_logger

#endif  // MAV_FLIGHT_LOGGER_CSV_LOGGER_HPP_
