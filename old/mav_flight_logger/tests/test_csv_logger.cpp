// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file test_csv_logger.cpp
 *
 * Self-contained unit tests for mav_flight_logger::CsvLogger.
 */

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "mav_flight_logger/csv_logger.hpp"
#include "mav_flight_logger/log_row.hpp"

namespace {

std::filesystem::path makeTempDir() {
  static int counter = 0;
  const auto dir = std::filesystem::temp_directory_path() /
                   ("mav_flight_logger_test_" + std::to_string(++counter));
  std::filesystem::create_directories(dir);
  return dir;
}

std::vector<std::string> readLines(const std::filesystem::path& p) {
  std::ifstream f(p);
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(f, line)) {
    lines.push_back(line);
  }
  return lines;
}

size_t countCommas(const std::string& s) {
  size_t n = 0;
  for (const char c : s) {
    if (c == ',') ++n;
  }
  return n;
}

}  // namespace

TEST(MavFlightLoggerColumnHeader, Has45Columns) {
  // The column count must match the public constant and contain exactly 44
  // commas (45 fields).
  EXPECT_EQ(mav_flight_logger::kColumnCount, 45);
  EXPECT_EQ(countCommas(mav_flight_logger::kColumnHeader), 44u);
}

TEST(MavFlightLoggerCsvLogger, WritesHeaderAndSingleRow) {
  const auto tmp = makeTempDir();
  const auto out = tmp / "run.csv";

  mav_flight_logger::RunMetadata md{"pid", "waypoints", "20260422_120000", "cpp"};
  {
    mav_flight_logger::CsvLogger logger(out.string(), md);
    mav_flight_logger::LogRow row;
    row.time            = 0.01;
    row.position        = {1.0, 2.0, 3.0};
    row.thrust_n        = 9.81;
    row.waypoint_index  = 2;
    row.hover_active    = true;
    row.max_speed       = 3.0;
    logger.writeRow(row);
  }  // destructor closes the file

  ASSERT_TRUE(std::filesystem::exists(out));
  const auto lines = readLines(out);
  ASSERT_GE(lines.size(), 6u);

  // 4 metadata comments + 1 header line + at least 1 data row.
  EXPECT_EQ(lines[0], "# controller: pid");
  EXPECT_EQ(lines[1], "# generator: waypoints");
  EXPECT_EQ(lines[2], "# run_id: 20260422_120000");
  EXPECT_EQ(lines[3], "# language: cpp");
  EXPECT_EQ(lines[4], mav_flight_logger::kColumnHeader);

  // Data row must have exactly 44 commas (45 fields).
  EXPECT_EQ(countCommas(lines[5]), 44u);
}

TEST(MavFlightLoggerQuaternionToEuler, IdentityReturnsZero) {
  const mav_flight_logger::Quat q_id{1.0, 0.0, 0.0, 0.0};
  const auto euler = mav_flight_logger::quaternionToEuler(q_id);
  EXPECT_NEAR(euler[0], 0.0, 1e-12);
  EXPECT_NEAR(euler[1], 0.0, 1e-12);
  EXPECT_NEAR(euler[2], 0.0, 1e-12);
}

TEST(MavFlightLoggerCsvLogger, CreatesNestedDirectoryIfMissing) {
  const auto tmp = makeTempDir();
  const auto out = tmp / "nested" / "deep" / "run.csv";

  mav_flight_logger::RunMetadata md{"pid", "waypoints", "run", "cpp"};
  { mav_flight_logger::CsvLogger logger(out.string(), md); }
  EXPECT_TRUE(std::filesystem::exists(out));
}
