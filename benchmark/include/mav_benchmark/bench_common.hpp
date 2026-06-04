// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file bench_common.hpp
 * @brief Shared constants and helpers for the Google Benchmark suite.
 *
 * Solver-agnostic: depends only on yaml-cpp and the STL. Each benchmark
 * translation unit includes this for the default config paths and the
 * scenario parameters; the per-solver glue lives in the respective
 * `*_benchmark.cpp` files.
 *
 * Config paths are relative to the current working directory — run the
 * benchmark from the `mav_examples` repository root (see benchmark/README.md).
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MAV_BENCHMARK_BENCH_COMMON_HPP_
#define MAV_BENCHMARK_BENCH_COMMON_HPP_

#include <yaml-cpp/yaml.h>

#include <string>

namespace mav_benchmark {

/// Default controller / generator configuration YAMLs (relative to CWD).
inline constexpr const char* kMpcYaml = "configs/controllers/config_mpc.yaml";
inline constexpr const char* kSsaYaml = "configs/controllers/config_ssa_position_mpc.yaml";
inline constexpr const char* kGcopterYaml = "configs/generators/config_gcopter.yaml";
inline constexpr const char* kTrajYaml = "configs/generators/config_mav_traj_gen.yaml";

/// Scenario parameters shared across the solve / generation benchmarks.
inline constexpr double kMaxSpeed = 1.0;        ///< Cruise speed [m/s].
inline constexpr double kCarrotDistance = 5.0;  ///< Carrot lookahead (MPC) / hop length [m].

/// Reads `mpc.max_vel_percentage` from a controller YAML, or `fallback`.
inline double readMaxVelPercentage(const std::string& yaml_path, double fallback) {
  const YAML::Node root = YAML::LoadFile(yaml_path);
  const YAML::Node mpc = root["mpc"];
  if (mpc && mpc.IsMap() && mpc["max_vel_percentage"]) {
    return mpc["max_vel_percentage"].as<double>();
  }
  return fallback;
}

}  // namespace mav_benchmark

#endif  // MAV_BENCHMARK_BENCH_COMMON_HPP_
