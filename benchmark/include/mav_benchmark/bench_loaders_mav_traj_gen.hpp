// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file bench_loaders_mav_traj_gen.hpp
 * @brief YAML loader for `config_mav_traj_gen.yaml`, shared between the
 *        mav_trajectory_generation generate / evaluate benchmark TUs.
 *
 * Kept in its own header so the trajectory-MPC benchmark binary (which only
 * needs the gcopter loader) does not have to pull in
 * `mav_trajectory_generation_cpp` headers.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MAV_BENCHMARK_BENCH_LOADERS_MAV_TRAJ_GEN_HPP_
#define MAV_BENCHMARK_BENCH_LOADERS_MAV_TRAJ_GEN_HPP_

#include <yaml-cpp/yaml.h>

#include <stdexcept>
#include <string>

#include "mav_trajectory_generation_cpp/types.hpp"

namespace mav_benchmark {

inline mav_trajectory_generation_cpp::Solver parseTrajSolver(const std::string& value) {
  if (value == "linear") {
    return mav_trajectory_generation_cpp::Solver::Linear;
  }
  if (value == "nonlinear") {
    return mav_trajectory_generation_cpp::Solver::Nonlinear;
  }
  throw std::invalid_argument(
      "config_mav_traj_gen.yaml: optimization.solver must be 'linear' or 'nonlinear' (got '" +
      value + "').");
}

/// Parse `config_mav_traj_gen.yaml`.
inline mav_trajectory_generation_cpp::GeneratorConfig loadMavTrajGenConfig(
    const std::string& path) {
  const YAML::Node root = YAML::LoadFile(path);
  if (!root["optimization"]) {
    throw std::runtime_error(path + " must contain an 'optimization' section.");
  }
  const YAML::Node opt = root["optimization"];

  mav_trajectory_generation_cpp::GeneratorConfig cfg;
  auto& o = cfg.optimization;
  if (opt["derivative_to_optimize"]) {
    o.derivative_to_optimize = opt["derivative_to_optimize"].as<int>();
  }
  if (opt["solver"]) {
    o.solver = parseTrajSolver(opt["solver"].as<std::string>());
  }
  if (opt["a_max"]) {
    o.a_max = opt["a_max"].as<double>();
  }
  if (opt["nl_max_iterations"]) {
    o.nl_max_iterations = opt["nl_max_iterations"].as<int>();
  }
  if (opt["nl_f_rel"]) {
    o.nl_f_rel = opt["nl_f_rel"].as<double>();
  }
  if (opt["nl_x_rel"]) {
    o.nl_x_rel = opt["nl_x_rel"].as<double>();
  }
  if (opt["nl_time_penalty"]) {
    o.nl_time_penalty = opt["nl_time_penalty"].as<double>();
  }
  if (opt["nl_initial_stepsize_rel"]) {
    o.nl_initial_stepsize_rel = opt["nl_initial_stepsize_rel"].as<double>();
  }
  if (opt["nl_inequality_constraint_tolerance"]) {
    o.nl_inequality_constraint_tolerance = opt["nl_inequality_constraint_tolerance"].as<double>();
  }
  return cfg;
}

}  // namespace mav_benchmark

#endif  // MAV_BENCHMARK_BENCH_LOADERS_MAV_TRAJ_GEN_HPP_
