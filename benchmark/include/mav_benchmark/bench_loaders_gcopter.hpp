// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file bench_loaders_gcopter.hpp
 * @brief YAML loader for `config_gcopter.yaml`, shared between the gcopter
 *        generate / evaluate benchmark translation units (and the T-MPC
 *        reference-adaptation benchmark that samples a gcopter polynomial).
 *
 * Kept in its own header — independent of mav_trajectory_generation — so the
 * trajectory-MPC benchmark binary does not need to pull in
 * `mav_trajectory_generation_cpp` headers.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MAV_BENCHMARK_BENCH_LOADERS_GCOPTER_HPP_
#define MAV_BENCHMARK_BENCH_LOADERS_GCOPTER_HPP_

#include <yaml-cpp/yaml.h>

#include <stdexcept>
#include <string>

#include "gcopter_lib/types.hpp"

namespace mav_benchmark {

inline double requireDouble(const YAML::Node& node, const char* key, const char* yaml_file) {
  if (!node[key]) {
    throw std::runtime_error(std::string(yaml_file) + ": missing key '" + key + "'.");
  }
  return node[key].as<double>();
}

/// Parse `config_gcopter.yaml`, plugging the scenario `max_speed` in.
inline gcopter_lib::GeneratorConfig loadGcopterConfig(const std::string& path, double max_speed) {
  const YAML::Node root = YAML::LoadFile(path);
  if (!root["trajectory_generator"] || !root["drone_params"] || !root["drone_limits"] ||
      !root["optimization"]) {
    throw std::runtime_error(
        path + " must contain trajectory_generator, drone_params, drone_limits and optimization "
               "sections.");
  }
  const YAML::Node tg = root["trajectory_generator"];
  const YAML::Node dp = root["drone_params"];
  const YAML::Node dl = root["drone_limits"];
  const YAML::Node opt = root["optimization"];
  const char* yaml = path.c_str();

  gcopter_lib::GeneratorConfig cfg;
  cfg.params.mass = requireDouble(dp, "mass", yaml);
  cfg.params.gravity = requireDouble(dp, "gravity", yaml);
  cfg.params.horizontal_drag = requireDouble(dp, "horizontal_drag", yaml);
  cfg.params.vertical_drag = requireDouble(dp, "vertical_drag", yaml);
  cfg.params.parasitic_drag = requireDouble(dp, "parasitic_drag", yaml);
  cfg.params.speed_smooth_factor = requireDouble(dp, "speed_smooth_factor", yaml);

  cfg.limits.max_velocity = max_speed;  // single source of truth (scenario max_speed)
  cfg.limits.max_body_rate = requireDouble(dl, "max_body_rate", yaml);
  cfg.limits.max_tilt_angle = requireDouble(dl, "max_tilt_angle", yaml);
  cfg.limits.min_thrust = requireDouble(dl, "min_thrust", yaml);
  cfg.limits.max_thrust = requireDouble(dl, "max_thrust", yaml);

  cfg.optimization.time_weight = requireDouble(opt, "time_weight", yaml);
  cfg.optimization.position_weight = requireDouble(opt, "position_weight", yaml);
  cfg.optimization.velocity_weight = requireDouble(opt, "velocity_weight", yaml);
  cfg.optimization.body_rate_weight = requireDouble(opt, "body_rate_weight", yaml);
  cfg.optimization.tilt_weight = requireDouble(opt, "tilt_weight", yaml);
  cfg.optimization.thrust_weight = requireDouble(opt, "thrust_weight", yaml);
  if (opt["smoothing_eps"]) {
    cfg.optimization.smoothing_eps = opt["smoothing_eps"].as<double>();
  }
  if (opt["integral_resolution"]) {
    cfg.optimization.integral_resolution = opt["integral_resolution"].as<int>();
  }
  if (opt["rel_cost_tol"]) {
    cfg.optimization.rel_cost_tol = opt["rel_cost_tol"].as<double>();
  }
  cfg.optimization.corridor_margin = requireDouble(tg, "corridor_margin", yaml);
  return cfg;
}

}  // namespace mav_benchmark

#endif  // MAV_BENCHMARK_BENCH_LOADERS_GCOPTER_HPP_
