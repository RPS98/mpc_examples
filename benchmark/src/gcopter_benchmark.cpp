// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file gcopter_benchmark.cpp
 * @brief Google Benchmark of the GCOPTER point-to-point trajectory
 *        optimisation (`gcopter_lib::TrajectoryGenerator::generate()`, L-BFGS).
 *
 * GCOPTER is stateless between calls, so every generate() runs the full
 * polytope-SFC + L-BFGS optimisation. A representative two-waypoint hop of
 * length `kCarrotDistance` is solved; the heading is rotated slightly each
 * iteration to avoid trivial degeneracy at constant problem size. The config
 * is parsed directly from config_gcopter.yaml to stay independent of the
 * mav_simulator-backed adapter layer.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include <benchmark/benchmark.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "gcopter_lib/trajectory_generator.hpp"
#include "gcopter_lib/types.hpp"
#include "mav_benchmark/bench_common.hpp"

namespace {

double req(const YAML::Node& node, const char* key) {
  if (!node[key]) {
    throw std::runtime_error(std::string("config_gcopter.yaml: missing key '") + key + "'.");
  }
  return node[key].as<double>();
}

gcopter_lib::GeneratorConfig loadGcopterConfig(const std::string& path, double max_speed) {
  const YAML::Node root = YAML::LoadFile(path);
  if (!root["trajectory_generator"] || !root["drone_params"] || !root["drone_limits"] ||
      !root["optimization"]) {
    throw std::runtime_error(
        "config_gcopter.yaml must contain trajectory_generator, drone_params, drone_limits "
        "and optimization sections.");
  }
  const YAML::Node tg = root["trajectory_generator"];
  const YAML::Node dp = root["drone_params"];
  const YAML::Node dl = root["drone_limits"];
  const YAML::Node opt = root["optimization"];

  gcopter_lib::GeneratorConfig cfg;
  cfg.params.mass = req(dp, "mass");
  cfg.params.gravity = req(dp, "gravity");
  cfg.params.horizontal_drag = req(dp, "horizontal_drag");
  cfg.params.vertical_drag = req(dp, "vertical_drag");
  cfg.params.parasitic_drag = req(dp, "parasitic_drag");
  cfg.params.speed_smooth_factor = req(dp, "speed_smooth_factor");

  cfg.limits.max_velocity = max_speed;  // single source of truth (scenario max_speed)
  cfg.limits.max_body_rate = req(dl, "max_body_rate");
  cfg.limits.max_tilt_angle = req(dl, "max_tilt_angle");
  cfg.limits.min_thrust = req(dl, "min_thrust");
  cfg.limits.max_thrust = req(dl, "max_thrust");

  cfg.optimization.time_weight = req(opt, "time_weight");
  cfg.optimization.position_weight = req(opt, "position_weight");
  cfg.optimization.velocity_weight = req(opt, "velocity_weight");
  cfg.optimization.body_rate_weight = req(opt, "body_rate_weight");
  cfg.optimization.tilt_weight = req(opt, "tilt_weight");
  cfg.optimization.thrust_weight = req(opt, "thrust_weight");
  if (opt["smoothing_eps"]) {
    cfg.optimization.smoothing_eps = opt["smoothing_eps"].as<double>();
  }
  if (opt["integral_resolution"]) {
    cfg.optimization.integral_resolution = opt["integral_resolution"].as<int>();
  }
  if (opt["rel_cost_tol"]) {
    cfg.optimization.rel_cost_tol = opt["rel_cost_tol"].as<double>();
  }
  cfg.optimization.corridor_margin = req(tg, "corridor_margin");
  return cfg;
}

void BM_GcopterGenerate(benchmark::State& state) {
  const gcopter_lib::GeneratorConfig cfg =
      loadGcopterConfig(mav_benchmark::kGcopterYaml, mav_benchmark::kMaxSpeed);
  gcopter_lib::TrajectoryGenerator generator(cfg);

  long i = 0;
  int failures = 0;
  for (auto _ : state) {
    const double yaw = 0.01 * static_cast<double>(i++);
    std::vector<gcopter_lib::Waypoint> wps(2);
    wps[0].position = Eigen::Vector3d(0.0, 0.0, 1.0);
    wps[1].position = Eigen::Vector3d(mav_benchmark::kCarrotDistance * std::cos(yaw),
                                      mav_benchmark::kCarrotDistance * std::sin(yaw), 1.0);
    const bool ok = generator.generate(wps, mav_benchmark::kMaxSpeed);
    benchmark::DoNotOptimize(ok);
    if (!ok) {
      ++failures;
    } else {
      double d = generator.duration();
      benchmark::DoNotOptimize(d);
    }
  }
  if (failures > 0) {
    state.counters["generate_failures"] = failures;
  }
}

BENCHMARK(BM_GcopterGenerate)->Unit(benchmark::kMicrosecond)->Threads(1)->Repetitions(10);

}  // namespace
