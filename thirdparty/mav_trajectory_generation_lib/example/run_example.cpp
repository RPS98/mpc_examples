// Copyright 2025 mav_trajectory_generation_lib contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.

/**
 * @file run_example.cpp
 * @brief End-to-end CLI: YAML -> loop over trajectories -> CSV per trajectory.
 *
 * Usage:
 *   run_example <config_example.yaml> <config_trajectory.yaml>
 *
 * Reads a list of trajectories from the example YAML and plans each with
 * the same `TrajectoryGenerator` instance (generator is reused — not
 * destroyed between trajectories). The second trajectory is chained to the
 * end state of the first by seeding its initial velocity from
 * `evaluate(maxTime()).velocity`. The third demonstrates that `generate()`
 * fails gracefully when given too few waypoints and the loop continues.
 *
 * Each successful trajectory is written to its own CSV with schema
 * `t,x,y,z,vx,vy,vz,ax,ay,az`.
 */

#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "mav_trajectory_generation_cpp/trajectory_generator.hpp"
#include "mav_trajectory_generation_cpp/types.hpp"

namespace {

const std::string LOGS_DIR = "logs";

struct TrajectorySpec {
  std::string label;
  std::string output_csv;
  std::vector<Eigen::Vector3d> waypoints;
};

struct ExampleParams {
  double step_dt   = 0.01;
  double max_speed = 3.0;
  std::vector<TrajectorySpec> trajectories;
};

mav_trajectory_generation_cpp::Solver solverFromString(const std::string& name) {
  if (name == "linear") {
    return mav_trajectory_generation_cpp::Solver::Linear;
  }
  if (name == "nonlinear") {
    return mav_trajectory_generation_cpp::Solver::Nonlinear;
  }
  throw std::invalid_argument("Unknown solver '" + name + "'. Expected 'linear' or 'nonlinear'.");
}

ExampleParams loadExampleYaml(const std::string& path) {
  const YAML::Node doc = YAML::LoadFile(path);
  ExampleParams out;
  if (const YAML::Node sim = doc["simulation"]) {
    if (sim["step_dt"]) out.step_dt = sim["step_dt"].as<double>();
    if (sim["max_speed"]) out.max_speed = sim["max_speed"].as<double>();
  }
  const YAML::Node trajs = doc["trajectories"];
  if (!trajs || !trajs.IsSequence()) {
    throw std::runtime_error("Expected 'trajectories:' sequence at top level of '" + path + "'.");
  }
  for (const auto& entry : trajs) {
    TrajectorySpec spec;
    if (entry["label"]) spec.label = entry["label"].as<std::string>();
    if (entry["output_csv"]) spec.output_csv = entry["output_csv"].as<std::string>();
    if (spec.output_csv.empty()) {
      throw std::runtime_error("Trajectory entry missing 'output_csv' in '" + path + "'.");
    }
    const YAML::Node wps = entry["waypoints"];
    if (!wps || !wps.IsSequence() || wps.size() == 0) {
      throw std::runtime_error("Trajectory '" + spec.label + "' has no waypoints.");
    }
    for (const auto& w : wps) {
      if (w.size() != 3) {
        throw std::runtime_error("Each waypoint must have exactly 3 entries (x, y, z).");
      }
      spec.waypoints.emplace_back(w[0].as<double>(), w[1].as<double>(), w[2].as<double>());
    }
    out.trajectories.push_back(std::move(spec));
  }
  if (out.trajectories.empty()) {
    throw std::runtime_error("No trajectories declared in '" + path + "'.");
  }
  return out;
}

mav_trajectory_generation_cpp::OptimizationConfig loadTrajectoryYaml(const std::string& path) {
  const YAML::Node doc = YAML::LoadFile(path);
  mav_trajectory_generation_cpp::OptimizationConfig cfg;
  const YAML::Node opt = doc["optimization"];
  if (!opt) return cfg;
  if (opt["derivative_to_optimize"])
    cfg.derivative_to_optimize = opt["derivative_to_optimize"].as<int>();
  if (opt["solver"]) cfg.solver = solverFromString(opt["solver"].as<std::string>());
  if (opt["a_max"]) cfg.a_max = opt["a_max"].as<double>();
  if (opt["nl_max_iterations"]) cfg.nl_max_iterations = opt["nl_max_iterations"].as<int>();
  if (opt["nl_f_rel"]) cfg.nl_f_rel = opt["nl_f_rel"].as<double>();
  if (opt["nl_x_rel"]) cfg.nl_x_rel = opt["nl_x_rel"].as<double>();
  if (opt["nl_time_penalty"]) cfg.nl_time_penalty = opt["nl_time_penalty"].as<double>();
  if (opt["nl_initial_stepsize_rel"])
    cfg.nl_initial_stepsize_rel = opt["nl_initial_stepsize_rel"].as<double>();
  if (opt["nl_inequality_constraint_tolerance"]) {
    cfg.nl_inequality_constraint_tolerance = opt["nl_inequality_constraint_tolerance"].as<double>();
  }
  return cfg;
}

// Build the Waypoint vector for a given trajectory, picking the first entry's
// behaviour based on whether a carry-over velocity is available.
std::vector<mav_trajectory_generation_cpp::Waypoint> buildWaypoints(
    const TrajectorySpec& spec,
    const std::optional<Eigen::Vector3d>& initial_velocity) {
  using mav_trajectory_generation_cpp::EndWaypoint;
  using mav_trajectory_generation_cpp::Waypoint;

  std::vector<Waypoint> out;
  out.reserve(spec.waypoints.size());

  for (std::size_t i = 0; i < spec.waypoints.size(); ++i) {
    const bool is_first = (i == 0);
    const bool is_last  = (i + 1 == spec.waypoints.size());

    if (is_first && initial_velocity) {
      // Chained start: keep velocity continuous with the previous trajectory.
      Waypoint wp(spec.waypoints[i]);
      wp.velocity = *initial_velocity;
      out.push_back(wp);
    } else if (is_first || is_last) {
      // Default endpoints: come to rest.
      out.push_back(EndWaypoint(spec.waypoints[i]));
    } else {
      // Intermediate: position-only (optimiser is free in vel/acc).
      out.push_back(Waypoint(spec.waypoints[i]));
    }
  }
  return out;
}

bool writeCsv(const std::string& path,
              mav_trajectory_generation_cpp::TrajectoryGenerator& gen,
              double step_dt) {
  std::ofstream csv(path);
  if (!csv) {
    std::cerr << "  Could not open output file: " << path << "\n";
    return false;
  }
  csv << "t,x,y,z,vx,vy,vz,ax,ay,az\n";
  csv.setf(std::ios::fixed);
  csv.precision(6);

  const double t_end = gen.maxTime();
  for (double t = gen.minTime(); t <= t_end + 1.0e-9; t += step_dt) {
    const auto s = gen.evaluate(t);
    csv << t << "," << s.position.x() << "," << s.position.y() << "," << s.position.z() << ","
        << s.velocity.x() << "," << s.velocity.y() << "," << s.velocity.z() << ","
        << s.acceleration.x() << "," << s.acceleration.y() << "," << s.acceleration.z() << "\n";
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <config_example.yaml> <config_trajectory.yaml>\n";
    return EXIT_FAILURE;
  }
  try {
    // Create logs directory if it doesn't exist
    std::filesystem::create_directories(LOGS_DIR);

    const ExampleParams example = loadExampleYaml(argv[1]);
    const auto cfg              = loadTrajectoryYaml(argv[2]);

    // Single generator reused for every trajectory — not destroyed between
    // calls; generate() resets and re-plans in place.
    mav_trajectory_generation_cpp::TrajectoryGenerator gen(cfg);

    std::optional<Eigen::Vector3d> carry_velocity;  // initially no carry-over
    int successes = 0;

    for (std::size_t i = 0; i < example.trajectories.size(); ++i) {
      const TrajectorySpec& spec = example.trajectories[i];
      const auto waypoints       = buildWaypoints(spec, carry_velocity);

      if (!gen.generate(waypoints, example.max_speed)) {
        std::cerr << "[FAIL] " << spec.label << ": generate() failed ("
                  << spec.waypoints.size() << " waypoint"
                  << (spec.waypoints.size() == 1 ? "" : "s") << ")\n";
        carry_velocity.reset();  // break the chain on failure
        continue;
      }

      const std::string csv_path =
          std::filesystem::path(LOGS_DIR) / std::filesystem::path(spec.output_csv).filename();
      if (!writeCsv(csv_path, gen, example.step_dt)) {
        carry_velocity.reset();
        continue;
      }
      std::cout << "[OK]   " << spec.label << ": wrote " << csv_path
                << " (duration=" << gen.duration() << " s)\n";
      ++successes;

      // Seed the next trajectory's initial velocity with this trajectory's
      // final velocity — used iff the next entry is authored as "chained".
      carry_velocity = gen.evaluate(gen.maxTime()).velocity;
    }

    if (successes == 0) {
      std::cerr << "No trajectories were produced.\n";
      return EXIT_FAILURE;
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
