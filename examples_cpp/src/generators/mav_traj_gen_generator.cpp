// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file mav_traj_gen_generator.cpp
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "generators/mav_traj_gen_generator.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "utils/example_config_utils.hpp"
#include "utils/utils.hpp"

namespace mpc_examples::adapters {

namespace {

constexpr double kMinHorizontalSpeedForYaw = 0.5;  // [m/s]
constexpr double kMaxYawRateRefRadPerSec   = 1.0;  // [rad/s]

double wrapToPi(double x) {
  while (x > M_PI) {
    x -= 2.0 * M_PI;
  }
  while (x < -M_PI) {
    x += 2.0 * M_PI;
  }
  return x;
}

mav_trajectory_generation_cpp::Solver parseSolver(const std::string& value, const std::string& path) {
  if (value == "linear") {
    return mav_trajectory_generation_cpp::Solver::Linear;
  }
  if (value == "nonlinear") {
    return mav_trajectory_generation_cpp::Solver::Nonlinear;
  }
  throw std::invalid_argument(path + " must be 'linear' or 'nonlinear' (got '" + value + "').");
}

}  // namespace

MavTrajGenGenerator::MavTrajGenGenerator(const Config& cfg) : cfg_(cfg) {}

MavTrajGenGenerator::Config MavTrajGenGenerator::loadConfigFromYaml(const std::string& path) {
  const YAML::Node root = detail::loadYamlRoot(path);
  Config cfg;

  if (!root["optimization"]) {
    throw std::runtime_error("MavTrajGenGenerator: YAML must contain an 'optimization' section.");
  }
  const YAML::Node opt = root["optimization"];

  // derivative_to_optimize: optional integer (defaults to SNAP=4 in OptimizationConfig).
  if (opt["derivative_to_optimize"]) {
    cfg.optimization.derivative_to_optimize = opt["derivative_to_optimize"].as<int>();
  }

  // solver: optional string ('linear' | 'nonlinear').
  if (opt["solver"]) {
    cfg.optimization.solver = parseSolver(opt["solver"].as<std::string>(), "optimization.solver");
  }

  // a_max: optional double, used by both solvers (timing alloc + nonlinear bound).
  cfg.optimization.a_max = detail::readDoubleOptional(opt["a_max"], "optimization.a_max",
                                                       cfg.optimization.a_max);

  // NLopt-only knobs: optional doubles / ints.
  if (opt["nl_max_iterations"]) {
    cfg.optimization.nl_max_iterations = opt["nl_max_iterations"].as<int>();
  }
  cfg.optimization.nl_f_rel = detail::readDoubleOptional(opt["nl_f_rel"], "optimization.nl_f_rel",
                                                          cfg.optimization.nl_f_rel);
  cfg.optimization.nl_x_rel = detail::readDoubleOptional(opt["nl_x_rel"], "optimization.nl_x_rel",
                                                          cfg.optimization.nl_x_rel);
  cfg.optimization.nl_time_penalty = detail::readDoubleOptional(
      opt["nl_time_penalty"], "optimization.nl_time_penalty", cfg.optimization.nl_time_penalty);
  cfg.optimization.nl_initial_stepsize_rel = detail::readDoubleOptional(
      opt["nl_initial_stepsize_rel"], "optimization.nl_initial_stepsize_rel",
      cfg.optimization.nl_initial_stepsize_rel);
  cfg.optimization.nl_inequality_constraint_tolerance = detail::readDoubleOptional(
      opt["nl_inequality_constraint_tolerance"], "optimization.nl_inequality_constraint_tolerance",
      cfg.optimization.nl_inequality_constraint_tolerance);

  return cfg;
}

void MavTrajGenGenerator::initialize(const mav_model::State& initial_state,
                                     const ExampleConfig& example_cfg) {
  if (example_cfg.max_speed <= 0.0) {
    throw std::invalid_argument(
        "MavTrajGenGenerator: ExampleConfig::max_speed must be > 0 (set in config_example.yaml).");
  }
  path_facing_  = example_cfg.path_facing;
  max_speed_    = example_cfg.max_speed;
  has_prev_t_   = false;
  prev_t_       = 0.0;
  yaw_ref_hold_ = quaternionToEuler(initial_state.getOrientationVector()).z();

  hold_pos_        = initial_state.getPositionVector();
  target_wp_       = hold_pos_;
  duration_        = 0.0;
  t_segment_start_ = 0.0;
  has_plan_        = false;

  mav_trajectory_generation_cpp::GeneratorConfig generator_cfg;
  generator_cfg.optimization = cfg_.optimization;
  ctrl_ = std::make_unique<mav_trajectory_generation_cpp::TrajectoryGenerator>(generator_cfg);
}

void MavTrajGenGenerator::onWaypointChanged(const Eigen::Vector3d& next_waypoint,
                                            const mav_model::State& state,
                                            double t_start) {
  const Eigen::Vector3d p0 = state.getPositionVector();
  target_wp_               = next_waypoint;
  hold_pos_                = next_waypoint;
  t_segment_start_         = t_start;

  if ((next_waypoint - p0).norm() < 1e-6) {
    has_plan_ = false;
    duration_ = 0.0;
    return;
  }

  // Insert a midpoint to encourage well-conditioned segment-time allocation.
  // Pin the start waypoint's velocity to the drone's current velocity so the
  // trajectory begins smoothly when the WaypointScheduler fires before the
  // previous segment fully decelerates to zero.
  const Eigen::Vector3d v0 = state.getLinearVelocityVector();
  std::vector<mav_trajectory_generation_cpp::Waypoint> wps;
  wps.reserve(3);
  mav_trajectory_generation_cpp::Waypoint start_wp;
  start_wp.position = p0;
  if (v0.norm() > 1e-3) {
    start_wp.velocity = v0;
  }
  wps.push_back(start_wp);
  wps.emplace_back(0.5 * (p0 + next_waypoint));
  wps.emplace_back(next_waypoint);

  has_plan_ = ctrl_->generate(wps, max_speed_);
  if (!has_plan_) {
    duration_ = 0.0;
    return;
  }
  duration_ = ctrl_->duration();
}

void MavTrajGenGenerator::update(double t, const mav_model::State& /*state*/) {
  const double dt = has_prev_t_ ? std::max(t - prev_t_, 0.0) : 0.0;
  prev_t_         = t;
  has_prev_t_     = true;

  if (!has_plan_) {
    return;
  }
  const double t_local = std::clamp(t - t_segment_start_, 0.0, duration_);

  if (path_facing_) {
    const Eigen::Vector3d vel = ctrl_->velocity(t_local);
    const double horiz_speed  = vel.head<2>().norm();
    if (horiz_speed > kMinHorizontalSpeedForYaw) {
      const double yaw_target = std::atan2(vel.y(), vel.x());
      double yaw_delta        = wrapToPi(yaw_target - yaw_ref_hold_);
      const double step       = kMaxYawRateRefRadPerSec * dt;
      yaw_delta               = std::clamp(yaw_delta, -step, step);
      yaw_ref_hold_           = yaw_ref_hold_ + yaw_delta;
    }
  } else {
    yaw_ref_hold_ = 0.0;
  }
}

framework::ReferenceSample MavTrajGenGenerator::evaluate(double t) const {
  framework::ReferenceSample sample;
  sample.yaw = yaw_ref_hold_;

  if (!has_plan_) {
    sample.position = hold_pos_;
    return sample;
  }
  const double t_rel  = t - t_segment_start_;
  const double t_eval = std::clamp(t_rel, 0.0, duration_);
  sample.position     = ctrl_->position(t_eval);
  sample.velocity     = ctrl_->velocity(t_eval);
  sample.acceleration = ctrl_->acceleration(t_eval);

  // Past the segment end, freeze on the endpoint with zero motion.
  if (t_rel >= duration_) {
    sample.velocity.setZero();
    sample.acceleration.setZero();
  }
  return sample;
}

}  // namespace mpc_examples::adapters
