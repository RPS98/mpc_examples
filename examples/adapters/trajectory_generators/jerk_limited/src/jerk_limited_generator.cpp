// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file jerk_limited_generator.cpp
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "adapters/trajectory_generators/jerk_limited_generator.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "utils/example_config_utils.hpp"
#include "utils/utils.hpp"

namespace mpc_examples::adapters {

namespace {

constexpr double kMinHorizontalSpeedForYaw = 0.5;   // [m/s]
constexpr double kMaxYawRateRefRadPerSec   = 1.0;   // [rad/s]

double wrapToPi(double x) {
  while (x > M_PI) x -= 2.0 * M_PI;
  while (x < -M_PI) x += 2.0 * M_PI;
  return x;
}

}  // namespace

JerkLimitedGenerator::JerkLimitedGenerator(const Config& cfg) : cfg_(cfg) {
  if (cfg_.reach_threshold <= 0.0) {
    throw std::invalid_argument("JerkLimitedGenerator: reach_threshold must be > 0.");
  }
}

JerkLimitedGenerator::Config JerkLimitedGenerator::loadConfigFromYaml(const std::string& path) {
  const YAML::Node root = detail::loadYamlRoot(path);
  Config cfg;
  const auto read_opt = [&](const char* key, double& dst) {
    if (root[key]) {
      dst = detail::readDoubleRequired(root[key], key);
    }
  };
  read_opt("max_acceleration", cfg.max_acceleration);
  read_opt("max_jerk", cfg.max_jerk);
  read_opt("max_tracking_error", cfg.max_tracking_error);
  read_opt("reach_threshold", cfg.reach_threshold);
  return cfg;
}

void JerkLimitedGenerator::initialize(const std::vector<Eigen::Vector3d>& waypoints,
                                      const mav_model::State& initial_state,
                                      const ExampleConfig& example_cfg) {
  if (waypoints.empty()) {
    throw std::invalid_argument("JerkLimitedGenerator: waypoints must not be empty.");
  }
  waypoints_    = waypoints;
  wp_index_     = 0U;
  finished_     = false;
  path_facing_  = example_cfg.path_facing;
  has_prev_t_   = false;
  prev_t_       = 0.0;
  yaw_ref_hold_ = quaternionToEuler(initial_state.getOrientationVector()).z();

  if (example_cfg.max_speed <= 0.0) {
    throw std::invalid_argument(
        "JerkLimitedGenerator: ExampleConfig::max_speed must be > 0 (set in config_example.yaml).");
  }
  trajectory_generator_jerk_limited::TrajectoryParameters params;
  params.max_speed          = example_cfg.max_speed;
  params.max_acceleration   = cfg_.max_acceleration;
  params.max_jerk           = cfg_.max_jerk;
  params.max_tracking_error = cfg_.max_tracking_error;

  ctrl_ = std::make_unique<trajectory_generator_jerk_limited::WaypointTrajectoryController>(params);
  const Eigen::Vector3d p0 = initial_state.getPositionVector();
  ctrl_->reset(p0);

  last_sample_.position     = p0;
  last_sample_.velocity     = Eigen::Vector3d::Zero();
  last_sample_.acceleration = Eigen::Vector3d::Zero();
  last_sample_.yaw          = yaw_ref_hold_;
}

void JerkLimitedGenerator::update(double t, const mav_model::State& state) {
  if (!ctrl_) {
    throw std::runtime_error("JerkLimitedGenerator: initialize() must be called before update().");
  }

  const double dt = has_prev_t_ ? std::max(t - prev_t_, 0.0) : 0.0;
  prev_t_         = t;
  has_prev_t_     = true;

  const Eigen::Vector3d position = state.getPositionVector();
  const Eigen::Vector3d target   = waypoints_[wp_index_];

  // Advance generator only when dt > 0 (the very first update() has dt==0 and
  // simply snapshots the initial setpoint).
  if (dt > 0.0) {
    const auto set = ctrl_->update(dt, position, target);
    last_sample_.position     = set.position;
    last_sample_.velocity     = set.velocity;
    last_sample_.acceleration = set.acceleration;
  }

  // Path-facing yaw with slew-rate limit (same policy as the legacy
  // mpc_trajectory_controller example).
  double yaw_used = yaw_ref_hold_;
  if (path_facing_) {
    const double horiz_speed = last_sample_.velocity.head<2>().norm();
    if (horiz_speed > kMinHorizontalSpeedForYaw) {
      const double yaw_target = std::atan2(last_sample_.velocity.y(), last_sample_.velocity.x());
      double yaw_delta        = wrapToPi(yaw_target - yaw_ref_hold_);
      const double step       = kMaxYawRateRefRadPerSec * (dt > 0.0 ? dt : 0.0);
      yaw_delta               = std::clamp(yaw_delta, -step, step);
      yaw_ref_hold_           = yaw_ref_hold_ + yaw_delta;
      yaw_used                = yaw_ref_hold_;
    }
  } else {
    yaw_used = 0.0;
  }
  last_sample_.yaw = yaw_used;

  // Waypoint advance when the drone enters the acceptance radius.
  const double err = (position - target).norm();
  if (err < cfg_.reach_threshold) {
    if (wp_index_ + 1U < waypoints_.size()) {
      ++wp_index_;
    } else {
      finished_ = true;
    }
  }
}

framework::ReferenceSample JerkLimitedGenerator::evaluate(double /*t*/) const {
  return last_sample_;
}

}  // namespace mpc_examples::adapters
