// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file jerk_limited_generator.cpp
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "generators/jerk_limited_generator.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "utils/example_config_utils.hpp"
#include "utils/utils.hpp"

namespace mpc_examples::adapters {

namespace tg = trajectory_generator_jerk_limited;

namespace {

constexpr double kMinHorizontalSpeedForYaw = 0.5;  // [m/s]
constexpr double kMaxYawRateRefRadPerSec   = 1.0;  // [rad/s]

double wrapToPi(double x) {
  while (x > M_PI) x -= 2.0 * M_PI;
  while (x < -M_PI) x += 2.0 * M_PI;
  return x;
}

}  // namespace

JerkLimitedGenerator::JerkLimitedGenerator(const Config& cfg) : cfg_(cfg) {}

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
  return cfg;
}

void JerkLimitedGenerator::initialize(const mav_model::State& initial_state,
                                      const ExampleConfig& example_cfg) {
  if (example_cfg.max_speed <= 0.0) {
    throw std::invalid_argument(
        "JerkLimitedGenerator: ExampleConfig::max_speed must be > 0 (set in config_example.yaml).");
  }
  path_facing_  = example_cfg.path_facing;
  has_prev_t_   = false;
  prev_t_       = 0.0;
  yaw_ref_hold_ = quaternionToEuler(initial_state.getOrientationVector()).z();
  max_speed_    = example_cfg.max_speed;

  hold_pos_        = initial_state.getPositionVector();
  target_wp_       = hold_pos_;
  duration_        = 0.0;
  t_segment_start_ = 0.0;
  has_plan_        = false;

  // Construct the offline solver once; it is reused across segments via generate().
  tg::GeneratorConfig generator_cfg;
  generator_cfg.params.max_speed          = example_cfg.max_speed;
  generator_cfg.params.max_acceleration   = cfg_.max_acceleration;
  generator_cfg.params.max_jerk           = cfg_.max_jerk;
  generator_cfg.params.max_tracking_error = cfg_.max_tracking_error;
  ctrl_ = std::make_unique<tg::TrajectoryGenerator>(generator_cfg);
}

void JerkLimitedGenerator::onWaypointChanged(const Eigen::Vector3d& next_waypoint,
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

  // Two-waypoint hop. The one-shot API resets internally to waypoints[0]
  // with v=0, a=0 — same as the rest of p2p adapters (gcopter, dynamic,
  // mav_traj_gen). The settle margin in the WaypointScheduler absorbs the
  // resulting v0 discontinuity at segment boundaries.
  std::vector<tg::Waypoint> wps = {tg::Waypoint(p0), tg::EndWaypoint(next_waypoint)};
  has_plan_                     = ctrl_->generate(wps, max_speed_);
  if (!has_plan_) {
    duration_ = 0.0;
    return;
  }
  duration_ = ctrl_->duration();
}

void JerkLimitedGenerator::update(double t, const mav_model::State& /*state*/) {
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

framework::ReferenceSample JerkLimitedGenerator::evaluate(double t) const {
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

  // Past the segment end, freeze on the exact target with zero motion.
  // The integrator stops once ‖v‖ ≤ settle_velocity, which can leave the
  // last grabbed sample a few centimetres short of the waypoint; pin the
  // outgoing reference to the requested target so the controller does not
  // see a residual offset during the hover plateau.
  if (t_rel >= duration_) {
    sample.position = target_wp_;
    sample.velocity.setZero();
    sample.acceleration.setZero();
  }
  return sample;
}

}  // namespace mpc_examples::adapters
