// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dynamic_trajectory_generator.cpp
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "generators/dynamic_trajectory_generator.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dynamic_trajectory_generator/dynamic_waypoint.hpp"
#include "utils/example_config_utils.hpp"
#include "utils/utils.hpp"

namespace mpc_examples::adapters {

namespace {

constexpr double kMinHorizontalSpeedForYaw = 0.5;  // [m/s]
constexpr double kMaxYawRateRefRadPerSec   = 1.0;  // [rad/s]

double wrapToPi(double x) {
  while (x > M_PI) x -= 2.0 * M_PI;
  while (x < -M_PI) x += 2.0 * M_PI;
  return x;
}

bool isFiniteVec3(const Eigen::Vector3d& v) {
  return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

bool isFiniteRefs(const dynamic_traj_generator::References& r) {
  return isFiniteVec3(r.position) && isFiniteVec3(r.velocity) && isFiniteVec3(r.acceleration);
}

dynamic_traj_generator::DynamicWaypoint::Vector makeSegmentWaypoints(const Eigen::Vector3d& start,
                                                                     const Eigen::Vector3d& end) {
  dynamic_traj_generator::DynamicWaypoint::Vector result;
  result.reserve(2);
  dynamic_traj_generator::DynamicWaypoint a;
  a.resetWaypoint(start);
  dynamic_traj_generator::DynamicWaypoint b;
  b.resetWaypoint(end);
  result.emplace_back(std::move(a));
  result.emplace_back(std::move(b));
  return result;
}

}  // namespace

DynamicTrajectoryGenerator::DynamicTrajectoryGenerator(const Config& cfg) : cfg_(cfg) {}

DynamicTrajectoryGenerator::Config DynamicTrajectoryGenerator::loadConfigFromYaml(
    const std::string& path) {
  if (!detail::fileExists(path)) {
    throw std::invalid_argument("Config file not found at " +
                                std::filesystem::absolute(path).string() + ".");
  }
  return Config{};
}

void DynamicTrajectoryGenerator::initialize(const mav_model::State& initial_state,
                                            const ExampleConfig& example_cfg) {
  if (example_cfg.max_speed <= 0.0) {
    throw std::invalid_argument(
        "DynamicTrajectoryGenerator: ExampleConfig::max_speed must be > 0 "
        "(set in config_example.yaml).");
  }
  path_facing_  = example_cfg.path_facing;
  max_speed_    = example_cfg.max_speed;
  has_prev_t_   = false;
  prev_t_       = 0.0;
  yaw_ref_hold_ = quaternionToEuler(initial_state.getOrientationVector()).z();

  hold_pos_          = initial_state.getPositionVector();
  target_wp_         = hold_pos_;
  t_min_             = 0.0;
  t_max_             = 0.0;
  has_plan_          = false;
  segment_completed_ = false;

  traj_ = std::make_unique<dynamic_traj_generator::DynamicTrajectory>();
  traj_->updateVehiclePosition(hold_pos_);
  traj_->setSpeed(max_speed_);

  last_sample_.position     = hold_pos_;
  last_sample_.velocity     = Eigen::Vector3d::Zero();
  last_sample_.acceleration = Eigen::Vector3d::Zero();
  last_sample_.yaw          = yaw_ref_hold_;
}

void DynamicTrajectoryGenerator::onWaypointChanged(const Eigen::Vector3d& next_waypoint,
                                                   const mav_model::State& state,
                                                   double /*t_start*/) {
  const Eigen::Vector3d p0 = state.getPositionVector();
  target_wp_               = next_waypoint;
  hold_pos_                = next_waypoint;

  segment_completed_ = false;

  if ((next_waypoint - p0).norm() < 1e-6) {
    has_plan_ = false;
    t_min_    = 0.0;
    t_max_    = 0.0;
    return;
  }

  traj_->updateVehiclePosition(p0);
  traj_->setWaypoints(makeSegmentWaypoints(p0, next_waypoint));

  // The library exposes absolute (global) time bounds; blocks until the first
  // optimisation run finishes.
  t_min_    = traj_->getMinTime();
  t_max_    = traj_->getMaxTime();
  has_plan_ = true;
}

void DynamicTrajectoryGenerator::update(double t, const mav_model::State& state) {
  if (!traj_) {
    throw std::runtime_error(
        "DynamicTrajectoryGenerator: initialize() must be called before update().");
  }

  const double dt = has_prev_t_ ? std::max(t - prev_t_, 0.0) : 0.0;
  prev_t_         = t;
  has_prev_t_     = true;

  const Eigen::Vector3d position = state.getPositionVector();
  traj_->updateVehiclePosition(position);

  // Keep internal bounds fresh only while we're still within the planned
  // window. Once we've crossed t_max_, we switch to a static setpoint and
  // ignore further library state — otherwise asynchronous restitching can
  // push the reference velocity/acceleration out of bounds during hover.
  if (has_plan_ && !segment_completed_) {
    t_max_ = traj_->getMaxTime();
  }

  if (has_plan_ && !segment_completed_ && t >= t_min_ && t <= t_max_) {
    const double t_eval = std::clamp(t, t_min_, t_max_);
    dynamic_traj_generator::References refs;
    if (traj_->evaluateTrajectory(static_cast<float>(t_eval), refs) && isFiniteRefs(refs)) {
      last_sample_.position     = refs.position;
      last_sample_.velocity     = refs.velocity;
      last_sample_.acceleration = refs.acceleration;
    }
  } else if (has_plan_ && t > t_max_) {
    segment_completed_    = true;
    last_sample_.position = target_wp_;
    last_sample_.velocity.setZero();
    last_sample_.acceleration.setZero();
  } else if (!has_plan_) {
    last_sample_.position = hold_pos_;
    last_sample_.velocity.setZero();
    last_sample_.acceleration.setZero();
  }

  if (path_facing_) {
    const double horiz_speed = last_sample_.velocity.head<2>().norm();
    if (horiz_speed > kMinHorizontalSpeedForYaw) {
      const double yaw_target = std::atan2(last_sample_.velocity.y(), last_sample_.velocity.x());
      double yaw_delta        = wrapToPi(yaw_target - yaw_ref_hold_);
      const double step       = kMaxYawRateRefRadPerSec * dt;
      yaw_delta               = std::clamp(yaw_delta, -step, step);
      yaw_ref_hold_           = yaw_ref_hold_ + yaw_delta;
    }
  } else {
    yaw_ref_hold_ = 0.0;
  }
  last_sample_.yaw = yaw_ref_hold_;
}

framework::ReferenceSample DynamicTrajectoryGenerator::evaluate(double t) const {
  framework::ReferenceSample sample = last_sample_;
  sample.yaw                        = yaw_ref_hold_;

  if (!traj_ || !has_plan_) {
    sample.position = hold_pos_;
    sample.velocity.setZero();
    sample.acceleration.setZero();
    return sample;
  }

  if (t > t_max_) {
    sample.position = target_wp_;
    sample.velocity.setZero();
    sample.acceleration.setZero();
    return sample;
  }
  if (t < t_min_) {
    return sample;
  }
  const double t_eval = std::clamp(t, t_min_, t_max_);
  dynamic_traj_generator::References refs;
  if (traj_->evaluateTrajectory(static_cast<float>(t_eval), refs) && isFiniteRefs(refs)) {
    sample.position     = refs.position;
    sample.velocity     = refs.velocity;
    sample.acceleration = refs.acceleration;
  }
  return sample;
}

}  // namespace mpc_examples::adapters
