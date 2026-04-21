// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dynamic_trajectory_generator.cpp
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "adapters/trajectory_generators/dynamic_trajectory_generator.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

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

dynamic_traj_generator::DynamicWaypoint::Vector toDynamicWaypoints(
    const std::vector<Eigen::Vector3d>& waypoints) {
  dynamic_traj_generator::DynamicWaypoint::Vector result;
  result.reserve(waypoints.size());
  for (const Eigen::Vector3d& position : waypoints) {
    dynamic_traj_generator::DynamicWaypoint wp;
    wp.resetWaypoint(position);
    result.emplace_back(std::move(wp));
  }
  return result;
}

int nearestWaypointIndex(const std::vector<Eigen::Vector3d>& waypoints,
                         const Eigen::Vector3d& position) {
  int best_index     = 0;
  double best_dist_2 = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < waypoints.size(); ++i) {
    const double d2 = (waypoints[i] - position).squaredNorm();
    if (d2 < best_dist_2) {
      best_dist_2 = d2;
      best_index  = static_cast<int>(i);
    }
  }
  return best_index;
}

}  // namespace

DynamicTrajectoryGenerator::DynamicTrajectoryGenerator(const Config& cfg) : cfg_(cfg) {}

DynamicTrajectoryGenerator::Config DynamicTrajectoryGenerator::loadConfigFromYaml(
    const std::string& path) {
  // The adapter has no tunables; require only that the file exists so the CLI
  // contract (-t <yaml>) stays uniform across generators.
  if (!detail::fileExists(path)) {
    throw std::invalid_argument("Config file not found at " +
                                std::filesystem::absolute(path).string() + ".");
  }
  return Config{};
}

void DynamicTrajectoryGenerator::initialize(const std::vector<Eigen::Vector3d>& waypoints,
                                            const mav_model::State& initial_state,
                                            const ExampleConfig& example_cfg) {
  if (waypoints.empty()) {
    throw std::invalid_argument("DynamicTrajectoryGenerator: waypoints must not be empty.");
  }
  waypoints_    = waypoints;
  wp_index_     = 0;
  path_facing_  = example_cfg.path_facing;
  has_prev_t_   = false;
  prev_t_       = 0.0;
  yaw_ref_hold_ = quaternionToEuler(initial_state.getOrientationVector()).z();

  if (example_cfg.max_speed <= 0.0) {
    throw std::invalid_argument(
        "DynamicTrajectoryGenerator: ExampleConfig::max_speed must be > 0 "
        "(set in config_example.yaml).");
  }

  traj_ = std::make_unique<dynamic_traj_generator::DynamicTrajectory>();
  traj_->updateVehiclePosition(initial_state.getPositionVector());
  traj_->setSpeed(example_cfg.max_speed);
  traj_->setWaypoints(toDynamicWaypoints(waypoints_));

  // Blocks until the first optimisation run finishes.
  t_min_ = traj_->getMinTime();
  t_max_ = traj_->getMaxTime();

  last_sample_.position     = waypoints_.back();
  last_sample_.velocity     = Eigen::Vector3d::Zero();
  last_sample_.acceleration = Eigen::Vector3d::Zero();
  last_sample_.yaw          = yaw_ref_hold_;
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

  // Refresh last_sample_ from the stage-0 evaluation.
  if (t <= t_max_) {
    const double t_eval = std::clamp(t, t_min_, t_max_);
    dynamic_traj_generator::References refs;
    if (traj_->evaluateTrajectory(static_cast<float>(t_eval), refs) && isFiniteRefs(refs)) {
      last_sample_.position     = refs.position;
      last_sample_.velocity     = refs.velocity;
      last_sample_.acceleration = refs.acceleration;
    }
  } else {
    last_sample_.velocity.setZero();
    last_sample_.acceleration.setZero();
  }

  // Path-facing yaw with slew-rate limit.
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

  wp_index_ = nearestWaypointIndex(waypoints_, position);
}

framework::ReferenceSample DynamicTrajectoryGenerator::evaluate(double t) const {
  framework::ReferenceSample sample = last_sample_;
  if (!traj_) {
    return sample;
  }
  if (t > t_max_) {
    sample.velocity.setZero();
    sample.acceleration.setZero();
    return sample;
  }
  const double t_eval = std::clamp(t, t_min_, t_max_);
  dynamic_traj_generator::References refs;
  if (traj_->evaluateTrajectory(static_cast<float>(t_eval), refs) && isFiniteRefs(refs)) {
    sample.position     = refs.position;
    sample.velocity     = refs.velocity;
    sample.acceleration = refs.acceleration;
  }
  sample.yaw = yaw_ref_hold_;
  return sample;
}

}  // namespace mpc_examples::adapters
