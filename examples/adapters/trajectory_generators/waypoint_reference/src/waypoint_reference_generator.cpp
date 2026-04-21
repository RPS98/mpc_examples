// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file waypoint_reference_generator.cpp
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "adapters/trajectory_generators/waypoint_reference_generator.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "utils/example_config_utils.hpp"
#include "utils/utils.hpp"

namespace mpc_examples::adapters {

namespace {

Eigen::Vector3d clampToDistance(const Eigen::Vector3d& current,
                                const Eigen::Vector3d& target,
                                const double d_max) {
  const Eigen::Vector3d delta = target - current;
  const double dist           = delta.norm();
  if (dist < 1e-9 || dist <= d_max) {
    return target;
  }
  return current + (delta / dist) * d_max;
}

double pathFacingYaw(const Eigen::Vector3d& from,
                     const Eigen::Vector3d& to,
                     const double fallback_yaw,
                     const double reach_threshold) {
  const Eigen::Vector3d diff = to - from;
  if (diff.head<2>().norm() < reach_threshold) {
    return fallback_yaw;
  }
  return std::atan2(diff.y(), diff.x());
}

}  // namespace

WaypointReferenceGenerator::WaypointReferenceGenerator(const Config& cfg) : cfg_(cfg) {
  if (cfg_.d_max <= 0.0) {
    throw std::invalid_argument("WaypointReferenceGenerator: d_max must be > 0.");
  }
  if (cfg_.reach_threshold <= 0.0) {
    throw std::invalid_argument(
        "WaypointReferenceGenerator: reach_threshold must be > 0.");
  }
}

WaypointReferenceGenerator::Config WaypointReferenceGenerator::loadConfigFromYaml(
    const std::string& path) {
  const YAML::Node root = detail::loadYamlRoot(path);
  Config cfg;
  if (root["d_max"]) {
    cfg.d_max = detail::readDoubleRequired(root["d_max"], "d_max");
  }
  if (root["reach_threshold"]) {
    cfg.reach_threshold =
        detail::readDoubleRequired(root["reach_threshold"], "reach_threshold");
  }
  return cfg;
}

void WaypointReferenceGenerator::initialize(const std::vector<Eigen::Vector3d>& waypoints,
                                            const mav_model::State& initial_state,
                                            const ExampleConfig& example_cfg) {
  if (waypoints.empty()) {
    throw std::invalid_argument(
        "WaypointReferenceGenerator: waypoints must contain at least one element.");
  }
  waypoints_     = waypoints;
  wp_index_      = 0U;
  finished_      = false;
  path_facing_   = example_cfg.path_facing;
  last_position_ = initial_state.getPositionVector();
  reference_wp_  = waypoints_.front();
  cached_yaw_    = quaternionToEuler(initial_state.getOrientationVector()).z();
}

void WaypointReferenceGenerator::update(double /*t*/, const mav_model::State& state) {
  // Match the legacy ordering: the current control step must use the waypoint
  // that was active at entry; waypoint advance only takes effect in the next
  // step. The reference and yaw are therefore snapshotted BEFORE advancing
  // wp_index_.
  last_position_         = state.getPositionVector();
  const double drone_yaw = quaternionToEuler(state.getOrientationVector()).z();

  reference_wp_ = waypoints_[wp_index_];
  if (path_facing_) {
    cached_yaw_ = pathFacingYaw(last_position_, reference_wp_, drone_yaw, cfg_.reach_threshold);
  } else {
    cached_yaw_ = 0.0;
  }

  const double error_to_wp = (last_position_ - reference_wp_).norm();
  if (error_to_wp < cfg_.reach_threshold) {
    if (wp_index_ + 1U < waypoints_.size()) {
      ++wp_index_;
    } else {
      finished_ = true;
    }
  }
}

framework::ReferenceSample WaypointReferenceGenerator::evaluate(double /*t*/) const {
  framework::ReferenceSample sample;
  sample.position = clampToDistance(last_position_, reference_wp_, cfg_.d_max);
  sample.yaw      = cached_yaw_;
  return sample;
}

}  // namespace mpc_examples::adapters
