// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file waypoint_reference_generator.cpp
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "generators/waypoint_reference_generator.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <stdexcept>

#include "utils/example_config_utils.hpp"
#include "utils/utils.hpp"

namespace mpc_examples::adapters {

namespace {

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
  if (cfg_.reach_threshold <= 0.0) {
    throw std::invalid_argument("WaypointReferenceGenerator: reach_threshold must be > 0.");
  }
}

WaypointReferenceGenerator::Config WaypointReferenceGenerator::loadConfigFromYaml(
    const std::string& path) {
  const YAML::Node root = detail::loadYamlRoot(path);
  Config cfg;
  if (root["reach_threshold"]) {
    cfg.reach_threshold = detail::readDoubleRequired(root["reach_threshold"], "reach_threshold");
  }
  return cfg;
}

void WaypointReferenceGenerator::initialize(const mav_model::State& initial_state,
                                            const ExampleConfig& example_cfg) {
  path_facing_   = example_cfg.path_facing;
  last_position_ = initial_state.getPositionVector();
  target_wp_     = last_position_;
  cached_yaw_    = quaternionToEuler(initial_state.getOrientationVector()).z();
}

void WaypointReferenceGenerator::onWaypointChanged(const Eigen::Vector3d& next_waypoint,
                                                   const mav_model::State& state,
                                                   double /*t_start*/) {
  target_wp_             = next_waypoint;
  last_position_         = state.getPositionVector();
  const double drone_yaw = quaternionToEuler(state.getOrientationVector()).z();
  cached_yaw_            = path_facing_
                               ? pathFacingYaw(last_position_, target_wp_, drone_yaw, cfg_.reach_threshold)
                               : 0.0;
}

void WaypointReferenceGenerator::update(double /*t*/, const mav_model::State& state) {
  last_position_         = state.getPositionVector();
  const double drone_yaw = quaternionToEuler(state.getOrientationVector()).z();
  if (path_facing_) {
    cached_yaw_ = pathFacingYaw(last_position_, target_wp_, drone_yaw, cfg_.reach_threshold);
  }
}

framework::ReferenceSample WaypointReferenceGenerator::evaluate(double /*t*/) const {
  framework::ReferenceSample sample;
  sample.position = target_wp_;
  sample.yaw      = cached_yaw_;
  return sample;
}

}  // namespace mpc_examples::adapters
