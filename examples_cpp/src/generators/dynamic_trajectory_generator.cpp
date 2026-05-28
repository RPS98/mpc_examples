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
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
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

/// Mission waypoint list (targets only) for `global` mode. The backend
/// prepends the current vehicle position internally (updateVehiclePosition),
/// so the start is NOT included here — mirroring aerostack2's
/// dynamic_mav_trajectory_generator plugin. Including it would create a
/// degenerate zero-length first segment that wrecks the global time allocation.
dynamic_traj_generator::DynamicWaypoint::Vector makeWaypointList(
    const std::vector<Eigen::Vector3d>& waypoints) {
  dynamic_traj_generator::DynamicWaypoint::Vector result;
  result.reserve(waypoints.size());
  for (const auto& w : waypoints) {
    dynamic_traj_generator::DynamicWaypoint wp;
    wp.resetWaypoint(w);
    result.emplace_back(std::move(wp));
  }
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
  Config cfg;
  try {
    const YAML::Node node = YAML::LoadFile(path);
    if (node && node["mode"]) {
      cfg.mode = node["mode"].as<std::string>();
    }
  } catch (const std::exception&) {
    // Tolerate a comment-only placeholder: keep the default mode.
  }
  return cfg;
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
  t_segment_start_   = 0.0;
  t_min_             = 0.0;
  t_max_             = 0.0;
  has_plan_          = false;
  segment_completed_ = false;

  // Resolve the planning mode: env override (higher-level launchers select
  // racing without editing repo YAMLs) wins over the config value, default
  // point_to_point.
  std::string mode = cfg_.mode;
  if (const char* env = std::getenv("DYNAMIC_TRAJECTORY_MODE"); env && *env) {
    mode = env;
  }
  global_   = (mode == "global");
  eval_dt_  = example_cfg.mpc_dt > 0.0 ? example_cfg.mpc_dt : example_cfg.controller_dt;

  // Drop any previous instance so the next onWaypointChanged() starts from a
  // clean state (matters when the same adapter is reused across runs). In
  // global mode the instance is built once, here, through all waypoints.
  traj_.reset();

  if (global_) {
    if (example_cfg.waypoints.empty()) {
      throw std::invalid_argument(
          "DynamicTrajectoryGenerator(global): example_cfg.waypoints is empty.");
    }
    // Mirror aerostack2's dynamic_mav_trajectory_generator plugin: seed the
    // vehicle position (backend prepends it as the start) and set ONLY the
    // target waypoints. getMinTime()/getMaxTime() block until the trajectory
    // is generated, turning the async backend synchronous.
    const Eigen::Vector3d p0 = initial_state.getPositionVector();
    traj_ = std::make_unique<dynamic_traj_generator::DynamicTrajectory>();
    traj_->setSpeed(max_speed_);
    traj_->updateVehiclePosition(p0);
    traj_->setWaypoints(makeWaypointList(example_cfg.waypoints));
    const double t_min_backend = traj_->getMinTime();  // blocks → sync
    const double t_max_backend = traj_->getMaxTime();
    // Anchor sim time 0 to the backend's min time: evaluate() maps
    // t_local = t - t_segment_start_ = t + t_min_backend (the backend axis),
    // and t_max_ is the sim-time duration so the "t > t_max_" hold still works.
    t_segment_start_ = -t_min_backend;
    t_min_           = 0.0;
    t_max_           = t_max_backend - t_min_backend;
    has_plan_        = true;
    target_wp_       = example_cfg.waypoints.back();
    hold_pos_        = example_cfg.waypoints.back();
  }

  last_sample_.position     = hold_pos_;
  last_sample_.velocity     = Eigen::Vector3d::Zero();
  last_sample_.acceleration = Eigen::Vector3d::Zero();
  last_sample_.yaw          = yaw_ref_hold_;
}

void DynamicTrajectoryGenerator::onWaypointChanged(const Eigen::Vector3d& next_waypoint,
                                                   const mav_model::State& state,
                                                   double t_start) {
  if (global_) {
    return;  // The single global trajectory already covers every waypoint.
  }
  const Eigen::Vector3d p0 = state.getPositionVector();
  target_wp_               = next_waypoint;
  hold_pos_                = next_waypoint;
  t_segment_start_         = t_start;

  segment_completed_ = false;

  if ((next_waypoint - p0).norm() < 1e-6) {
    has_plan_ = false;
    t_min_    = t_start;
    t_max_    = t_start;
    return;
  }

  // Recreate the underlying instance on every segment so the new trajectory
  // is generated from scratch with a fresh internal time origin (no stitching
  // onto the previous segment, no carry-over of last_global_time_evaluated).
  traj_ = std::make_unique<dynamic_traj_generator::DynamicTrajectory>();
  traj_->setSpeed(max_speed_);
  traj_->updateVehiclePosition(p0);
  traj_->setWaypoints(makeSegmentWaypoints(p0, next_waypoint));

  // With a brand-new instance, getMinTime()/getMaxTime() are local times in
  // [0, T_seg]. Map them into sim time using t_segment_start_ as the origin.
  // Both calls block until the first optimisation run completes.
  t_min_    = t_start + traj_->getMinTime();
  t_max_    = t_start + traj_->getMaxTime();
  has_plan_ = true;
}

void DynamicTrajectoryGenerator::update(double t, const mav_model::State& state) {
  (void)state;  // The current pose is not fed back: each segment is an
                // independent open-loop polynomial (see onWaypointChanged()).

  const double dt = has_prev_t_ ? std::max(t - prev_t_, 0.0) : 0.0;
  prev_t_         = t;
  has_prev_t_     = true;

  if (has_plan_ && !segment_completed_ && t >= t_min_ && t <= t_max_) {
    const double t_local = std::clamp(t - t_segment_start_, 0.0,
                                      std::max(0.0, t_max_ - t_segment_start_
                                                        - (global_ ? eval_dt_ : 0.0)));
    dynamic_traj_generator::References refs;
    if (traj_->evaluateTrajectory(static_cast<float>(t_local), refs) && isFiniteRefs(refs)) {
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
  const double t_local = std::clamp(t - t_segment_start_, 0.0,
                                    std::max(0.0, t_max_ - t_segment_start_
                                                      - (global_ ? eval_dt_ : 0.0)));
  dynamic_traj_generator::References refs;
  if (traj_->evaluateTrajectory(static_cast<float>(t_local), refs) && isFiniteRefs(refs)) {
    sample.position     = refs.position;
    sample.velocity     = refs.velocity;
    sample.acceleration = refs.acceleration;
  }
  return sample;
}

}  // namespace mpc_examples::adapters
