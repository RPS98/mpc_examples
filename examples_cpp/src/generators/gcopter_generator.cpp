// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file gcopter_generator.cpp
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "generators/gcopter_generator.hpp"

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

}  // namespace

GcopterGenerator::GcopterGenerator(const Config& cfg) : cfg_(cfg) {
  if (cfg_.corridor_margin <= 0.0) {
    throw std::invalid_argument("GcopterGenerator: corridor_margin must be > 0.");
  }
}

GcopterGenerator::Config GcopterGenerator::loadConfigFromYaml(const std::string& path) {
  const YAML::Node root = detail::loadYamlRoot(path);
  Config cfg;

  const auto req = [](const YAML::Node& n, const char* key) {
    return detail::readDoubleRequired(n[key], key);
  };

  if (!root["trajectory_generator"] || !root["drone_params"] || !root["drone_limits"] ||
      !root["optimization"]) {
    throw std::runtime_error(
        "GcopterGenerator: YAML must contain trajectory_generator, drone_params, "
        "drone_limits and optimization sections.");
  }

  const YAML::Node tg  = root["trajectory_generator"];
  const YAML::Node dp  = root["drone_params"];
  const YAML::Node dl  = root["drone_limits"];
  const YAML::Node opt = root["optimization"];

  cfg.corridor_margin = req(tg, "corridor_margin");

  cfg.drone_params.mass                = req(dp, "mass");
  cfg.drone_params.gravity             = req(dp, "gravity");
  cfg.drone_params.horizontal_drag     = req(dp, "horizontal_drag");
  cfg.drone_params.vertical_drag       = req(dp, "vertical_drag");
  cfg.drone_params.parasitic_drag      = req(dp, "parasitic_drag");
  cfg.drone_params.speed_smooth_factor = req(dp, "speed_smooth_factor");

  // drone_limits.max_velocity is filled in initialize() from ExampleConfig::max_speed.
  cfg.drone_limits.max_body_rate  = req(dl, "max_body_rate");
  cfg.drone_limits.max_tilt_angle = req(dl, "max_tilt_angle");
  cfg.drone_limits.min_thrust     = req(dl, "min_thrust");
  cfg.drone_limits.max_thrust     = req(dl, "max_thrust");

  cfg.optimization.time_weight      = req(opt, "time_weight");
  cfg.optimization.position_weight  = req(opt, "position_weight");
  cfg.optimization.velocity_weight  = req(opt, "velocity_weight");
  cfg.optimization.body_rate_weight = req(opt, "body_rate_weight");
  cfg.optimization.tilt_weight      = req(opt, "tilt_weight");
  cfg.optimization.thrust_weight    = req(opt, "thrust_weight");
  if (opt["smoothing_eps"]) {
    cfg.optimization.smoothing_eps =
        detail::readDoubleRequired(opt["smoothing_eps"], "smoothing_eps");
  }
  if (opt["integral_resolution"]) {
    cfg.optimization.integral_resolution = opt["integral_resolution"].as<int>();
  }
  if (opt["rel_cost_tol"]) {
    cfg.optimization.rel_cost_tol = detail::readDoubleRequired(opt["rel_cost_tol"], "rel_cost_tol");
  }
  return cfg;
}

void GcopterGenerator::initialize(const mav_model::State& initial_state,
                                  const ExampleConfig& example_cfg) {
  if (example_cfg.max_speed <= 0.0) {
    throw std::invalid_argument(
        "GcopterGenerator: ExampleConfig::max_speed must be > 0 "
        "(set in config_example.yaml).");
  }
  path_facing_  = example_cfg.path_facing;
  has_prev_t_   = false;
  prev_t_       = 0.0;
  yaw_ref_hold_ = quaternionToEuler(initial_state.getOrientationVector()).z();

  cfg_.drone_limits.max_velocity = example_cfg.max_speed;

  hold_pos_        = initial_state.getPositionVector();
  target_wp_       = hold_pos_;
  duration_        = 0.0;
  t_segment_start_ = 0.0;
  has_plan_        = false;

  // Construct the solver once; it is reused across segments via generate().
  gcopter_lib::GeneratorConfig generator_cfg;
  generator_cfg.params                       = cfg_.drone_params;
  generator_cfg.limits                       = cfg_.drone_limits;
  generator_cfg.optimization                 = cfg_.optimization;
  generator_cfg.optimization.corridor_margin = cfg_.corridor_margin;
  ctrl_ = std::make_unique<gcopter_lib::TrajectoryGenerator>(generator_cfg);
}

void GcopterGenerator::onWaypointChanged(const Eigen::Vector3d& next_waypoint,
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

  // Two-waypoint hop. MINCO + L-BFGS converges fine for any non-degenerate
  // segment; the pure-vertical degeneracy is handled inside gcopter_lib via
  // OptimizationConfig::vertical_perturbation, so the adapter just hands
  // (start, end) to the solver and trusts it to converge.
  std::vector<gcopter_lib::Waypoint> wps(2);
  wps[0].position = p0;
  wps[1].position = next_waypoint;
  has_plan_       = ctrl_->generate(wps, cfg_.drone_limits.max_velocity);
  if (!has_plan_) {
    // Fall back to a static setpoint at next_waypoint. The comparison remains
    // meaningful: gcopter failing here is an intrinsic property of batch
    // optimisation on a degenerate 2-point segment.
    duration_ = 0.0;
    return;
  }
  duration_ = ctrl_->duration();
}

void GcopterGenerator::update(double t, const mav_model::State& /*state*/) {
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

framework::ReferenceSample GcopterGenerator::evaluate(double t) const {
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
