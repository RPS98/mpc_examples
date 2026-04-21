// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file gcopter_generator.cpp
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "adapters/trajectory_generators/gcopter_generator.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

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

struct ExpandedPath {
  std::vector<Eigen::Vector3d> positions;
  std::vector<double> margins;
};

// Mirrors the anchor expansion used by gcopter_lib/example/run_example.cpp.
ExpandedPath expandPathWithAnchors(const std::vector<Eigen::Vector3d>& waypoints,
                                   double loose_margin, double tight_margin,
                                   double anchor_radius) {
  ExpandedPath out;
  const std::size_t N = waypoints.size();
  if (N < 3 || tight_margin <= 0.0 || anchor_radius <= 0.0) {
    out.positions = waypoints;
    out.margins.assign(N > 0 ? N - 1 : 0, loose_margin);
    return out;
  }

  out.positions.reserve(3 * (N - 2) + 2);
  out.positions.emplace_back(waypoints.front());

  for (std::size_t i = 1; i + 1 < N; ++i) {
    const Eigen::Vector3d& prev = waypoints[i - 1];
    const Eigen::Vector3d& curr = waypoints[i];
    const Eigen::Vector3d& next = waypoints[i + 1];
    const Eigen::Vector3d in_vec  = curr - prev;
    const Eigen::Vector3d out_vec = next - curr;
    const double in_norm          = in_vec.norm();
    const double out_norm         = out_vec.norm();
    if (in_norm <= 0.0 || out_norm <= 0.0) {
      out.positions.emplace_back(curr);
      continue;
    }
    const double r_in  = std::min(anchor_radius, 0.45 * in_norm);
    const double r_out = std::min(anchor_radius, 0.45 * out_norm);
    out.positions.emplace_back(curr - in_vec * (r_in / in_norm));
    out.positions.emplace_back(curr);
    out.positions.emplace_back(curr + out_vec * (r_out / out_norm));
  }
  out.positions.emplace_back(waypoints.back());

  const std::size_t M = N - 2;
  out.margins.reserve(3 * M + 1);
  out.margins.emplace_back(loose_margin);
  for (std::size_t i = 0; i < M; ++i) {
    out.margins.emplace_back(tight_margin);
    out.margins.emplace_back(tight_margin);
    if (i + 1 < M) {
      out.margins.emplace_back(loose_margin);
    }
  }
  out.margins.emplace_back(loose_margin);
  return out;
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

  const YAML::Node tg   = root["trajectory_generator"];
  const YAML::Node dp   = root["drone_params"];
  const YAML::Node dl   = root["drone_limits"];
  const YAML::Node opt  = root["optimization"];

  cfg.corridor_margin = req(tg, "corridor_margin");
  if (tg["waypoint_margin"]) {
    cfg.waypoint_margin = detail::readDoubleRequired(tg["waypoint_margin"], "waypoint_margin");
  }
  if (tg["waypoint_anchor_radius"]) {
    cfg.waypoint_anchor_radius =
        detail::readDoubleRequired(tg["waypoint_anchor_radius"], "waypoint_anchor_radius");
  }

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

  cfg.optimization.time_weight       = req(opt, "time_weight");
  cfg.optimization.position_weight   = req(opt, "position_weight");
  cfg.optimization.velocity_weight   = req(opt, "velocity_weight");
  cfg.optimization.body_rate_weight  = req(opt, "body_rate_weight");
  cfg.optimization.tilt_weight       = req(opt, "tilt_weight");
  cfg.optimization.thrust_weight     = req(opt, "thrust_weight");
  if (opt["smoothing_eps"]) {
    cfg.optimization.smoothing_eps = detail::readDoubleRequired(opt["smoothing_eps"],
                                                                "smoothing_eps");
  }
  if (opt["integral_resolution"]) {
    cfg.optimization.integral_resolution = opt["integral_resolution"].as<int>();
  }
  if (opt["rel_cost_tol"]) {
    cfg.optimization.rel_cost_tol = detail::readDoubleRequired(opt["rel_cost_tol"],
                                                               "rel_cost_tol");
  }
  return cfg;
}

void GcopterGenerator::initialize(const std::vector<Eigen::Vector3d>& waypoints,
                                  const mav_model::State& initial_state,
                                  const ExampleConfig& example_cfg) {
  if (waypoints.size() < 2U) {
    throw std::invalid_argument("GcopterGenerator: at least two waypoints are required.");
  }
  if (example_cfg.max_speed <= 0.0) {
    throw std::invalid_argument(
        "GcopterGenerator: ExampleConfig::max_speed must be > 0 "
        "(set in config_example.yaml).");
  }
  waypoints_    = waypoints;
  wp_index_     = 0;
  path_facing_  = example_cfg.path_facing;
  has_prev_t_   = false;
  prev_t_       = 0.0;
  yaw_ref_hold_ = quaternionToEuler(initial_state.getOrientationVector()).z();

  // Pull the velocity bound from the shared simulation config so every example
  // operates under the same speed envelope.
  cfg_.drone_limits.max_velocity = example_cfg.max_speed;

  // Prepend the initial position so the polynomial starts at the drone,
  // matching the mission definition used by the legacy examples.
  std::vector<Eigen::Vector3d> full_path;
  full_path.reserve(waypoints.size() + 1U);
  const Eigen::Vector3d p0 = initial_state.getPositionVector();
  if ((waypoints.front() - p0).norm() > 1e-6) {
    full_path.emplace_back(p0);
  }
  full_path.insert(full_path.end(), waypoints.begin(), waypoints.end());

  const bool anchors_enabled =
      cfg_.waypoint_margin > 0.0 && cfg_.waypoint_anchor_radius > 0.0;
  const ExpandedPath expanded = expandPathWithAnchors(full_path, cfg_.corridor_margin,
                                                      cfg_.waypoint_margin,
                                                      cfg_.waypoint_anchor_radius);

  std::vector<gcopter_lib::Waypoint> wps;
  wps.reserve(expanded.positions.size());
  for (const Eigen::Vector3d& p : expanded.positions) {
    gcopter_lib::Waypoint wp;
    wp.position = p;
    wps.emplace_back(std::move(wp));
  }

  ctrl_ = std::make_unique<gcopter_lib::TrajectoryGenerator>(
      cfg_.drone_params, cfg_.drone_limits, cfg_.optimization);

  const bool ok = anchors_enabled ? ctrl_->generate(wps, expanded.margins)
                                  : ctrl_->generate(wps, cfg_.corridor_margin);
  if (!ok) {
    throw std::runtime_error(
        "GcopterGenerator: L-BFGS optimisation failed (check waypoints, limits, margins).");
  }
  duration_ = ctrl_->duration();
}

void GcopterGenerator::update(double t, const mav_model::State& /*state*/) {
  if (!ctrl_) {
    throw std::runtime_error("GcopterGenerator: initialize() must be called before update().");
  }

  const double dt = has_prev_t_ ? std::max(t - prev_t_, 0.0) : 0.0;
  prev_t_         = t;
  has_prev_t_     = true;

  // Update the "current waypoint" index for logging — pick the next pending
  // waypoint based on the drone's distance to the mission waypoints. This is
  // purely informational; GCOPTER does not re-plan at runtime.
  const double t_eval = std::clamp(t, 0.0, duration_);
  const Eigen::Vector3d pos = ctrl_->position(t_eval);
  for (int i = wp_index_; i < static_cast<int>(waypoints_.size()) - 1; ++i) {
    if ((pos - waypoints_[static_cast<std::size_t>(i)]).norm() < 0.1) {
      wp_index_ = i + 1;
    }
  }

  // Path-facing yaw with slew-rate limit.
  if (path_facing_) {
    const Eigen::Vector3d vel = ctrl_->velocity(t_eval);
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
  if (!ctrl_) {
    return sample;
  }
  const double t_eval = std::clamp(t, 0.0, duration_);
  sample.position     = ctrl_->position(t_eval);
  sample.velocity     = ctrl_->velocity(t_eval);
  sample.acceleration = ctrl_->acceleration(t_eval);
  sample.yaw          = yaw_ref_hold_;
  // Once the polynomial is exhausted, hold the last point with zero motion
  // so the controller sees a stationary setpoint during hover.
  if (t >= duration_) {
    sample.velocity.setZero();
    sample.acceleration.setZero();
  }
  return sample;
}

}  // namespace mpc_examples::adapters
