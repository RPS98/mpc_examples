// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file circuit_generator.cpp
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "generators/circuit_generator.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>

#include "utils/example_config_utils.hpp"
#include "utils/mission_loader.hpp"

namespace mpc_examples::adapters {

namespace ml = mpc_examples::utils::mission_loader;

namespace {

constexpr double kZeroSpeedEpsilon = 1.0e-6;

std::string envOrEmpty(const char* name) {
  const char* v = std::getenv(name);
  return (v != nullptr) ? std::string(v) : std::string();
}

/// Resolve the (mission, gates) YAML pair following the precedence:
///   1. env var override
///   2. config YAML value
///   3. package demo defaults
std::pair<std::string, std::string> resolveMissionPaths(const CircuitGenerator::Config& cfg) {
  const std::string env_mission = envOrEmpty("CIRCUIT_MISSION_YAML");
  const std::string env_gates   = envOrEmpty("CIRCUIT_GATES_YAML");
  const auto [demo_mission, demo_gates] = ml::defaultPaths();
  const std::string mission =
      !env_mission.empty() ? env_mission : (!cfg.mission_yaml.empty() ? cfg.mission_yaml : demo_mission);
  const std::string gates =
      !env_gates.empty() ? env_gates : (!cfg.gates_yaml.empty() ? cfg.gates_yaml : demo_gates);
  return {mission, gates};
}

spline::Setpoint makeSetpoint(const std::string& id, const Eigen::Vector3d& pos,
                              const Eigen::Vector3d& vel) {
  spline::Setpoint sp;
  sp.id       = id;
  sp.position = {pos.x(), pos.y(), pos.z()};
  sp.velocity = {vel.x(), vel.y(), vel.z()};
  return sp;
}

double tangentYaw(double tx, double ty) { return std::atan2(ty, tx); }

}  // namespace

CircuitGenerator::CircuitGenerator(const Config& cfg) : cfg_(cfg) {
  if (cfg_.desired_speed <= 0.0) {
    throw std::invalid_argument("CircuitGenerator: desired_speed must be > 0.");
  }
  if (cfg_.n_req_points < 2) {
    throw std::invalid_argument("CircuitGenerator: n_req_points must be >= 2.");
  }
}

CircuitGenerator::Config CircuitGenerator::loadConfigFromYaml(const std::string& path) {
  const YAML::Node root = mpc_examples::detail::loadYamlRoot(path);
  Config cfg;
  if (root["mission_yaml"]) cfg.mission_yaml = root["mission_yaml"].as<std::string>();
  if (root["gates_yaml"])   cfg.gates_yaml   = root["gates_yaml"].as<std::string>();
  if (root["desired_speed"]) {
    cfg.desired_speed = root["desired_speed"].as<double>();
  }
  if (root["origin_offset_m"]) {
    cfg.origin_offset_m = root["origin_offset_m"].as<double>();
  }
  if (root["closing_exit_margin_m"]) {
    cfg.closing_exit_margin_m = root["closing_exit_margin_m"].as<double>();
  }
  if (root["target_segment_length"]) {
    cfg.target_segment_length = root["target_segment_length"].as<double>();
  }
  if (root["samples_per_segment"]) {
    cfg.samples_per_segment = root["samples_per_segment"].as<int>();
  }
  if (root["n_req_points"]) {
    cfg.n_req_points = root["n_req_points"].as<int>();
  }
  return cfg;
}

std::vector<spline::Setpoint> CircuitGenerator::buildSetpoints(
    const Eigen::Vector3d& initial_position,
    const std::string& mission_yaml,
    const std::string& gates_yaml,
    double origin_offset_m,
    double closing_exit_margin_m) {
  const ml::MissionData mission = ml::load(mission_yaml, gates_yaml);
  if (mission.fly.empty()) {
    throw std::runtime_error("CircuitGenerator::buildSetpoints: mission has no fly_waypoints.");
  }

  const Eigen::Vector3d first      = mission.fly.front().position;
  const Eigen::Vector3d gate1_vel  = mission.fly.front().velocity;
  const double n_vel               = gate1_vel.norm();
  // Wrap each branch in Eigen::Vector3d(...) so the ternary has a unified
  // type (the raw Eigen expression templates are not commutable).
  Eigen::Vector3d forward = (n_vel > kZeroSpeedEpsilon)
                                ? Eigen::Vector3d(gate1_vel / n_vel)
                                : Eigen::Vector3d(Eigen::Vector3d::UnitX());

  Eigen::Vector3d origin_pos =
      (origin_offset_m > 0.0) ? Eigen::Vector3d(first - origin_offset_m * forward)
                              : initial_position;
  Eigen::Vector3d direction = first - origin_pos;
  const double n_dir        = direction.norm();
  if (n_dir > kZeroSpeedEpsilon) {
    direction /= n_dir;
  } else {
    direction = Eigen::Vector3d::UnitX();
  }

  std::vector<spline::Setpoint> setpoints;
  setpoints.reserve(mission.fly.size() + 2);
  setpoints.push_back(makeSetpoint("origin", origin_pos, direction));
  for (const auto& wp : mission.fly) {
    setpoints.push_back(makeSetpoint(wp.modifiers, wp.position, wp.velocity));
  }

  // Exit setpoint past the closing waypoint.
  const auto& closing = mission.fly.back();
  Eigen::Vector3d fwd = closing.velocity;
  const double nv     = fwd.norm();
  fwd = (nv > kZeroSpeedEpsilon) ? Eigen::Vector3d(fwd / nv)
                                  : Eigen::Vector3d(Eigen::Vector3d::UnitX());
  const Eigen::Vector3d exit_pos = closing.position + closing_exit_margin_m * fwd;
  setpoints.push_back(makeSetpoint("exit", exit_pos, fwd));
  return setpoints;
}

void CircuitGenerator::initialize(const mav_model::State& initial_state,
                                  const ExampleConfig& example_cfg) {
  const auto [mission_yaml, gates_yaml] = resolveMissionPaths(cfg_);

  // Align the spline origin with the post-takeoff pose (the framework's
  // synthetic takeoff lifts the drone to (initial.xy, takeoff_altitude_m)
  // before the first project waypoint).
  Eigen::Vector3d origin_pose = initial_state.getPositionVector();
  if (example_cfg.takeoff_altitude_m > 0.0) {
    origin_pose.z() = example_cfg.takeoff_altitude_m;
  }

  const auto setpoints = buildSetpoints(origin_pose, mission_yaml, gates_yaml,
                                        cfg_.origin_offset_m, cfg_.closing_exit_margin_m);
  traj_ = std::make_unique<spline::TrajectoryGenerator>(
      setpoints, cfg_.n_req_points, cfg_.target_segment_length, cfg_.samples_per_segment);

  s_eval_   = 0.0;
  t_anchor_ = 0.0;
  s_anchor_ = 0.0;
  const auto [win, _s] = traj_->getTrajectoryWindow(0.0);
  window_ = win;
  s_max_  = window_.s_values.empty() ? 0.0 : window_.s_values.back();
}

void CircuitGenerator::onWaypointChanged(const Eigen::Vector3d& /*next_waypoint*/,
                                         const mav_model::State& /*state*/,
                                         double /*t_start*/) {
  // Nothing to do: the spline covers the full circuit.
}

void CircuitGenerator::update(double t, const mav_model::State& state) {
  if (!traj_) {
    return;
  }
  auto [win, s_eval] = traj_->getTrajectoryWindow(s_eval_);
  window_            = win;

  // Project the drone onto the current window (monotonic).
  const Eigen::Vector3d position = state.getPositionVector();
  double best_s                  = s_eval;
  double min_d                   = std::numeric_limits<double>::infinity();
  const std::size_t n_points     = window_.s_values.size();
  const std::size_t n_path =
      std::min(n_points, window_.path.position.size() / 3);
  for (std::size_t i = 0; i < n_path; ++i) {
    const double dx = window_.path.position[3 * i + 0] - position.x();
    const double dy = window_.path.position[3 * i + 1] - position.y();
    const double dz = window_.path.position[3 * i + 2] - position.z();
    const double d  = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (d < min_d) {
      min_d  = d;
      best_s = window_.s_values[i];
    }
  }
  s_eval_   = std::max(best_s, s_eval);
  t_anchor_ = t;
  s_anchor_ = s_eval_;
  s_max_    = window_.s_values.empty() ? 0.0 : window_.s_values.back();
}

framework::ReferenceSample CircuitGenerator::evaluate(double t) const {
  framework::ReferenceSample sample;
  if (!traj_ || window_.s_values.empty()) {
    return sample;
  }
  const double speed = cfg_.desired_speed;
  double s_query     = s_anchor_ + speed * (t - t_anchor_);
  s_query = std::max(window_.s_values.front(), std::min(s_query, s_max_));
  const spline::arc_length_reparam::EvaluatedPoint ev =
      spline::arc_length_reparam::evaluateArcLengthSpline(s_query, window_);
  if (ev.position.size() < 3 || ev.tangent.size() < 3) {
    return sample;
  }
  sample.position     = Eigen::Vector3d(ev.position[0], ev.position[1], ev.position[2]);
  const Eigen::Vector3d tangent(ev.tangent[0], ev.tangent[1], ev.tangent[2]);
  sample.velocity     = speed * tangent;
  sample.acceleration = Eigen::Vector3d::Zero();  // matches Python (centripetal disabled)
  sample.yaw          = tangentYaw(tangent.x(), tangent.y());
  return sample;
}

}  // namespace mpc_examples::adapters
