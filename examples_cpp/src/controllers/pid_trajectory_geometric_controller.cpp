// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file pid_trajectory_geometric_controller.cpp
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "controllers/pid_trajectory_geometric_controller.hpp"

#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "utils/example_config_utils.hpp"

namespace mpc_examples::adapters {

namespace {

pid_controller::PIDParameters<double> parsePidParameters(const YAML::Node& node,
                                                         const std::string& section) {
  pid_controller::PIDParameters<double> params;
  const auto read_vec3 = [&](const char* key) -> Eigen::Vector3d {
    const YAML::Node child = node[key];
    if (!child) {
      throw std::invalid_argument(section + "." + key + " is required.");
    }
    return detail::readVector<3>(child, section + "." + key);
  };

  params.Kp_gains = read_vec3("kp");
  params.Ki_gains = read_vec3("ki");
  params.Kd_gains = read_vec3("kd");

  if (node["antiwindup_cte"]) {
    const double v =
        detail::readDoubleRequired(node["antiwindup_cte"], section + ".antiwindup_cte");
    params.antiwindup_cte = Eigen::Vector3d::Constant(v);
  }
  if (node["alpha"]) {
    const double v = detail::readDoubleRequired(node["alpha"], section + ".alpha");
    params.alpha   = Eigen::Vector3d::Constant(v);
  }
  if (node["a_max"]) {
    const double v = detail::readDoubleRequired(node["a_max"], section + ".a_max");
    if (v > 0.0) {
      params.upper_output_saturation = Eigen::Vector3d::Constant(v);
      params.lower_output_saturation = Eigen::Vector3d::Constant(-v);
    }
  }
  return params;
}

}  // namespace

PidTrajectoryGeometricController::PidTrajectoryGeometricController(const Config& cfg)
    : cfg_(cfg) {
  if (cfg_.v_max <= 0.0) {
    throw std::invalid_argument("PidTrajectoryGeometricController: v_max must be > 0.");
  }
}

PidTrajectoryGeometricController::Config PidTrajectoryGeometricController::loadConfigFromYaml(
    const std::string& path) {
  const YAML::Node root = detail::loadYamlRoot(path);
  Config cfg;

  if (root["v_max"]) {
    cfg.v_max = detail::readDoubleRequired(root["v_max"], "v_max");
  }

  const YAML::Node ctrl = root["controller"];
  if (!ctrl || !ctrl.IsMap()) {
    throw std::invalid_argument("pid_trajectory_geometric config: 'controller' must be a mapping.");
  }
  if (!ctrl["trajectory"]) {
    throw std::invalid_argument(
        "pid_trajectory_geometric config: 'controller.trajectory' is required.");
  }
  cfg.trajectory_pid_params = parsePidParameters(ctrl["trajectory"], "controller.trajectory");

  if (!ctrl["geometric"] || !ctrl["geometric"].IsMap()) {
    throw std::invalid_argument(
        "pid_trajectory_geometric config: 'controller.geometric' must be a mapping.");
  }
  const YAML::Node geo = ctrl["geometric"];
  if (!geo["mass"]) {
    throw std::invalid_argument(
        "pid_trajectory_geometric config: 'controller.geometric.mass' is required.");
  }
  cfg.attitude_params.vehicle_mass =
      detail::readDoubleRequired(geo["mass"], "controller.geometric.mass");
  if (!geo["rotation_kp"]) {
    throw std::invalid_argument(
        "pid_trajectory_geometric config: 'controller.geometric.rotation_kp' is required.");
  }
  cfg.rates_params.kp_rotation =
      detail::readVector<3>(geo["rotation_kp"], "controller.geometric.rotation_kp");

  return cfg;
}

void PidTrajectoryGeometricController::initialize(const mav_model::State& /*initial_state*/,
                                                  const ExampleConfig& example_cfg) {
  control_period_ = example_cfg.pid_dt;
  if (control_period_ <= 0.0) {
    throw std::invalid_argument("PidTrajectoryGeometricController: example_cfg.pid_dt must be > 0.");
  }

  pid_controllers::TrajectoryControllerParameters<double> traj_params;
  traj_params.pid_parameters = cfg_.trajectory_pid_params;
  traj_ctrl_ = std::make_unique<pid_controllers::TrajectoryController<double>>(traj_params);

  geo_ctrl_ = std::make_unique<geometric_controller::GeometricController<double>>(
      cfg_.attitude_params, cfg_.rates_params);
}

framework::ControlCommand PidTrajectoryGeometricController::computeCommand(
    const mav_model::State& state,
    const std::vector<framework::ReferenceSample>& references) {
  if (references.empty()) {
    throw std::invalid_argument("PidTrajectoryGeometricController: references must not be empty.");
  }
  const framework::ReferenceSample& ref = references.front();

  const Eigen::Vector3d position       = state.getPositionVector();
  const Eigen::Vector3d velocity       = state.getLinearVelocityVector();
  const Eigen::Quaterniond orientation = state.getOrientationVector();

  const auto t0 = std::chrono::high_resolution_clock::now();

  const Eigen::Vector3d acc_des = traj_ctrl_->trajectoryToLinearAcceleration(
      position, velocity, ref.position, ref.velocity, Eigen::Vector3d::Zero(), control_period_);

  const auto [thrust, rates] = geo_ctrl_->accelerationToRates(acc_des, ref.yaw, orientation);

  const auto t1  = std::chrono::high_resolution_clock::now();
  last_solve_us_ = std::chrono::duration<double>(t1 - t0).count() * 1e6;

  framework::ControlCommand cmd;
  cmd.thrust_n     = thrust;
  cmd.angular_rate = rates;
  return cmd;
}

}  // namespace mpc_examples::adapters
