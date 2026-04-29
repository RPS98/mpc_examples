// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file pid_position_geometric_controller.cpp
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "controllers/pid_position_geometric_controller.hpp"

#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "utils/example_config_utils.hpp"

namespace mpc_examples::adapters {

namespace {

Eigen::Vector3d saturateVelocity(const Eigen::Vector3d& v, const double v_max) {
  const double speed = v.norm();
  if (speed > v_max) {
    return (v / speed) * v_max;
  }
  return v;
}

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
  if (node["saturation_upper"] && node["saturation_lower"]) {
    params.upper_output_saturation =
        detail::readVector<3>(node["saturation_upper"], section + ".saturation_upper");
    params.lower_output_saturation =
        detail::readVector<3>(node["saturation_lower"], section + ".saturation_lower");
    params.proportional_saturation_flag = true;
  }
  return params;
}

}  // namespace

PidPositionGeometricController::PidPositionGeometricController(const Config& cfg) : cfg_(cfg) {}

PidPositionGeometricController::Config PidPositionGeometricController::loadConfigFromYaml(
    const std::string& path) {
  const YAML::Node root = detail::loadYamlRoot(path);
  Config cfg;

  const YAML::Node ctrl = root["controller"];
  if (!ctrl || !ctrl.IsMap()) {
    throw std::invalid_argument("pid_position_geometric config: 'controller' must be a mapping.");
  }
  if (!ctrl["position"]) {
    throw std::invalid_argument(
        "pid_position_geometric config: 'controller.position' is required.");
  }
  cfg.position_pid_params = parsePidParameters(ctrl["position"], "controller.position");

  if (!ctrl["velocity"]) {
    throw std::invalid_argument(
        "pid_position_geometric config: 'controller.velocity' is required.");
  }
  cfg.velocity_pid_params = parsePidParameters(ctrl["velocity"], "controller.velocity");

  if (!ctrl["geometric"] || !ctrl["geometric"].IsMap()) {
    throw std::invalid_argument(
        "pid_position_geometric config: 'controller.geometric' must be a mapping.");
  }
  const YAML::Node geo = ctrl["geometric"];
  if (!geo["mass"]) {
    throw std::invalid_argument(
        "pid_position_geometric config: 'controller.geometric.mass' is required.");
  }
  cfg.attitude_params.vehicle_mass =
      detail::readDoubleRequired(geo["mass"], "controller.geometric.mass");
  if (!geo["rotation_kp"]) {
    throw std::invalid_argument(
        "pid_position_geometric config: 'controller.geometric.rotation_kp' is required.");
  }
  cfg.rates_params.kp_rotation =
      detail::readVector<3>(geo["rotation_kp"], "controller.geometric.rotation_kp");

  return cfg;
}

void PidPositionGeometricController::initialize(const mav_model::State& /*initial_state*/,
                                                const ExampleConfig& example_cfg) {
  control_period_ = example_cfg.pid_dt;
  if (control_period_ <= 0.0) {
    throw std::invalid_argument("PidPositionGeometricController: example_cfg.pid_dt must be > 0.");
  }
  if (example_cfg.max_speed <= 0.0) {
    throw std::invalid_argument(
        "PidPositionGeometricController: example_cfg.max_speed must be > 0 "
        "(set in config_example.yaml).");
  }
  v_max_ = example_cfg.max_speed;

  pid_controllers::PositionControllerParameters<double> pos_params;
  pos_params.pid_parameters = cfg_.position_pid_params;
  pos_ctrl_ = std::make_unique<pid_controllers::PositionController<double>>(pos_params);

  pid_controllers::VelocityControllerParameters<double> vel_params;
  vel_params.pid_parameters = cfg_.velocity_pid_params;
  vel_ctrl_ = std::make_unique<pid_controllers::VelocityController<double>>(vel_params);

  geo_ctrl_ = std::make_unique<geometric_controller::GeometricController<double>>(
      cfg_.attitude_params, cfg_.rates_params);
}

framework::ControlCommand PidPositionGeometricController::computeCommand(
    const mav_model::State& state,
    const std::vector<framework::ReferenceSample>& references) {
  if (references.empty()) {
    throw std::invalid_argument("PidPositionGeometricController: references must not be empty.");
  }
  const framework::ReferenceSample& ref = references.front();

  const Eigen::Vector3d position       = state.getPositionVector();
  const Eigen::Vector3d velocity       = state.getLinearVelocityVector();
  const Eigen::Quaterniond orientation = state.getOrientationVector();

  const auto t0 = std::chrono::high_resolution_clock::now();

  Eigen::Vector3d vel_des =
      pos_ctrl_->positionToLinearVelocity(position, ref.position, control_period_);
  vel_des       = saturateVelocity(vel_des, v_max_);
  last_vel_des_ = vel_des;

  const Eigen::Vector3d acc_des =
      vel_ctrl_->linearVelocityToLinearAcceleration(velocity, vel_des, control_period_);

  const auto [thrust, rates] = geo_ctrl_->accelerationToRates(acc_des, ref.yaw, orientation);

  const auto t1  = std::chrono::high_resolution_clock::now();
  last_solve_us_ = std::chrono::duration<double>(t1 - t0).count() * 1e6;

  framework::ControlCommand cmd;
  cmd.thrust_n     = thrust;
  cmd.angular_rate = rates;
  return cmd;
}

}  // namespace mpc_examples::adapters
