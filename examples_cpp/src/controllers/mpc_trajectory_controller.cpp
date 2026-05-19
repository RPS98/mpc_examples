// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file mpc_trajectory_controller.cpp
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "controllers/mpc_trajectory_controller.hpp"

#include <yaml-cpp/yaml.h>

#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "acados_mpc/acados_mpc_yaml.hpp"
#include "controllers/mpc_speed_utils.hpp"
#include "utils/example_config_utils.hpp"

namespace mpc_examples::adapters {

namespace {

std::array<double, 4> yawToQuatArray(double yaw) {
  const double half = 0.5 * yaw;
  return {std::cos(half), 0.0, 0.0, std::sin(half)};
}

}  // namespace

MpcTrajectoryController::MpcTrajectoryController(const Config& cfg) : cfg_(cfg) {
  if (cfg_.mpc_yaml_path.empty()) {
    throw std::invalid_argument("MpcTrajectoryController: mpc_yaml_path must be provided.");
  }
  if (cfg_.max_vel_percentage <= 0.0 || cfg_.max_vel_percentage > 1.0) {
    throw std::invalid_argument(
        "MpcTrajectoryController: max_vel_percentage must be in (0, 1].");
  }
}

MpcTrajectoryController::Config MpcTrajectoryController::loadConfigFromYaml(
    const std::string& path) {
  const YAML::Node root = detail::loadYamlRoot(path);
  Config cfg;
  cfg.mpc_yaml_path = path;

  const YAML::Node mpc_node = root["mpc"];
  if (mpc_node && mpc_node.IsMap() && mpc_node["max_vel_percentage"]) {
    cfg.max_vel_percentage =
        detail::readDoubleRequired(mpc_node["max_vel_percentage"], "mpc.max_vel_percentage");
  }
  return cfg;
}

void MpcTrajectoryController::initialize(const mav_model::State& /*initial_state*/,
                                         const ExampleConfig& example_cfg) {
  if (example_cfg.mpc_dt <= 0.0) {
    throw std::invalid_argument("MpcTrajectoryController: example_cfg.mpc_dt must be > 0.");
  }

  mpc_ = std::make_unique<acados_mpc::MPC>();
  acados_mpc::configureMpcFromYaml(*mpc_, cfg_.mpc_yaml_path);

  // Cap the solver's runtime `uh` so the per-stage references handed in by
  // the generator are guaranteed to satisfy ‖v‖² ≤ (sqrt(uh) * pct)². Same
  // convention as MpcPositionController; matches aerostack2's plugin.
  const double uh_default =
      speed_utils::readUhDefault(*mpc_, "MpcTrajectoryController");
  const double v_ref = speed_utils::deriveVRef(
      uh_default, cfg_.max_vel_percentage, "MpcTrajectoryController");
  speed_utils::updateSpeedConstraint(*mpc_, v_ref);

  control_period_ = example_cfg.mpc_dt;
  horizon_steps_  = mpc_->getPredictionSteps();
  dt_horizon_     = mpc_->getPredictionTimeStep();
}

framework::ControlCommand MpcTrajectoryController::computeCommand(
    const mav_model::State& state,
    const std::vector<framework::ReferenceSample>& references) {
  if (!mpc_) {
    throw std::runtime_error("MpcTrajectoryController: initialize() must be called before use.");
  }
  const int expected = horizon_steps_ + 1;
  if (static_cast<int>(references.size()) != expected) {
    throw std::invalid_argument("MpcTrajectoryController: references size mismatch.");
  }

  const Eigen::Vector3d position       = state.getPositionVector();
  const Eigen::Quaterniond orientation = state.getOrientationVector();
  const Eigen::Vector3d velocity       = state.getLinearVelocityVector();

  acados_mpc::MPCData* mpc_data = mpc_->getData();
  mpc_data->state.setPosition({position.x(), position.y(), position.z()});
  mpc_data->state.setOrientation(
      {orientation.w(), orientation.x(), orientation.y(), orientation.z()});
  mpc_data->state.setLinearVelocity({velocity.x(), velocity.y(), velocity.z()});

  const std::array<double, 4> desired_quat = yawToQuatArray(references.front().yaw);

  for (int k = 0; k <= horizon_steps_; ++k) {
    const framework::ReferenceSample& r = references[static_cast<std::size_t>(k)];
    mpc_data->p_params.setDesiredPosition({r.position.x(), r.position.y(), r.position.z()}, k);
    mpc_data->p_params.setDesiredVelocity({r.velocity.x(), r.velocity.y(), r.velocity.z()}, k);
    mpc_data->p_params.setDesiredAcceleration(
        {r.acceleration.x(), r.acceleration.y(), r.acceleration.z()}, k);
    mpc_data->p_params.setDesiredOrientation(desired_quat, k);
  }

  const auto t0    = std::chrono::high_resolution_clock::now();
  const int status = mpc_->solve();
  const auto t1    = std::chrono::high_resolution_clock::now();
  last_solve_us_   = std::chrono::duration<double>(t1 - t0).count() * 1e6;

  if (status != 0) {
    throw std::runtime_error("MpcTrajectoryController: solver returned status " +
                             std::to_string(status));
  }

  const auto stage1_v   = mpc_->getStage1Velocity();
  last_desired_velocity_ = {stage1_v[0], stage1_v[1], stage1_v[2]};

  framework::ControlCommand cmd;
  cmd.thrust_n     = mpc_data->actuation.getThrust();
  const auto w     = mpc_data->actuation.getAngularVelocity();
  cmd.angular_rate = {w[0], w[1], w[2]};
  return cmd;
}

}  // namespace mpc_examples::adapters
