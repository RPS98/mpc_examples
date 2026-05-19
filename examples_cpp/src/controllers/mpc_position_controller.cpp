// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file mpc_position_controller.cpp
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "controllers/mpc_position_controller.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "acados_mpc/acados_mpc_yaml.hpp"
#include "controllers/mpc_speed_utils.hpp"
#include "utils/example_config_utils.hpp"
#include "utils/utils.hpp"

namespace mpc_examples::adapters {

namespace {

Eigen::Quaterniond yawToQuaternion(double yaw) { return eulerToQuaternion(0.0, 0.0, yaw); }

/**
 * @brief Build stage references along the straight line from current to goal.
 *
 * Stage k gets p_ref[k] = p0 + min((k+1) * v_ref * dt_h, L) * d_hat, i.e. the
 * solver sees a forward-progressing target that saturates at the goal once the
 * accumulated arc length covers the distance L. Equivalent to the legacy
 * setProgressiveReferences() in examples/mpc_controller/run_example.cpp.
 */
void setProgressiveReferences(acados_mpc::MPCData* mpc_data,
                              const Eigen::Vector3d& current_position,
                              const Eigen::Vector3d& goal_position,
                              const Eigen::Quaterniond& desired_orientation,
                              double v_ref,
                              double dt_horizon,
                              int N) {
  const Eigen::Vector3d delta = goal_position - current_position;
  const double distance       = delta.norm();

  if (distance < 1e-9) {
    mpc_data->p_params.setDesiredPosition(
        {goal_position.x(), goal_position.y(), goal_position.z()});
  } else {
    const Eigen::Vector3d direction = delta / distance;
    for (int k = 0; k <= N; ++k) {
      const double s_k                     = std::min((k + 1) * v_ref * dt_horizon, distance);
      const Eigen::Vector3d stage_position = current_position + s_k * direction;
      mpc_data->p_params.setDesiredPosition(
          {stage_position.x(), stage_position.y(), stage_position.z()}, k);
    }
  }

  mpc_data->p_params.setDesiredOrientation({desired_orientation.w(), desired_orientation.x(),
                                            desired_orientation.y(), desired_orientation.z()});
}

}  // namespace

MpcPositionController::MpcPositionController(const Config& cfg) : cfg_(cfg) {
  if (cfg_.mpc_yaml_path.empty()) {
    throw std::invalid_argument("MpcPositionController: mpc_yaml_path must be provided.");
  }
  if (cfg_.max_vel_percentage <= 0.0 || cfg_.max_vel_percentage > 1.0) {
    throw std::invalid_argument("MpcPositionController: max_vel_percentage must be in (0, 1].");
  }
}

MpcPositionController::Config MpcPositionController::loadConfigFromYaml(const std::string& path) {
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

void MpcPositionController::initialize(const mav_model::State& /*initial_state*/,
                                       const ExampleConfig& example_cfg) {
  if (example_cfg.mpc_dt <= 0.0) {
    throw std::invalid_argument("MpcPositionController: example_cfg.mpc_dt must be > 0.");
  }

  mpc_ = std::make_unique<acados_mpc::MPC>();
  acados_mpc::configureMpcFromYaml(*mpc_, cfg_.mpc_yaml_path);

  // Speed knobs derived from the YAML's `constraints.uh[0]` (= max_speed²),
  // matching the aerostack2 `as2_position_mpc_plugin` convention. The ramp
  // built by setProgressiveReferences() and the solver's runtime soft bound
  // share v_ref by construction, so they cannot drift.
  const double uh_default =
      speed_utils::readUhDefault(*mpc_, "MpcPositionController");
  v_ref_ =
      speed_utils::deriveVRef(uh_default, cfg_.max_vel_percentage, "MpcPositionController");
  speed_utils::updateSpeedConstraint(*mpc_, v_ref_);

  control_period_ = example_cfg.mpc_dt;
  horizon_steps_  = mpc_->getPredictionSteps();
  dt_horizon_     = mpc_->getPredictionTimeStep();
}

framework::ControlCommand MpcPositionController::computeCommand(
    const mav_model::State& state,
    const std::vector<framework::ReferenceSample>& references) {
  if (!mpc_) {
    throw std::runtime_error("MpcPositionController: initialize() must be called before use.");
  }
  if (references.empty()) {
    throw std::invalid_argument("MpcPositionController: references must not be empty.");
  }

  const framework::ReferenceSample& ref = references.front();

  const Eigen::Vector3d position       = state.getPositionVector();
  const Eigen::Quaterniond orientation = state.getOrientationVector();
  const Eigen::Vector3d velocity       = state.getLinearVelocityVector();

  acados_mpc::MPCData* mpc_data = mpc_->getData();
  mpc_data->state.setPosition({position.x(), position.y(), position.z()});
  mpc_data->state.setOrientation(
      {orientation.w(), orientation.x(), orientation.y(), orientation.z()});
  mpc_data->state.setLinearVelocity({velocity.x(), velocity.y(), velocity.z()});

  const Eigen::Quaterniond desired_orientation = yawToQuaternion(ref.yaw);
  setProgressiveReferences(mpc_data, position, ref.position, desired_orientation, v_ref_,
                           dt_horizon_, horizon_steps_);

  const auto t0    = std::chrono::high_resolution_clock::now();
  const int status = mpc_->solve();
  const auto t1    = std::chrono::high_resolution_clock::now();
  last_solve_us_   = std::chrono::duration<double>(t1 - t0).count() * 1e6;

  if (status != 0) {
    throw std::runtime_error("MpcPositionController: solver returned status " +
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
