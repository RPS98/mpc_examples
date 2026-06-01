// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file ssa_position_mpc_controller.cpp
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "controllers/ssa_position_mpc_controller.hpp"

#include <yaml-cpp/yaml.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "acados_ssa_mpc/acados_mpc_yaml.hpp"
#include "utils/example_config_utils.hpp"
#include "utils/utils.hpp"

namespace mpc_examples::adapters {

namespace {

Eigen::Quaterniond yawToQuaternion(double yaw) { return eulerToQuaternion(0.0, 0.0, yaw); }

/**
 * @brief Feed the set-point as a constant reference across the horizon (SSA).
 *
 * Steady-state-aware MPC optimizes its own artificial steady-state target
 * (free decision variable, ``artificial_position``). The admissible (speed-
 * limited) approach is produced by the artificial reference instead of a
 * hand-rolled progressive carrot.
 */
void setSetpointReference(acados_ssa_mpc::MPCData* mpc_data,
                          const Eigen::Vector3d& goal_position,
                          const Eigen::Quaterniond& desired_orientation) {
  mpc_data->p_params.setDesiredPosition({goal_position.x(), goal_position.y(), goal_position.z()});
  mpc_data->p_params.setDesiredOrientation({desired_orientation.w(), desired_orientation.x(),
                                            desired_orientation.y(), desired_orientation.z()});
}

// ── Speed-bound helpers (inlined locally to keep this TU isolated from the
//    P-MPC `mpc_speed_utils.hpp`, which is hard-coded to namespace
//    `acados_mpc` and would collide with `acados_ssa_mpc` if included here).

double readUhDefault(acados_ssa_mpc::MPC& mpc, const std::string& who) {
  if constexpr (acados_ssa_mpc::NonlinearConstraintBounds::Nh == 0) {
    (void)mpc;
    (void)who;
    return 0.0;
  } else {
    const auto uh = mpc.getNonlinearConstraintBounds()->getUhArray();
    if (uh[0] <= 0.0) {
      throw std::invalid_argument(
          who +
          ": constraints.uh[0] must be > 0 in the YAML (it encodes max_speed²). "
          "Got uh=" +
          std::to_string(uh[0]) + ".");
    }
    return uh[0];
  }
}

double deriveVRef(double uh_default, double max_vel_percentage, const std::string& who) {
  if (max_vel_percentage <= 0.0 || max_vel_percentage > 1.0) {
    throw std::invalid_argument(who + ": max_vel_percentage must be in (0, 1].");
  }
  return std::sqrt(uh_default) * max_vel_percentage;
}

void updateSpeedConstraint(acados_ssa_mpc::MPC& mpc, double v_ref) {
  constexpr std::size_t kNh = acados_ssa_mpc::NonlinearConstraintBounds::Nh;
  if constexpr (kNh > 0) {
    std::array<double, kNh> uh{};
    uh[0] = v_ref * v_ref;
    mpc.getNonlinearConstraintBounds()->setUh(uh);
    mpc.updateNonlinearConstraintBounds();
  } else {
    (void)mpc;
    (void)v_ref;
  }
}

}  // namespace

SsaPositionMpcController::SsaPositionMpcController(const Config& cfg) : cfg_(cfg) {
  if (cfg_.mpc_yaml_path.empty()) {
    throw std::invalid_argument("SsaPositionMpcController: mpc_yaml_path must be provided.");
  }
  if (cfg_.max_vel_percentage <= 0.0 || cfg_.max_vel_percentage > 1.0) {
    throw std::invalid_argument("SsaPositionMpcController: max_vel_percentage must be in (0, 1].");
  }
}

SsaPositionMpcController::Config SsaPositionMpcController::loadConfigFromYaml(
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

void SsaPositionMpcController::initialize(const mav_model::State& /*initial_state*/,
                                          const ExampleConfig& example_cfg) {
  if (example_cfg.mpc_dt <= 0.0) {
    throw std::invalid_argument("SsaPositionMpcController: example_cfg.mpc_dt must be > 0.");
  }

  mpc_ = std::make_unique<acados_ssa_mpc::MPC>();
  acados_ssa_mpc::configureMpcFromYaml(*mpc_, cfg_.mpc_yaml_path);

  // Speed knob derived from the YAML's `constraints.uh[0]` (= max_speed²).
  // Only caps the solver's runtime soft bound; the trajectory shape is now
  // governed by the SSA artificial reference, not by a manual carrot.
  const double uh_default = readUhDefault(*mpc_, "SsaPositionMpcController");
  v_ref_ = deriveVRef(uh_default, cfg_.max_vel_percentage, "SsaPositionMpcController");
  updateSpeedConstraint(*mpc_, v_ref_);

  control_period_ = example_cfg.mpc_dt;
}

framework::ControlCommand SsaPositionMpcController::computeCommand(
    const mav_model::State& state,
    const std::vector<framework::ReferenceSample>& references) {
  if (!mpc_) {
    throw std::runtime_error("SsaPositionMpcController: initialize() must be called before use.");
  }
  if (references.empty()) {
    throw std::invalid_argument("SsaPositionMpcController: references must not be empty.");
  }

  const framework::ReferenceSample& ref = references.front();

  const Eigen::Vector3d position       = state.getPositionVector();
  const Eigen::Quaterniond orientation = state.getOrientationVector();
  const Eigen::Vector3d velocity       = state.getLinearVelocityVector();

  acados_ssa_mpc::MPCData* mpc_data = mpc_->getData();
  mpc_data->state.setPosition({position.x(), position.y(), position.z()});
  mpc_data->state.setOrientation(
      {orientation.w(), orientation.x(), orientation.y(), orientation.z()});
  mpc_data->state.setLinearVelocity({velocity.x(), velocity.y(), velocity.z()});

  const Eigen::Quaterniond desired_orientation = yawToQuaternion(ref.yaw);
  setSetpointReference(mpc_data, ref.position, desired_orientation);

  const auto t0    = std::chrono::high_resolution_clock::now();
  const int status = mpc_->solve();
  const auto t1    = std::chrono::high_resolution_clock::now();
  last_solve_us_   = std::chrono::duration<double>(t1 - t0).count() * 1e6;

  {
    const auto* p = mpc_->getAcadosSolverPointers();
    double time_tot = 0.0;
    ocp_nlp_get(p->nlp_solver, "time_tot", &time_tot);
    last_acados_solver_us_ = time_tot * 1e6;
  }

  if (status != 0) {
    throw std::runtime_error("SsaPositionMpcController: solver returned status " +
                             std::to_string(status));
  }

  const auto stage1_v    = mpc_->getStage1Velocity();
  last_desired_velocity_ = {stage1_v[0], stage1_v[1], stage1_v[2]};

  framework::ControlCommand cmd;
  cmd.thrust_n     = mpc_data->actuation.getThrust();
  const auto w     = mpc_data->actuation.getAngularVelocity();
  cmd.angular_rate = {w[0], w[1], w[2]};
  return cmd;
}

}  // namespace mpc_examples::adapters
