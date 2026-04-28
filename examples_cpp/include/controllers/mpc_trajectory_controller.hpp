// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file mpc_trajectory_controller.hpp
 *
 * Acados trajectory-MPC wrapped behind the IController interface.
 *
 * Expects per-stage (position, velocity, acceleration) references for all
 * N+1 prediction stages. Yaw reference is taken from stage 0 and broadcast
 * across all stages (matching the legacy mpc_trajectory_controller behaviour).
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_ADAPTERS_MPC_TRAJECTORY_CONTROLLER_HPP_
#define MPC_EXAMPLES_ADAPTERS_MPC_TRAJECTORY_CONTROLLER_HPP_

#include <Eigen/Dense>

#include <memory>
#include <string>
#include <vector>

#include "acados_mpc/acados_mpc.hpp"
#include "framework/controller_base.hpp"

namespace mpc_examples::adapters {

/**
 * @brief IController adapter around the trajectory variant of the acados MPC.
 *
 * Horizon size: N+1 (feeds per-stage position/velocity/acceleration).
 * Required reference fields: kPosition | kVelocity | kAcceleration.
 */
class MpcTrajectoryController : public framework::IController {
public:
  struct Config {
    std::string mpc_yaml_path;  //!< Path to the acados MPC YAML definition.
  };

  explicit MpcTrajectoryController(const Config& cfg);

  /**
   * @brief Load the adapter configuration from a YAML file.
   *
   * The @p path is stored as @c mpc_yaml_path and later passed to
   * acados_mpc::configureMpcFromYaml() during initialize().
   */
  static Config loadConfigFromYaml(const std::string& path);

  void initialize(const mav_model::State& initial_state, const ExampleConfig& example_cfg) override;

  int referenceHorizonSize() const override { return horizon_steps_ + 1; }
  double referenceHorizonDt() const override { return dt_horizon_; }
  double controlPeriod() const override { return control_period_; }

  framework::ControlCommand computeCommand(
      const mav_model::State& state,
      const std::vector<framework::ReferenceSample>& references) override;

  framework::ReferenceFieldMask requiredReferenceFields() const override {
    return framework::makeMask({framework::ReferenceField::kPosition,
                                framework::ReferenceField::kVelocity,
                                framework::ReferenceField::kAcceleration});
  }

  const std::string& name() const override { return name_; }
  double lastSolveTimeMicros() const override { return last_solve_us_; }

private:
  Config cfg_;
  std::unique_ptr<acados_mpc::MPC> mpc_;

  double control_period_ = 0.01;
  double dt_horizon_     = 0.05;
  int horizon_steps_     = 0;
  double last_solve_us_  = 0.0;
  std::string name_      = "MpcTrajectoryController";
};

}  // namespace mpc_examples::adapters

#endif  // MPC_EXAMPLES_ADAPTERS_MPC_TRAJECTORY_CONTROLLER_HPP_
