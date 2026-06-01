// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file ssa_position_mpc_controller.hpp
 *
 * Acados Steady-State-Aware Position MPC wrapped behind the IController
 * interface.
 *
 * SSA-PMPC carries an extra ``artificial_position`` state (free decision
 * variable, zero dynamics) that the optimizer drives towards the desired
 * set-point via the offset cost ``‖p_a − desired_position‖²_Qr``. The adapter
 * feeds the generator's goal as a constant set-point across the horizon; the
 * admissible (speed-limited) approach emerges from the SSA cost balance
 * instead of from a hand-rolled progressive carrot.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_ADAPTERS_SSA_POSITION_MPC_CONTROLLER_HPP_
#define MPC_EXAMPLES_ADAPTERS_SSA_POSITION_MPC_CONTROLLER_HPP_

#include <Eigen/Dense>

#include <memory>
#include <string>
#include <vector>

#include "acados_ssa_mpc/acados_mpc.hpp"
#include "framework/controller_base.hpp"

namespace mpc_examples::adapters {

/**
 * @brief IController adapter around the SSA Position MPC.
 *
 * Horizon size: 1 (the generator only needs to provide the instantaneous
 * set-point; the SSA artificial reference handles horizon progression).
 * Required reference fields: ``kPosition``.
 */
class SsaPositionMpcController : public framework::IController {
public:
  struct Config {
    std::string mpc_yaml_path;        //!< Path to the SSA MPC YAML definition.
    double max_vel_percentage = 1.0;  //!< Safety knob in (0, 1] applied to sqrt(uh).
  };

  explicit SsaPositionMpcController(const Config& cfg);

  static Config loadConfigFromYaml(const std::string& path);

  void initialize(const mav_model::State& initial_state, const ExampleConfig& example_cfg) override;

  int referenceHorizonSize() const override { return 1; }
  double referenceHorizonDt() const override { return control_period_; }
  double controlPeriod() const override { return control_period_; }

  framework::ControlCommand computeCommand(
      const mav_model::State& state,
      const std::vector<framework::ReferenceSample>& references) override;

  framework::ReferenceFieldMask requiredReferenceFields() const override {
    return framework::makeMask({framework::ReferenceField::kPosition});
  }

  const std::string& name() const override { return name_; }
  double lastSolveTimeMicros() const override { return last_solve_us_; }
  double lastAcadosSolverTimeMicros() const override { return last_acados_solver_us_; }

  /// Stage-1 predicted linear velocity in world frame (the solver's
  /// immediate prediction after applying the first control input).
  Eigen::Vector3d lastDesiredVelocity() const override { return last_desired_velocity_; }
  bool providesDesiredVelocity() const override { return true; }

private:
  Config cfg_;
  std::unique_ptr<acados_ssa_mpc::MPC> mpc_;

  double control_period_                 = 0.01;
  double v_ref_                          = 1.0;
  double last_solve_us_                  = 0.0;
  double last_acados_solver_us_          = 0.0;
  Eigen::Vector3d last_desired_velocity_ = Eigen::Vector3d::Zero();
  std::string name_                      = "SsaPositionMpcController";
};

}  // namespace mpc_examples::adapters

#endif  // MPC_EXAMPLES_ADAPTERS_SSA_POSITION_MPC_CONTROLLER_HPP_
