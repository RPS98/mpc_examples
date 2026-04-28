// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file mpc_position_controller.hpp
 *
 * Acados position-MPC wrapped behind the IController interface.
 *
 * Uses the one-shot goal strategy from the legacy mpc_controller example:
 *   - the generator provides a single position target (horizon size 1);
 *   - the adapter internally expands it into N+1 stage references via
 *     setProgressiveReferences(), sampling along the straight line that joins
 *     the current state to the goal at speed example_cfg.max_speed.
 *
 * Controllers with richer horizon expectations (velocity/acceleration feed-
 * forward) should use MpcTrajectoryController instead.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_ADAPTERS_MPC_POSITION_CONTROLLER_HPP_
#define MPC_EXAMPLES_ADAPTERS_MPC_POSITION_CONTROLLER_HPP_

#include <Eigen/Dense>

#include <memory>
#include <string>
#include <vector>

#include "acados_mpc/acados_mpc.hpp"
#include "framework/controller_base.hpp"

namespace mpc_examples::adapters {

/**
 * @brief IController adapter around the position variant of the acados MPC.
 *
 * Horizon size: 1 (the generator only needs to provide the instantaneous goal;
 * stage references are built internally from max_speed, N and dt_horizon).
 * Required reference fields: kPosition.
 */
class MpcPositionController : public framework::IController {
public:
  struct Config {
    std::string mpc_yaml_path;       //!< Path to the acados MPC YAML definition.
    double soft_speed_margin = 1.0;  //!< Fraction of max_speed used as soft speed bound.
  };

  explicit MpcPositionController(const Config& cfg);

  /**
   * @brief Load the adapter configuration from a YAML file.
   *
   * Expected structure (mirrors config_mpc.yaml):
   * @code
   * mpc:
   *   soft_speed_margin: 1.0  # optional (default 1.0)
   * @endcode
   *
   * The @p path itself is stored as @c mpc_yaml_path and later passed to
   * acados_mpc::configureMpcFromYaml() during initialize().
   */
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

private:
  Config cfg_;
  std::unique_ptr<acados_mpc::MPC> mpc_;

  double control_period_ = 0.01;
  double v_ref_          = 1.0;
  double dt_horizon_     = 0.05;
  int horizon_steps_     = 0;
  double last_solve_us_  = 0.0;
  std::string name_      = "MpcPositionController";
};

}  // namespace mpc_examples::adapters

#endif  // MPC_EXAMPLES_ADAPTERS_MPC_POSITION_CONTROLLER_HPP_
