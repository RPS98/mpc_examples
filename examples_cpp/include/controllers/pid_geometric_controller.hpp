// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file pid_geometric_controller.hpp
 *
 * Cascaded PID position + velocity + geometric attitude controller adapter.
 *
 * Wraps mav_controllers::{PositionController, VelocityController} and
 * geometric_controller::GeometricController behind the IController interface
 * so WaypointsSimulator can drive it uniformly with any trajectory generator.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_ADAPTERS_PID_GEOMETRIC_CONTROLLER_HPP_
#define MPC_EXAMPLES_ADAPTERS_PID_GEOMETRIC_CONTROLLER_HPP_

#include <Eigen/Dense>

#include <memory>
#include <string>
#include <vector>

#include "framework/controller_base.hpp"
#include "geometric_controller/geometric_controller.hpp"
#include "pid_controller/pid.hpp"
#include "pid_controllers/position_controller.hpp"
#include "pid_controllers/velocity_controller.hpp"

namespace mpc_examples::adapters {

/**
 * @brief Outer-loop PID cascade (horizon size == 1).
 *
 * Pipeline, executed once per control period:
 *   1. position PID:   (state.position, ref.position)        → vel_des
 *   2. (optional) add ref.velocity feedforward when @c Config::feedforward_velocity
 *   3. saturate to v_max (preserves direction)
 *   4. velocity PID:   (state.velocity, vel_des)             → acc_des
 *   5. geometric:      (acc_des, ref.yaw, state.orientation) → (thrust, rates)
 *
 * Requires ReferenceField::kPosition; ReferenceField::kVelocity is added when
 * @c feedforward_velocity is enabled so WaypointsSimulator emits a clear
 * compatibility warning if the active generator does not provide it.
 */
class PidGeometricController : public framework::IController {
public:
  struct Config {
    pid_controller::PIDParameters<double> position_pid_params;
    pid_controller::PIDParameters<double> velocity_pid_params;
    geometric_controller::AttitudeGeometricControllerParameters<double> attitude_params;
    geometric_controller::RatesGeometricControllerParameters<double> rates_params;
    double v_max = 1.0;  //!< Saturation applied to the position-PID velocity output [m/s].
    /// When true, sum the reference linear velocity into the velocity setpoint
    /// fed to the velocity PID. Recommended when the upstream generator emits
    /// a smooth trajectory (jerk_limited, gcopter, dynamic, mav_traj_gen).
    /// Disabled by default to preserve the legacy waypoint-tracking behaviour.
    bool feedforward_velocity = false;
  };

  explicit PidGeometricController(const Config& cfg);

  /**
   * @brief Load the controller configuration from a YAML file.
   *
   * Expected structure (matches the legacy config_pid.yaml):
   * @code
   * v_max: 3.0
   * controller:
   *   position:  {kp, ki, kd, antiwindup_cte, alpha}
   *   velocity:  {kp, ki, kd, antiwindup_cte, alpha,
   *               saturation_upper, saturation_lower}
   *   geometric: {mass, rotation_kp}
   * @endcode
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
    if (cfg_.feedforward_velocity) {
      return framework::makeMask(
          {framework::ReferenceField::kPosition, framework::ReferenceField::kVelocity});
    }
    return framework::makeMask({framework::ReferenceField::kPosition});
  }

  const std::string& name() const override { return name_; }
  double lastSolveTimeMicros() const override { return last_solve_us_; }

private:
  Config cfg_;
  std::unique_ptr<pid_controllers::PositionController<double>> pos_ctrl_;
  std::unique_ptr<pid_controllers::VelocityController<double>> vel_ctrl_;
  std::unique_ptr<geometric_controller::GeometricController<double>> geo_ctrl_;

  double control_period_ = 0.01;
  double last_solve_us_  = 0.0;
  std::string name_      = "PidGeometricController";
};

}  // namespace mpc_examples::adapters

#endif  // MPC_EXAMPLES_ADAPTERS_PID_GEOMETRIC_CONTROLLER_HPP_
