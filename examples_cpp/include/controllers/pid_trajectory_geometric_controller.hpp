// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file pid_trajectory_geometric_controller.hpp
 *
 * Direct trajectory PID + geometric attitude controller adapter.
 *
 * Wraps mav_controllers::TrajectoryController (parallel pos/vel feedback) and
 * geometric_controller::GeometricController behind the IController interface
 * for trajectory-based references (position + velocity + acceleration).
 *
 * Unlike PidPositionGeometricController (which cascades position → velocity),
 * this controller feeds position and velocity errors in parallel into a single
 * PID that outputs acceleration, matching the trajectory generator's expected
 * reference semantics.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_ADAPTERS_PID_TRAJECTORY_GEOMETRIC_CONTROLLER_HPP_
#define MPC_EXAMPLES_ADAPTERS_PID_TRAJECTORY_GEOMETRIC_CONTROLLER_HPP_

#include <Eigen/Dense>

#include <memory>
#include <string>
#include <vector>

#include "framework/controller_base.hpp"
#include "geometric_controller/geometric_controller.hpp"
#include "pid_controller/pid.hpp"
#include "pid_controllers/trajectory_controller.hpp"

namespace mpc_examples::adapters {

/**
 * @brief Direct trajectory PID for smooth trajectory tracking (horizon size == 1).
 *
 * Pipeline, executed once per control period:
 *   1. trajectory PID: (state.pos, state.vel, ref.pos, ref.vel, ref.acc) → acc_des
 *      (parallel pos/vel feedback into a single PID, with acceleration feedforward)
 *   2. geometric:      (acc_des, ref.yaw, state.orientation)    → (thrust, rates)
 *
 * Requires ReferenceField::kPosition | kVelocity | kAcceleration.
 * Suitable for smooth trajectory generators (gcopter, jerk_limited, dynamic,
 * mav_traj_gen) and trajectory MPC.
 */
class PidTrajectoryGeometricController : public framework::IController {
public:
  struct Config {
    pid_controller::PIDParameters<double> trajectory_pid_params;
    geometric_controller::AttitudeGeometricControllerParameters<double> attitude_params;
    geometric_controller::RatesGeometricControllerParameters<double> rates_params;
    double v_max = 1.0;  //!< Reserved for reference saturation if needed [m/s].
  };

  explicit PidTrajectoryGeometricController(const Config& cfg);

  /**
   * @brief Load the controller configuration from a YAML file.
   *
   * Expected structure (config_pid_trajectory.yaml):
   * @code
   * v_max: 3.0
   * controller:
   *   trajectory: {kp, ki, kd, antiwindup_cte, alpha}
   *   geometric:  {mass, rotation_kp}
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
    return framework::makeMask({framework::ReferenceField::kPosition,
                                framework::ReferenceField::kVelocity,
                                framework::ReferenceField::kAcceleration});
  }

  const std::string& name() const override { return name_; }
  double lastSolveTimeMicros() const override { return last_solve_us_; }

private:
  Config cfg_;
  std::unique_ptr<pid_controllers::TrajectoryController<double>> traj_ctrl_;
  std::unique_ptr<geometric_controller::GeometricController<double>> geo_ctrl_;

  double control_period_ = 0.01;
  double last_solve_us_  = 0.0;
  std::string name_      = "PidTrajectoryGeometricController";
};

}  // namespace mpc_examples::adapters

#endif  // MPC_EXAMPLES_ADAPTERS_PID_TRAJECTORY_GEOMETRIC_CONTROLLER_HPP_
