// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file jerk_limited_generator.hpp
 *
 * Reference generator that wraps WaypointTrajectoryController from
 * trajectory_generator_jerk_limited.
 *
 * Point-to-point contract: the scheduler calls onWaypointChanged() with the
 * next target; between transitions, update() integrates the S-curve toward
 * that target. Waypoint ownership lives in the scheduler.
 *
 * The underlying generator is stateful (per-tick S-curve integration) so
 * evaluate() returns the latest stage-0 setpoint for every horizon sample.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_ADAPTERS_JERK_LIMITED_GENERATOR_HPP_
#define MPC_EXAMPLES_ADAPTERS_JERK_LIMITED_GENERATOR_HPP_

#include <Eigen/Dense>

#include <memory>
#include <string>

#include "framework/trajectory_generator_base.hpp"
#include "trajectory_generator_jerk_limited/waypoint_controller.hpp"

namespace mpc_examples::adapters {

/**
 * @brief ITrajectoryGenerator adapter for the jerk-limited S-curve generator.
 *
 * Capabilities: position, velocity, acceleration.
 */
class JerkLimitedGenerator : public framework::ITrajectoryGenerator {
public:
  struct Config {
    double max_acceleration   = 0.0;  //!< <= 0 disables the 3D acceleration bound.
    double max_jerk           = 0.0;  //!< <= 0 disables the jerk bound.
    double max_tracking_error = 0.0;  //!< <= 0 disables the time-stretch gate.
  };
  // The 3D speed bound comes from ExampleConfig::max_speed at initialize() time.

  explicit JerkLimitedGenerator(const Config& cfg);

  static Config loadConfigFromYaml(const std::string& path);

  void initialize(const mav_model::State& initial_state, const ExampleConfig& example_cfg) override;

  void onWaypointChanged(const Eigen::Vector3d& next_waypoint,
                         const mav_model::State& state,
                         double t_start) override;

  void update(double t, const mav_model::State& state) override;

  framework::ReferenceSample evaluate(double t) const override;

  framework::ReferenceFieldMask providedReferenceFields() const override {
    return framework::makeMask({framework::ReferenceField::kPosition,
                                framework::ReferenceField::kVelocity,
                                framework::ReferenceField::kAcceleration});
  }

  const std::string& name() const override { return name_; }

private:
  Config cfg_;
  std::unique_ptr<trajectory_generator_jerk_limited::WaypointTrajectoryController> ctrl_;

  Eigen::Vector3d target_wp_ = Eigen::Vector3d::Zero();
  bool path_facing_          = true;

  double prev_t_        = 0.0;
  bool has_prev_t_      = false;
  double last_sample_t_ = 0.0;  //!< Sim-time at which last_sample_ was produced.

  framework::ReferenceSample last_sample_{};
  double yaw_ref_hold_ = 0.0;

  std::string name_ = "JerkLimitedGenerator";
};

}  // namespace mpc_examples::adapters

#endif  // MPC_EXAMPLES_ADAPTERS_JERK_LIMITED_GENERATOR_HPP_
