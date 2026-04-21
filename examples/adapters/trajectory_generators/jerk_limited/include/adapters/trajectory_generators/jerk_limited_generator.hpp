// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file jerk_limited_generator.hpp
 *
 * Reference generator that wraps WaypointTrajectoryController from
 * trajectory_generator_jerk_limited.
 *
 * The underlying generator is stateful (per-tick S-curve integration) so
 * evaluate() returns the latest stage-0 setpoint for every horizon sample k.
 * MPC controllers that consume velocity and acceleration at stages k>0 will
 * therefore see piecewise-constant feedforward within a single outer step;
 * this is acceptable because the outer step (mpc_dt or pid_dt) is short.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_ADAPTERS_JERK_LIMITED_GENERATOR_HPP_
#define MPC_EXAMPLES_ADAPTERS_JERK_LIMITED_GENERATOR_HPP_

#include <Eigen/Dense>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "framework/trajectory_generator_base.hpp"
#include "trajectory_generator_jerk_limited/waypoint_controller.hpp"

namespace mpc_examples::adapters {

/**
 * @brief ITrajectoryGenerator adapter for the jerk-limited S-curve generator.
 *
 * Produces (position, velocity, acceleration) along a multi-waypoint mission:
 * advances the internal generator by the outer control dt on every update(),
 * switches the target waypoint when the drone enters the acceptance radius,
 * and reports isFinished() once the last waypoint has been reached.
 */
class JerkLimitedGenerator : public framework::ITrajectoryGenerator {
public:
  struct Config {
    double max_acceleration    = 0.0;  //!< <= 0 disables the 3D acceleration bound.
    double max_jerk            = 0.0;  //!< <= 0 disables the jerk bound.
    double max_tracking_error  = 0.0;  //!< <= 0 disables the time-stretch gate.
    double reach_threshold     = 0.1;  //!< Waypoint acceptance radius [m].
  };
  // The 3D speed bound comes from ExampleConfig::max_speed at initialize() time.

  explicit JerkLimitedGenerator(const Config& cfg);

  /**
   * @brief Load the generator configuration from a YAML file.
   *
   * Keys (all optional except where noted):
   * @code
   * max_acceleration:     5.0
   * max_jerk:             10.0
   * max_tracking_error:   0.0
   * reach_threshold:      0.1
   * @endcode
   * The 3D speed bound is sourced from ExampleConfig::max_speed at initialize() time.
   */
  static Config loadConfigFromYaml(const std::string& path);

  void initialize(const std::vector<Eigen::Vector3d>& waypoints,
                  const mav_model::State& initial_state,
                  const ExampleConfig& example_cfg) override;

  void update(double t, const mav_model::State& state) override;

  framework::ReferenceSample evaluate(double t) const override;

  bool isFinished(double /*t*/) const override { return finished_; }

  int currentWaypointIndex() const override { return static_cast<int>(wp_index_); }

  framework::ReferenceFieldMask providedReferenceFields() const override {
    return framework::makeMask({framework::ReferenceField::kPosition,
                                framework::ReferenceField::kVelocity,
                                framework::ReferenceField::kAcceleration});
  }

  const std::string& name() const override { return name_; }

private:
  Config cfg_;
  std::unique_ptr<trajectory_generator_jerk_limited::WaypointTrajectoryController> ctrl_;

  std::vector<Eigen::Vector3d> waypoints_;
  std::size_t wp_index_ = 0U;
  bool finished_        = false;
  bool path_facing_     = true;

  double prev_t_        = 0.0;
  bool has_prev_t_      = false;

  framework::ReferenceSample last_sample_{};
  double yaw_ref_hold_  = 0.0;

  std::string name_     = "JerkLimitedGenerator";
};

}  // namespace mpc_examples::adapters

#endif  // MPC_EXAMPLES_ADAPTERS_JERK_LIMITED_GENERATOR_HPP_
