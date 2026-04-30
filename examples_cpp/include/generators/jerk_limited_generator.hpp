// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file jerk_limited_generator.hpp
 *
 * Reference generator that wraps the one-shot
 * trajectory_generator_jerk_limited::TrajectoryGenerator.
 *
 * Point-to-point contract: the scheduler calls onWaypointChanged() with the
 * next target; at each transition, generate() runs the offline S-curve
 * simulator with [current_pos, next_waypoint]. evaluate() then samples the
 * resulting trajectory using a segment-local time (t - t_start). Waypoint
 * ownership lives in the scheduler.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_ADAPTERS_JERK_LIMITED_GENERATOR_HPP_
#define MPC_EXAMPLES_ADAPTERS_JERK_LIMITED_GENERATOR_HPP_

#include <Eigen/Dense>

#include <memory>
#include <string>

#include "framework/trajectory_generator_base.hpp"
#include "trajectory_generator_jerk_limited/trajectory_generator.hpp"

namespace mpc_examples::adapters {

/**
 * @brief ITrajectoryGenerator adapter for the jerk-limited S-curve generator.
 *
 * Capabilities: position, velocity, acceleration.
 */
class JerkLimitedGenerator : public framework::ITrajectoryGenerator {
public:
  struct Config {
    double max_acceleration = 0.0;  //!< <= 0 disables the 3D acceleration bound.
    double max_jerk         = 0.0;  //!< <= 0 disables the jerk bound.
    //!< Forwarded to TrajectoryParameters but unused by the one-shot API
    //!< (the streaming time-stretch gate does not apply to offline planning).
    //!< Kept for schema compatibility; <= 0 disables the bound.
    double max_tracking_error = 0.0;
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
  std::unique_ptr<trajectory_generator_jerk_limited::TrajectoryGenerator> ctrl_;

  Eigen::Vector3d target_wp_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d hold_pos_  = Eigen::Vector3d::Zero();  //!< Position held when no plan exists.
  double duration_           = 0.0;
  double t_segment_start_    = 0.0;
  bool has_plan_             = false;
  bool path_facing_          = true;
  double max_speed_          = 0.0;  //!< Cached from ExampleConfig in initialize().

  double yaw_ref_hold_ = 0.0;
  double prev_t_       = 0.0;
  bool has_prev_t_     = false;

  std::string name_ = "JerkLimitedGenerator";
};

}  // namespace mpc_examples::adapters

#endif  // MPC_EXAMPLES_ADAPTERS_JERK_LIMITED_GENERATOR_HPP_
