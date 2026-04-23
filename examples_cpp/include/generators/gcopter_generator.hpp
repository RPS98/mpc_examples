// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file gcopter_generator.hpp
 *
 * Reference generator that wraps gcopter_lib::TrajectoryGenerator.
 *
 * Point-to-point contract: the scheduler calls onWaypointChanged() with the
 * next target; at each transition, the L-BFGS solver is re-run with a
 * two-waypoint path [current_pos, next_waypoint]. evaluate() samples the
 * resulting polynomial using a segment-local time (t - t_start).
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_ADAPTERS_GCOPTER_GENERATOR_HPP_
#define MPC_EXAMPLES_ADAPTERS_GCOPTER_GENERATOR_HPP_

#include <Eigen/Dense>

#include <memory>
#include <string>

#include "framework/trajectory_generator_base.hpp"
#include "gcopter_lib/trajectory_generator.hpp"
#include "gcopter_lib/types.hpp"

namespace mpc_examples::adapters {

/**
 * @brief ITrajectoryGenerator adapter for the GCOPTER polytope-SFC optimiser.
 *
 * Capabilities: position, velocity, acceleration.
 */
class GcopterGenerator : public framework::ITrajectoryGenerator {
public:
  struct Config {
    gcopter_lib::DroneParameters    drone_params{};
    gcopter_lib::DroneLimits        drone_limits{};
    gcopter_lib::OptimizationConfig optimization{};

    //!< Uniform AABB half-margin around the straight-line segment [m].
    double corridor_margin = 2.0;
  };

  explicit GcopterGenerator(const Config& cfg);

  static Config loadConfigFromYaml(const std::string& path);

  void initialize(const mav_model::State& initial_state,
                  const ExampleConfig& example_cfg) override;

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
  std::unique_ptr<gcopter_lib::TrajectoryGenerator> ctrl_;

  Eigen::Vector3d target_wp_  = Eigen::Vector3d::Zero();
  Eigen::Vector3d hold_pos_   = Eigen::Vector3d::Zero();  //!< Position held when no plan exists.
  double          duration_   = 0.0;
  double          t_segment_start_ = 0.0;
  bool            has_plan_   = false;
  bool            path_facing_ = true;

  double yaw_ref_hold_ = 0.0;
  double prev_t_       = 0.0;
  bool   has_prev_t_   = false;

  std::string name_ = "GcopterGenerator";
};

}  // namespace mpc_examples::adapters

#endif  // MPC_EXAMPLES_ADAPTERS_GCOPTER_GENERATOR_HPP_
