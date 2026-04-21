// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file gcopter_generator.hpp
 *
 * Reference generator that wraps gcopter_lib::TrajectoryGenerator.
 *
 * The whole multi-waypoint trajectory is optimised once in initialize() and
 * evaluate(t) then samples the resulting polynomial, giving accurate
 * per-stage (position, velocity, acceleration) over the controller horizon.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_ADAPTERS_GCOPTER_GENERATOR_HPP_
#define MPC_EXAMPLES_ADAPTERS_GCOPTER_GENERATOR_HPP_

#include <Eigen/Dense>

#include <memory>
#include <string>
#include <vector>

#include "framework/trajectory_generator_base.hpp"
#include "gcopter_lib/trajectory_generator.hpp"
#include "gcopter_lib/types.hpp"

namespace mpc_examples::adapters {

/**
 * @brief ITrajectoryGenerator adapter for the GCOPTER polytope-SFC optimiser.
 *
 * Produces full (position, velocity, acceleration) references along a
 * multi-waypoint mission. The trajectory is computed offline at
 * initialize() time; update()/evaluate() just sample the polynomial.
 */
class GcopterGenerator : public framework::ITrajectoryGenerator {
public:
  struct Config {
    gcopter_lib::DroneParameters    drone_params{};
    gcopter_lib::DroneLimits        drone_limits{};
    gcopter_lib::OptimizationConfig optimization{};

    //!< Uniform AABB half-margin used when anchors are disabled.
    double corridor_margin         = 2.0;
    //!< Tight half-margin for pinch segments around intermediate waypoints.
    //!< <= 0 disables the anchor mechanism.
    double waypoint_margin         = 0.0;
    //!< Offset of anchor points along the path at intermediate waypoints.
    //!< <= 0 disables the anchor mechanism.
    double waypoint_anchor_radius  = 0.0;
  };

  explicit GcopterGenerator(const Config& cfg);

  /**
   * @brief Load the generator configuration from a YAML file that matches
   *        gcopter_lib's config_trajectory.yaml schema.
   */
  static Config loadConfigFromYaml(const std::string& path);

  void initialize(const std::vector<Eigen::Vector3d>& waypoints,
                  const mav_model::State& initial_state,
                  const ExampleConfig& example_cfg) override;

  void update(double t, const mav_model::State& state) override;

  framework::ReferenceSample evaluate(double t) const override;

  bool isFinished(double t) const override { return t >= duration_; }

  int currentWaypointIndex() const override { return wp_index_; }

  framework::ReferenceFieldMask providedReferenceFields() const override {
    return framework::makeMask({framework::ReferenceField::kPosition,
                                framework::ReferenceField::kVelocity,
                                framework::ReferenceField::kAcceleration});
  }

  const std::string& name() const override { return name_; }

private:
  Config cfg_;
  std::unique_ptr<gcopter_lib::TrajectoryGenerator> ctrl_;

  std::vector<Eigen::Vector3d> waypoints_;
  double duration_      = 0.0;
  int    wp_index_      = 0;
  bool   path_facing_   = true;

  double yaw_ref_hold_  = 0.0;
  double prev_t_        = 0.0;
  bool   has_prev_t_    = false;

  std::string name_     = "GcopterGenerator";
};

}  // namespace mpc_examples::adapters

#endif  // MPC_EXAMPLES_ADAPTERS_GCOPTER_GENERATOR_HPP_
