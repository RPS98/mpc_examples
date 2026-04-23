// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file waypoint_reference_generator.hpp
 *
 * Reference generator that emits piecewise-constant waypoint targets.
 *
 * Produces position-only references (velocity and acceleration stay at zero).
 * Point-to-point contract: the scheduler notifies the next target through
 * onWaypointChanged(); no waypoint list is held internally.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_ADAPTERS_WAYPOINT_REFERENCE_GENERATOR_HPP_
#define MPC_EXAMPLES_ADAPTERS_WAYPOINT_REFERENCE_GENERATOR_HPP_

#include <Eigen/Dense>

#include <string>

#include "framework/trajectory_generator_base.hpp"

namespace mpc_examples::adapters {

/**
 * @brief Piecewise-constant waypoint reference with clamping and path-facing yaw.
 *
 * Emits the active waypoint position (clamped to cfg_.d_max from the last
 * observed drone position) with velocity and acceleration held at zero.
 *
 * Capabilities: provides ReferenceField::kPosition only.
 */
class WaypointReferenceGenerator : public framework::ITrajectoryGenerator {
public:
  struct Config {
    double d_max           = 2.0;   //!< Max reference distance from current position [m].
    double reach_threshold = 0.1;   //!< Minimum planar distance for yaw alignment [m].
  };

  explicit WaypointReferenceGenerator(const Config& cfg);

  static Config loadConfigFromYaml(const std::string& path);

  void initialize(const mav_model::State& initial_state,
                  const ExampleConfig& example_cfg) override;

  void onWaypointChanged(const Eigen::Vector3d& next_waypoint,
                         const mav_model::State& state,
                         double t_start) override;

  void update(double t, const mav_model::State& state) override;

  framework::ReferenceSample evaluate(double t) const override;

  framework::ReferenceFieldMask providedReferenceFields() const override {
    return framework::makeMask({framework::ReferenceField::kPosition});
  }

  const std::string& name() const override { return name_; }

private:
  Config cfg_;
  bool path_facing_              = true;
  Eigen::Vector3d last_position_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d target_wp_     = Eigen::Vector3d::Zero();
  double cached_yaw_             = 0.0;
  std::string name_              = "WaypointReferenceGenerator";
};

}  // namespace mpc_examples::adapters

#endif  // MPC_EXAMPLES_ADAPTERS_WAYPOINT_REFERENCE_GENERATOR_HPP_
