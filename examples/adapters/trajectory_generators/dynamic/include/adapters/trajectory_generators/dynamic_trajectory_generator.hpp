// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dynamic_trajectory_generator.hpp
 *
 * Reference generator that wraps dynamic_traj_generator::DynamicTrajectory.
 *
 * The underlying library computes a polynomial trajectory through the
 * waypoints in its own thread. setWaypoints()/getMaxTime() block until the
 * first optimisation finishes; subsequent update()/evaluate() calls are
 * lightweight polynomial evaluations.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_ADAPTERS_DYNAMIC_TRAJECTORY_GENERATOR_HPP_
#define MPC_EXAMPLES_ADAPTERS_DYNAMIC_TRAJECTORY_GENERATOR_HPP_

#include <Eigen/Dense>

#include <memory>
#include <string>
#include <vector>

#include "dynamic_trajectory_generator/dynamic_trajectory.hpp"
#include "framework/trajectory_generator_base.hpp"

namespace mpc_examples::adapters {

/**
 * @brief ITrajectoryGenerator adapter for dynamic_traj_generator::DynamicTrajectory.
 *
 * Produces full (position, velocity, acceleration) along a multi-waypoint
 * trajectory computed once in initialize(). Supports asynchronous
 * re-optimisation performed by the underlying library. Once the trajectory
 * end time is reached the reference is held stationary at the last sample.
 */
class DynamicTrajectoryGenerator : public framework::ITrajectoryGenerator {
public:
  struct Config {
    // The adapter has no tunables: travel speed is sourced from
    // ExampleConfig::max_speed at initialize() time.
  };

  explicit DynamicTrajectoryGenerator(const Config& cfg);

  /// Placeholder for API symmetry: the adapter does not require a YAML file.
  static Config loadConfigFromYaml(const std::string& path);

  void initialize(const std::vector<Eigen::Vector3d>& waypoints,
                  const mav_model::State& initial_state,
                  const ExampleConfig& example_cfg) override;

  void update(double t, const mav_model::State& state) override;

  framework::ReferenceSample evaluate(double t) const override;

  bool isFinished(double t) const override { return t >= t_max_; }

  int currentWaypointIndex() const override { return wp_index_; }

  framework::ReferenceFieldMask providedReferenceFields() const override {
    return framework::makeMask({framework::ReferenceField::kPosition,
                                framework::ReferenceField::kVelocity,
                                framework::ReferenceField::kAcceleration});
  }

  const std::string& name() const override { return name_; }

private:
  Config cfg_;
  std::unique_ptr<dynamic_traj_generator::DynamicTrajectory> traj_;

  std::vector<Eigen::Vector3d> waypoints_;
  double t_min_         = 0.0;
  double t_max_         = 0.0;
  int    wp_index_      = 0;
  bool   path_facing_   = true;

  double yaw_ref_hold_  = 0.0;
  double prev_t_        = 0.0;
  bool   has_prev_t_    = false;

  framework::ReferenceSample last_sample_{};

  std::string name_     = "DynamicTrajectoryGenerator";
};

}  // namespace mpc_examples::adapters

#endif  // MPC_EXAMPLES_ADAPTERS_DYNAMIC_TRAJECTORY_GENERATOR_HPP_
