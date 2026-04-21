// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file waypoint_reference_generator.hpp
 *
 * Reference generator that emits piecewise-constant waypoint targets.
 *
 * Produces position-only references (velocity and acceleration stay at zero).
 * Equivalent to the legacy clampReference() + waypoint advance logic extracted
 * into an ITrajectoryGenerator so the WaypointsSimulator stays generator-agnostic.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_ADAPTERS_WAYPOINT_REFERENCE_GENERATOR_HPP_
#define MPC_EXAMPLES_ADAPTERS_WAYPOINT_REFERENCE_GENERATOR_HPP_

#include <Eigen/Dense>

#include <string>
#include <vector>

#include "framework/trajectory_generator_base.hpp"

namespace mpc_examples::adapters {

/**
 * @brief Piecewise-constant waypoint reference with clamping and path-facing yaw.
 *
 * - Keeps the active waypoint fixed until the drone enters @c reach_threshold,
 *   then advances to the next one. Mission finishes when the last waypoint is
 *   reached.
 * - Clamps the emitted position so it is never more than @c d_max metres from
 *   the last observed state; prevents large step references when far from the
 *   goal (equivalent to the legacy clampReference()).
 * - When @c path_facing is true the yaw reference points from the current
 *   position towards the active waypoint; otherwise yaw is held at zero.
 *
 * Capabilities: provides ReferenceField::kPosition only. Velocity and
 * acceleration are always zero (callers that expect them will see the
 * compatibility warning emitted by WaypointsSimulator).
 */
class WaypointReferenceGenerator : public framework::ITrajectoryGenerator {
public:
  struct Config {
    double d_max           = 2.0;   //!< Max reference distance from current position [m].
    double reach_threshold = 0.1;   //!< Waypoint acceptance radius [m].
  };

  explicit WaypointReferenceGenerator(const Config& cfg);

  /**
   * @brief Load the generator configuration from a YAML file.
   *
   * Keys (all optional; defaults above):
   * @code
   * d_max: 2.0
   * reach_threshold: 0.1
   * @endcode
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
    return framework::makeMask({framework::ReferenceField::kPosition});
  }

  const std::string& name() const override { return name_; }

private:
  Config cfg_;
  std::vector<Eigen::Vector3d> waypoints_;
  std::size_t wp_index_          = 0U;
  bool finished_                 = false;
  bool path_facing_              = true;
  Eigen::Vector3d last_position_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d reference_wp_  = Eigen::Vector3d::Zero();
  double cached_yaw_             = 0.0;
  std::string name_              = "WaypointReferenceGenerator";
};

}  // namespace mpc_examples::adapters

#endif  // MPC_EXAMPLES_ADAPTERS_WAYPOINT_REFERENCE_GENERATOR_HPP_
