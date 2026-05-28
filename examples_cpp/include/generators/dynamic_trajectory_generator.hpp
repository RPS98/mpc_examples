// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dynamic_trajectory_generator.hpp
 *
 * Reference generator that wraps dynamic_traj_generator::DynamicTrajectory.
 *
 * Point-to-point contract: the scheduler calls onWaypointChanged() with the
 * next target; at each transition, the underlying library instance is
 * destroyed and reconstructed so that the new segment is generated from
 * scratch with a fresh internal time origin. Time is then evaluated relative
 * to the segment start (t_segment_start_), matching the convention used by
 * the gcopter and mav_traj_gen adapters.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_ADAPTERS_DYNAMIC_TRAJECTORY_GENERATOR_HPP_
#define MPC_EXAMPLES_ADAPTERS_DYNAMIC_TRAJECTORY_GENERATOR_HPP_

#include <Eigen/Dense>

#include <memory>
#include <string>

#include "dynamic_trajectory_generator/dynamic_trajectory.hpp"
#include "framework/trajectory_generator_base.hpp"

namespace mpc_examples::adapters {

/**
 * @brief ITrajectoryGenerator adapter for dynamic_traj_generator::DynamicTrajectory.
 *
 * Capabilities: position, velocity, acceleration.
 */
class DynamicTrajectoryGenerator : public framework::ITrajectoryGenerator {
public:
  /// mode: "point_to_point" (default, replan per waypoint — position/controller
  /// comparison experiments) or "global" (one min-jerk through all mission
  /// waypoints, sampled by time — racing/circuit tracking). See
  /// configs/generators/config_dynamic.yaml.
  struct Config {
    std::string mode = "point_to_point";
  };

  explicit DynamicTrajectoryGenerator(const Config& cfg);

  /// Reads the optional `mode` key; the file is otherwise a placeholder.
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
  std::unique_ptr<dynamic_traj_generator::DynamicTrajectory> traj_;

  double max_speed_          = 0.0;
  Eigen::Vector3d target_wp_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d hold_pos_  = Eigen::Vector3d::Zero();
  double t_segment_start_    = 0.0;  // sim time at which the active segment began
  double t_min_              = 0.0;  // sim time, = t_segment_start_
  double t_max_              = 0.0;  // sim time, = t_segment_start_ + T_seg
  bool has_plan_             = false;
  bool segment_completed_    = false;
  bool path_facing_          = true;
  bool global_               = false;  // build one trajectory through all waypoints
  double eval_dt_            = 0.0;    // horizon step; clamps the final eval time

  double yaw_ref_hold_ = 0.0;
  double prev_t_       = 0.0;
  bool has_prev_t_     = false;

  framework::ReferenceSample last_sample_{};

  std::string name_ = "DynamicTrajectoryGenerator";
};

}  // namespace mpc_examples::adapters

#endif  // MPC_EXAMPLES_ADAPTERS_DYNAMIC_TRAJECTORY_GENERATOR_HPP_
