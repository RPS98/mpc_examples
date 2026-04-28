// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file mav_traj_gen_generator.hpp
 *
 * Reference generator that wraps mav_trajectory_generation_cpp::TrajectoryGenerator,
 * a polynomial trajectory optimiser (degree 10, ETH-ASL upstream wrapped behind a
 * ROS-free facade). At every waypoint transition the scheduler calls
 * onWaypointChanged() with the next target; the optimiser is rerun with a
 * three-waypoint path [current_pos, midpoint, next_waypoint] so the solver always
 * sees ≥ 2 segments. evaluate() samples the resulting spline using a segment-local
 * time (t - t_start).
 *
 * Capabilities: position, velocity, acceleration.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_ADAPTERS_MAV_TRAJ_GEN_GENERATOR_HPP_
#define MPC_EXAMPLES_ADAPTERS_MAV_TRAJ_GEN_GENERATOR_HPP_

#include <Eigen/Dense>

#include <memory>
#include <string>

#include "framework/trajectory_generator_base.hpp"
#include "mav_trajectory_generation_cpp/trajectory_generator.hpp"
#include "mav_trajectory_generation_cpp/types.hpp"

namespace mpc_examples::adapters {

class MavTrajGenGenerator : public framework::ITrajectoryGenerator {
public:
  struct Config {
    mav_trajectory_generation_cpp::OptimizationConfig optimization{};
  };

  explicit MavTrajGenGenerator(const Config& cfg);

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
  std::unique_ptr<mav_trajectory_generation_cpp::TrajectoryGenerator> ctrl_;
  double max_speed_ = 0.0;

  Eigen::Vector3d target_wp_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d hold_pos_  = Eigen::Vector3d::Zero();  //!< Position held when no plan exists.
  double duration_           = 0.0;
  double t_segment_start_    = 0.0;
  bool has_plan_             = false;
  bool path_facing_          = true;

  double yaw_ref_hold_ = 0.0;
  double prev_t_       = 0.0;
  bool has_prev_t_     = false;

  std::string name_ = "MavTrajGenGenerator";
};

}  // namespace mpc_examples::adapters

#endif  // MPC_EXAMPLES_ADAPTERS_MAV_TRAJ_GEN_GENERATOR_HPP_
