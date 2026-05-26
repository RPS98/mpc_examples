// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file circuit_generator.hpp
 *
 * Closed-loop race-circuit reference generator. Reads a per-gate-frame
 * mission + gates_config, builds an arc-length-reparametrised Hermite
 * spline, and emits (position, velocity, acceleration) samples at any
 * absolute time along the lap.
 *
 * Mirrors examples_py/examples_py/generators/circuit_generator.py.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_ADAPTERS_CIRCUIT_GENERATOR_HPP_
#define MPC_EXAMPLES_ADAPTERS_CIRCUIT_GENERATOR_HPP_

#include <Eigen/Dense>

#include <memory>
#include <string>
#include <vector>

#include "framework/trajectory_generator_base.hpp"
#include "spline_trajectory_generator/arc_length_reparametrization.hpp"
#include "spline_trajectory_generator/trajectory_generator.hpp"

namespace mpc_examples::adapters {

/**
 * @brief ITrajectoryGenerator backed by a spline along a gate-circuit.
 *
 * Capabilities: provides POSITION | VELOCITY | ACCELERATION.
 *
 * Path resolution precedence (mission/gates YAMLs):
 *   1. env var ``CIRCUIT_MISSION_YAML`` / ``CIRCUIT_GATES_YAML``
 *      (used by higher-level launchers to inject their own mission).
 *   2. config YAML values.
 *   3. package demo defaults (`configs/missions/demo_*.yaml`).
 */
class CircuitGenerator : public framework::ITrajectoryGenerator {
public:
  struct Config {
    std::string mission_yaml;
    std::string gates_yaml;
    double desired_speed         = 4.0;  //!< Cruise speed along the path [m/s].
    double origin_offset_m       = 0.0;  //!< 0 → anchor at the drone's takeoff pose.
    double closing_exit_margin_m = 3.0;  //!< Extend spline past the closing gate [m].
    double target_segment_length = 1.0;
    int    samples_per_segment   = 400;
    int    n_req_points          = 20;   //!< Spline window size (MPCC knots).
  };

  explicit CircuitGenerator(const Config& cfg);

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

  /**
   * @brief Compose origin + canonical fly waypoints + exit setpoint.
   *
   * Exposed as `static` so :class:`MpccController` can build its own
   * spline from the same anchors without duplicating the loader logic.
   *   - Origin: drone start (forced to @p initial_position if
   *     @p origin_offset_m is 0, otherwise nudged behind gate1 along
   *     its forward normal so the lap-start sign flip can fire).
   *   - Mission waypoints: copied verbatim from the YAML.
   *   - Exit: placed @p closing_exit_margin_m past the closing
   *     waypoint along its forward velocity.
   */
  static std::vector<spline::Setpoint> buildSetpoints(
      const Eigen::Vector3d& initial_position,
      const std::string& mission_yaml,
      const std::string& gates_yaml,
      double origin_offset_m,
      double closing_exit_margin_m);

private:
  Config cfg_;
  std::unique_ptr<spline::TrajectoryGenerator> traj_;
  spline::arc_length_reparam::ArcLengthReparametrization window_;
  double s_eval_   = 0.0;
  double s_max_    = 0.0;
  double t_anchor_ = 0.0;
  double s_anchor_ = 0.0;
  std::string name_ = "CircuitGenerator";
};

}  // namespace mpc_examples::adapters

#endif  // MPC_EXAMPLES_ADAPTERS_CIRCUIT_GENERATOR_HPP_
