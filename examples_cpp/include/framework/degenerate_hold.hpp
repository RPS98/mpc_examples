// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file degenerate_hold.hpp
 * @brief Park-style fallback when a moving follow_reference target collapses
 *        onto the drone position.
 *
 * Mirrors aerostack2's `generate_polynomial_trajectory_behavior` degenerate
 * hold (constant `kDegenerateDistanceM` and `tryEnterDegenerateHold` /
 * `runDegenerateHold` in
 * `as2_behaviors_trajectory_generation/generate_polynomial_trajectory_behavior/
 * src/generate_polynomial_trajectory_behavior.cpp`).
 *
 * Semantics:
 *   - Enter the hold whenever the live single waypoint sits within
 *     `threshold_m` of the vehicle pose (the polynomial solver — LBFGS in
 *     gcopter, the jerk-limited integrator, the acados QP, ... — degenerates
 *     on near-zero displacement, so we skip planning altogether).
 *   - While the hold is active, publish a static horizon: every reference
 *     sample on the prediction window equals the latched target with
 *     velocity / acceleration set to zero and the yaw latched to the
 *     vehicle's current heading.
 *   - Leave the hold once the live target drifts back outside `threshold_m`;
 *     the caller is then expected to re-arm the generator with
 *     `onWaypointChanged(target, state, t)`.
 */

#ifndef MPC_EXAMPLES_FRAMEWORK_DEGENERATE_HOLD_HPP_
#define MPC_EXAMPLES_FRAMEWORK_DEGENERATE_HOLD_HPP_

#include <Eigen/Dense>

#include <cstddef>
#include <vector>

#include "framework/types.hpp"

namespace mpc_examples::framework {

/// Aerostack2 default constant for the park threshold (see
/// `generate_polynomial_trajectory_base.hpp:125`). Kept here for reference;
/// the runtime value is read from `ExampleConfig::degenerate_distance_m` so
/// both backends pick it up from the same project YAML.
inline constexpr double kDefaultDegenerateDistanceM = 0.05;

/// Return true when the live target sits within `threshold_m` of the
/// vehicle position. A non-positive `threshold_m` disables the gate.
inline bool isDegenerateTarget(const Eigen::Vector3d& target,
                               const Eigen::Vector3d& vehicle_position,
                               const double threshold_m) noexcept {
  if (threshold_m <= 0.0) {
    return false;
  }
  return (target - vehicle_position).norm() < threshold_m;
}

/// Fill an existing prediction-horizon buffer with a static reference
/// (`position = target`, `velocity = 0`, `acceleration = 0`, `yaw = yaw_rad`).
///
/// @param target    Latched target position in the world frame.
/// @param yaw_rad   Latched vehicle yaw (rad) — held constant during the hold.
/// @param refs      Output buffer; resized only when its current size differs
///                  from `n_samples` to avoid reallocation per tick.
/// @param n_samples Number of samples expected by the controller.
inline void fillStaticHorizon(const Eigen::Vector3d& target,
                              const double yaw_rad,
                              std::vector<ReferenceSample>& refs,
                              const std::size_t n_samples) {
  if (refs.size() != n_samples) {
    refs.assign(n_samples, ReferenceSample{});
  }
  for (auto& sample : refs) {
    sample.position     = target;
    sample.velocity     = Eigen::Vector3d::Zero();
    sample.acceleration = Eigen::Vector3d::Zero();
    sample.yaw          = yaw_rad;
  }
}

}  // namespace mpc_examples::framework

#endif  // MPC_EXAMPLES_FRAMEWORK_DEGENERATE_HOLD_HPP_
