// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file waypoint_scheduler.hpp
 *
 * Advances the active waypoint on a time basis so every example receives
 * identical references at identical simulator times (fair controller ×
 * generator comparison).
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_FRAMEWORK_WAYPOINT_SCHEDULER_HPP_
#define MPC_EXAMPLES_FRAMEWORK_WAYPOINT_SCHEDULER_HPP_

#include <Eigen/Dense>

#include <cstddef>
#include <vector>

namespace mpc_examples::framework {

/**
 * @brief Time-based waypoint advancement driven by a distance/velocity heuristic.
 *
 * Given a list of waypoints and the maximum allowed speed, precomputes a
 * switch time for each waypoint as
 *
 *     t_switch[i] = t_switch[i-1]
 *                 + ||wp[i] - wp[i-1]|| / (max_speed * scheduler_speed_factor)
 *                 + settle_margin_s
 *
 * where the hop from the initial position to wp[0] follows the same rule.
 * `scheduler_speed_factor` ∈ (0, 1] models the fact that smooth generators
 * (bell-shaped or trapezoidal) never sustain max_speed during the whole
 * segment; lowering the factor allocates more wall-clock time per hop so
 * the drone can settle before the next waypoint switch. A factor of 1.0
 * recovers the legacy distance/max_speed heuristic.
 *
 * Callers tick the scheduler with the current simulator time; the scheduler
 * reports whether the active waypoint has changed so the caller can notify
 * the trajectory generator (replan).
 */
class WaypointScheduler {
public:
  struct TickResult {
    bool waypoint_changed = false;  //!< True when the active waypoint index advanced.
    int active_index      = 0;      //!< Index of the waypoint the generator should track.
    bool finished         = false;  //!< True once every waypoint has been reached.
  };

  WaypointScheduler() = default;

  /**
   * @brief Prepare the scheduler for a run.
   *
   * @param waypoints              Mission waypoints (world frame, m). Must not be empty.
   * @param initial_position       Initial drone position (world frame, m). Used to
   *                               derive the time to reach the first waypoint.
   * @param max_speed              Maximum allowed cruise speed [m/s]. Must be > 0.
   * @param settle_margin_s        Extra time added to every waypoint hop [s]. >= 0.
   * @param scheduler_speed_factor Effective-speed factor in (0, 1]. The
   *                               heuristic uses max_speed * factor as the
   *                               expected average speed per segment.
   *                               Defaults to 1.0 (legacy behaviour).
   */
  void initialize(const std::vector<Eigen::Vector3d>& waypoints,
                  const Eigen::Vector3d& initial_position,
                  double max_speed,
                  double settle_margin_s,
                  double scheduler_speed_factor = 1.0);

  /**
   * @brief Query the scheduler at simulator time @p t.
   *
   * Returns @ref TickResult::waypoint_changed = true exactly once per
   * waypoint switch. Holds the final waypoint once all segments have
   * elapsed.
   */
  TickResult tick(double t);

  /** @return Simulator time at which waypoint @p i becomes active [s]. */
  double switchTime(std::size_t i) const { return switch_times_[i]; }

  /** @return Simulator time at which the last waypoint becomes active [s]. */
  double finalTime() const { return switch_times_.empty() ? 0.0 : switch_times_.back(); }

  /** @return Index of the currently active waypoint. */
  int activeIndex() const { return static_cast<int>(active_index_); }

  /** @return Number of waypoints in the mission. */
  std::size_t size() const { return waypoints_.size(); }

  /** @return Access the waypoint at index @p i. */
  const Eigen::Vector3d& waypoint(std::size_t i) const { return waypoints_[i]; }

  /** @return The switch-time schedule. */
  const std::vector<double>& switchTimes() const { return switch_times_; }

private:
  std::vector<Eigen::Vector3d> waypoints_;
  std::vector<double> switch_times_;  //!< Simulator time at which each waypoint activates.
  std::size_t active_index_ = 0;
};

}  // namespace mpc_examples::framework

#endif  // MPC_EXAMPLES_FRAMEWORK_WAYPOINT_SCHEDULER_HPP_
