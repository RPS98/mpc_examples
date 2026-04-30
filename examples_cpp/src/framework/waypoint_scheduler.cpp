// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file waypoint_scheduler.cpp
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "framework/waypoint_scheduler.hpp"

#include <stdexcept>

namespace mpc_examples::framework {

void WaypointScheduler::initialize(const std::vector<Eigen::Vector3d>& waypoints,
                                   const Eigen::Vector3d& initial_position,
                                   const double max_speed,
                                   const double settle_margin_s,
                                   const double scheduler_speed_factor) {
  if (waypoints.empty()) {
    throw std::invalid_argument("WaypointScheduler: waypoints must not be empty.");
  }
  if (max_speed <= 0.0) {
    throw std::invalid_argument("WaypointScheduler: max_speed must be > 0.");
  }
  if (settle_margin_s < 0.0) {
    throw std::invalid_argument("WaypointScheduler: settle_margin_s must be >= 0.");
  }
  if (scheduler_speed_factor <= 0.0 || scheduler_speed_factor > 1.0) {
    throw std::invalid_argument("WaypointScheduler: scheduler_speed_factor must lie in (0, 1].");
  }

  waypoints_    = waypoints;
  active_index_ = 0;
  switch_times_.clear();
  switch_times_.reserve(waypoints.size());

  // First hop: from initial_position to waypoints[0].
  const double effective_speed = max_speed * scheduler_speed_factor;
  Eigen::Vector3d previous     = initial_position;
  double t_cumulative          = 0.0;
  for (std::size_t i = 0; i < waypoints.size(); ++i) {
    const double distance = (waypoints[i] - previous).norm();
    t_cumulative += distance / effective_speed + settle_margin_s;
    switch_times_.push_back(t_cumulative);
    previous = waypoints[i];
  }
}

WaypointScheduler::TickResult WaypointScheduler::tick(const double t) {
  TickResult result;
  if (waypoints_.empty()) {
    return result;
  }

  const std::size_t previous_index = active_index_;

  // Advance the active index while the switch time of the next waypoint has
  // already elapsed. This is monotonic (time cannot go backwards under normal
  // usage).
  while (active_index_ + 1 < waypoints_.size() && t + 1e-12 >= switch_times_[active_index_]) {
    ++active_index_;
  }

  result.active_index     = static_cast<int>(active_index_);
  result.waypoint_changed = (active_index_ != previous_index);
  result.finished = (active_index_ + 1 == waypoints_.size()) && (t + 1e-12 >= switch_times_.back());
  return result;
}

}  // namespace mpc_examples::framework
