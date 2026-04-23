// Copyright 2025 mav_trajectory_generation_lib contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.

#ifndef MAV_TRAJECTORY_GENERATION_CPP_TRAJECTORY_GENERATOR_HPP_
#define MAV_TRAJECTORY_GENERATION_CPP_TRAJECTORY_GENERATOR_HPP_

#include <Eigen/Dense>

#include <memory>
#include <vector>

#include "mav_trajectory_generation_cpp/types.hpp"  // Waypoint, EndWaypoint, ...

namespace mav_trajectory_generation_cpp {

/**
 * @brief Pure-C++ facade around ETH-ASL `mav_trajectory_generation`.
 *
 * Generates smooth polynomial trajectories through a sequence of 3D
 * `Waypoint`s and exposes continuous evaluation of position, velocity and
 * acceleration.
 *
 * Each `Waypoint` carries an optional velocity and acceleration. When
 * present, those are enforced as hard constraints at that vertex. When
 * absent, the optimiser is free to pick their values. Use `EndWaypoint`
 * at the first and last entries to pin velocity and acceleration to zero
 * (the typical "start / end at rest" pattern).
 *
 * The generator is reusable: call `generate()` multiple times on the same
 * instance; each call resets internal state and replaces the trajectory.
 *
 * Implementation uses the Pimpl idiom so that this header does not leak
 * upstream includes (and transitively glog / nlopt) into consumers.
 */
class TrajectoryGenerator {
public:
  explicit TrajectoryGenerator(const OptimizationConfig& cfg = OptimizationConfig{});
  ~TrajectoryGenerator();

  TrajectoryGenerator(TrajectoryGenerator&&) noexcept;
  TrajectoryGenerator& operator=(TrajectoryGenerator&&) noexcept;

  TrajectoryGenerator(const TrajectoryGenerator&)            = delete;
  TrajectoryGenerator& operator=(const TrajectoryGenerator&) = delete;

  /**
   * @brief Generate a 3D polynomial trajectory through @p waypoints.
   *
   * Constraint handling per vertex:
   *   - Position is always pinned.
   *   - Intermediate vertices: velocity/acceleration pinned only when the
   *     corresponding `std::optional` fields are set; otherwise free.
   *   - Endpoints (first / last): all derivatives up to
   *     `cfg.derivative_to_optimize` must be pinned for a well-posed problem.
   *     The caller's explicit `Waypoint::velocity` / `::acceleration` wins;
   *     any unset derivative defaults to zero.
   *
   * Convenience: `EndWaypoint` is a `Waypoint` with velocity = 0 and
   * acceleration = 0. At an endpoint vertex it is functionally equivalent
   * to a bare `Waypoint(pos)` (which also defaults to zero there). It exists
   * as an explicit-intent marker and to document boundary behaviour.
   *
   * @param waypoints  Sequence of `Waypoint`s (implicit-convertible from
   *                   `Eigen::Vector3d` for position-only usage). Must have
   *                   at least two entries.
   * @param max_speed  Cruise speed used for segment-time allocation
   *                   (`estimateSegmentTimes()`) and, for
   *                   `Solver::Nonlinear`, as the velocity-magnitude
   *                   upper bound. Must be strictly positive [m/s].
   * @return  true iff a valid trajectory was produced. On failure the
   *          object transitions to an invalid state (`isValid() == false`)
   *          and subsequent sampling returns zero.
   */
  bool generate(const std::vector<Waypoint>& waypoints, double max_speed);

  /// True iff the last `generate()` call produced a usable trajectory.
  bool isValid() const noexcept;

  /// Start time of the trajectory [s]. Always 0 when valid.
  double minTime() const;

  /// End time of the trajectory [s]. Sum of segment durations when valid.
  double maxTime() const;

  /// Convenience: `maxTime() - minTime()`.
  double duration() const { return maxTime() - minTime(); }

  /**
   * @brief Evaluate position/velocity/acceleration at time @p t.
   *
   * The input @p t is clamped to `[minTime(), maxTime()]`. If the generator
   * holds no valid trajectory the returned sample is zero-initialised.
   */
  TrajectorySample evaluate(double t) const;

  /**
   * @brief Evaluate the @p derivative_order -th derivative at time @p t.
   *
   * Supports any derivative order exposed by the upstream polynomial
   * (0 = POSITION, 1 = VELOCITY, 2 = ACCELERATION, 3 = JERK, 4 = SNAP).
   * The input @p t is clamped to `[minTime(), maxTime()]`.
   */
  Eigen::Vector3d evaluateDerivative(double t, int derivative_order) const;

  /**
   * @brief Snapshot of the current planned spline.
   *
   * The returned @ref Spline owns its polynomial coefficients and is
   * independent of the generator — it can be stored, serialised, or
   * fed into another generator via @ref setSpline(). Default-constructed
   * (empty `segments`) if `isValid() == false`.
   */
  Spline spline() const;

  /**
   * @brief Replace the current trajectory with an externally-provided spline.
   *
   * Useful for reusing a pre-computed / cached trajectory without running
   * the solver again (e.g. loaded from a CSV via @ref Spline::loadFromCsv).
   * Validates each segment's coefficient shape against the facade's fixed
   * `(dimension=3, polynomial_order=10)`. On failure the generator
   * transitions to an invalid state (`isValid() == false`).
   *
   * Note: `Spline::start_time` is retained in the input but the upstream
   * `mav_trajectory_generation::Trajectory` always exposes `minTime() == 0`,
   * so after `setSpline()` the generator's `minTime()` is `0` regardless
   * of `spline.start_time`.
   *
   * @return true iff the spline was accepted.
   */
  bool setSpline(const Spline& spline);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mav_trajectory_generation_cpp

#endif  // MAV_TRAJECTORY_GENERATION_CPP_TRAJECTORY_GENERATOR_HPP_
