// Copyright 2025 Universidad Politécnica de Madrid
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the Universidad Politécnica de Madrid nor the
//      names of its contributors may be used to endorse or promote products
//      derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

/**
 * @file trajectory_generator_base.hpp
 *
 * Abstract reference generator interface (point-to-point with replan-on-change).
 *
 * Waypoint ownership lives in the scheduler/simulator layer, not in the
 * generator. The generator only sees the "next waypoint" through
 * onWaypointChanged(), which is called exactly once per waypoint transition
 * (replan boundary). Between transitions, update() and evaluate() query the
 * already-computed trajectory and must not re-plan.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_FRAMEWORK_TRAJECTORY_GENERATOR_BASE_HPP_
#define MPC_EXAMPLES_FRAMEWORK_TRAJECTORY_GENERATOR_BASE_HPP_

#include <Eigen/Dense>

#include <string>

#include "framework/types.hpp"
#include "mav_model/datatypes/state.hpp"

namespace mpc_examples {
struct ExampleConfig;
}  // namespace mpc_examples

namespace mpc_examples::framework {

/**
 * @brief Abstract base for point-to-point reference generators.
 *
 * Lifecycle:
 *   1. initialize(initial_state, example_cfg) — at t=0, before the first tick.
 *   2. onWaypointChanged(next_wp, state, t_start) — driven by the scheduler
 *      whenever a new target becomes active. This is the only place where
 *      expensive replanning is allowed.
 *   3. update(t, state) — called once per outer control step. Should be
 *      cheap: only query-side bookkeeping, no replan.
 *   4. evaluate(t) — sample the currently-planned trajectory. Called multiple
 *      times per outer step to fill the controller's prediction horizon.
 *
 * Units/frames:
 *   - Waypoints and reference fields in the world frame, SI units.
 *   - Yaw in rad; path-facing logic is internal to each generator.
 */
class ITrajectoryGenerator {
public:
  virtual ~ITrajectoryGenerator() = default;

  /**
   * @brief Configure the generator once at the start of the mission.
   *
   * Implementations typically record initial pose, read tunables from
   * example_cfg (max_speed, path_facing, dts) and reset their internal
   * trajectory buffers. Waypoints are not passed here — they arrive via
   * onWaypointChanged() as the scheduler advances through the mission.
   *
   * @param initial_state Ground-truth state at t=0.
   * @param example_cfg   Shared example configuration.
   */
  virtual void initialize(const mav_model::State& initial_state,
                          const ExampleConfig& example_cfg) = 0;

  /**
   * @brief Plan a trajectory from the current state to @p next_waypoint.
   *
   * Called once per waypoint transition (including the very first one when
   * the scheduler activates waypoint 0). This is where batch generators
   * (Gcopter, dynamic) should run their solvers. Simple setpoint-style
   * generators may just record the new target.
   *
   * @param next_waypoint Target position in world frame [m].
   * @param state         Current ground-truth state (starting condition for the plan).
   * @param t_start       Simulator time at which the new segment begins [s].
   */
  virtual void onWaypointChanged(const Eigen::Vector3d& next_waypoint,
                                 const mav_model::State& state,
                                 double t_start) = 0;

  /**
   * @brief Lightweight per-step hook. Must not replan.
   *
   * Generators that need to track drone state across steps (e.g. for the
   * path-facing yaw heuristic) can update their internal cache here.
   *
   * @param t     Current simulation time [s].
   * @param state Current ground-truth state.
   */
  virtual void update(double t, const mav_model::State& state) = 0;

  /**
   * @brief Evaluate the currently-planned trajectory at absolute time @p t.
   *
   * @param t Time [s]. Callers sample the horizon at [t, t+dt, ..., t+N*dt].
   * @return  Reference sample; unset fields must be zero.
   */
  virtual ReferenceSample evaluate(double t) const = 0;

  /** @return Bitmask of ReferenceField values produced by this generator. */
  virtual ReferenceFieldMask providedReferenceFields() const = 0;

  /** @return Human-readable generator name (used in warnings, logs). */
  virtual const std::string& name() const = 0;
};

}  // namespace mpc_examples::framework

#endif  // MPC_EXAMPLES_FRAMEWORK_TRAJECTORY_GENERATOR_BASE_HPP_
