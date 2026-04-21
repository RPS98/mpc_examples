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
 * Abstract reference generator interface consumed by WaypointsSimulator.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_FRAMEWORK_TRAJECTORY_GENERATOR_BASE_HPP_
#define MPC_EXAMPLES_FRAMEWORK_TRAJECTORY_GENERATOR_BASE_HPP_

#include <Eigen/Dense>

#include <string>
#include <vector>

#include "framework/types.hpp"
#include "mav_model/datatypes/state.hpp"

namespace mpc_examples {
struct ExampleConfig;
}  // namespace mpc_examples

namespace mpc_examples::framework {

/**
 * @brief Abstract base for reference generators (waypoints, jerk-limited, GCopter, ...).
 *
 * Adapts a waypoint list into a time-parameterised reference signal that the
 * controller can consume. evaluate(t) is called multiple times per outer step
 * to populate the controller's horizon.
 *
 * Implementations should leave the fields they do not produce at zero and
 * advertise their capabilities via providedReferenceFields(). For example, a
 * pure waypoint reference sets only position; a jerk-limited generator also
 * sets velocity and acceleration.
 *
 * Units/frames:
 *   - Waypoints and reference fields in the world frame, SI units.
 *   - Yaw in rad; path-facing logic is internal to each generator.
 */
class ITrajectoryGenerator {
public:
  virtual ~ITrajectoryGenerator() = default;

  /**
   * @brief Configure the generator before the simulation starts.
   *
   * @param waypoints     Mission waypoints in order (world frame, m).
   * @param initial_state Ground-truth state at t=0.
   * @param example_cfg   Shared example configuration (max_speed, path_facing, dts).
   */
  virtual void initialize(const std::vector<Eigen::Vector3d>& waypoints,
                          const mav_model::State& initial_state,
                          const ExampleConfig& example_cfg) = 0;

  /**
   * @brief Notify the generator of the current time and state.
   *
   * Called once per outer control step, before evaluate(). Generators that
   * replan online (e.g. waypoint advance, dynamic re-plan) use this hook.
   *
   * @param t     Current simulation time [s].
   * @param state Current ground-truth state.
   */
  virtual void update(double t, const mav_model::State& state) = 0;

  /**
   * @brief Evaluate the reference at an absolute time.
   *
   * @param t Time [s]. Callers sample the horizon at [t, t+dt, ..., t+N*dt].
   * @return  Reference sample; unset fields must be zero.
   */
  virtual ReferenceSample evaluate(double t) const = 0;

  /**
   * @brief Whether the mission is finished at time @p t.
   *
   * A true value triggers the hover phase in WaypointsSimulator.
   */
  virtual bool isFinished(double t) const = 0;

  /** @return Zero-based index of the waypoint currently being tracked. */
  virtual int currentWaypointIndex() const = 0;

  /** @return Bitmask of ReferenceField values produced by this generator. */
  virtual ReferenceFieldMask providedReferenceFields() const = 0;

  /** @return Human-readable generator name (used in warnings, logs). */
  virtual const std::string& name() const = 0;
};

}  // namespace mpc_examples::framework

#endif  // MPC_EXAMPLES_FRAMEWORK_TRAJECTORY_GENERATOR_BASE_HPP_
