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
 * @file controller_base.hpp
 *
 * Abstract controller interface consumed by WaypointsSimulator.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_FRAMEWORK_CONTROLLER_BASE_HPP_
#define MPC_EXAMPLES_FRAMEWORK_CONTROLLER_BASE_HPP_

#include "framework/types.hpp"
#include "mav_model/datatypes/state.hpp"

#include <string>
#include <vector>

namespace mpc_examples {
struct ExampleConfig;
}  // namespace mpc_examples

namespace mpc_examples::framework {

/**
 * @brief Abstract base for outer-loop controllers (PID cascade, MPC, ...).
 *
 * The WaypointsSimulator queries the horizon shape (N+1 samples at
 * referenceHorizonDt) and calls computeCommand() once per control period.
 * Ground-truth state is provided by the simulator; references are sampled
 * from an ITrajectoryGenerator at times [t, t+dt, ..., t+N*dt].
 *
 * Timing model:
 *   - Outer control loop runs at controlPeriod().
 *   - MPC controllers return N_horizon+1 samples; PID returns 1.
 *
 * Units:
 *   - Position [m], linear velocity [m/s], acceleration [m/s^2], yaw [rad].
 *   - Thrust [N] (collective), body rates [rad/s] (body frame).
 *
 * Frames:
 *   - Position/velocity/acceleration references: world frame.
 *   - Body rates in the returned ControlCommand: body frame.
 */
class IController {
public:
  virtual ~IController() = default;

  /**
   * @brief Configure the controller before the simulation starts.
   *
   * Called once by WaypointsSimulator after arming the simulator.
   *
   * @param initial_state Current ground-truth state at t=0.
   * @param example_cfg   Shared example configuration (max_speed, dts, ...).
   */
  virtual void initialize(const mav_model::State& initial_state,
                          const ExampleConfig& example_cfg) = 0;

  /** @return Number of reference samples required per computeCommand() call. */
  virtual int referenceHorizonSize() const = 0;

  /** @return Time step between consecutive reference samples [s]. */
  virtual double referenceHorizonDt() const = 0;

  /** @return Outer control period [s] (e.g. mpc_dt or pid_dt). */
  virtual double controlPeriod() const = 0;

  /**
   * @brief Compute the control command for the current step.
   *
   * @param state      Current ground-truth state.
   * @param references Reference samples sized referenceHorizonSize().
   *                   Fields not produced by the generator are zeroed.
   * @return Thrust + angular rate command.
   */
  virtual ControlCommand computeCommand(const mav_model::State& state,
                                        const std::vector<ReferenceSample>& references) = 0;

  /** @return Bitmask of ReferenceField values this controller requires. */
  virtual ReferenceFieldMask requiredReferenceFields() const = 0;

  /** @return Human-readable controller name (used in warnings, logs). */
  virtual const std::string& name() const = 0;

  /** @return Wall-clock time of the last computeCommand() call [microseconds]. */
  virtual double lastSolveTimeMicros() const = 0;
};

}  // namespace mpc_examples::framework

#endif  // MPC_EXAMPLES_FRAMEWORK_CONTROLLER_BASE_HPP_
