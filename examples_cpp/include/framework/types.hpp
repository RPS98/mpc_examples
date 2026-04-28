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
 * @file types.hpp
 *
 * Common data types used across the integrated examples framework.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_FRAMEWORK_TYPES_HPP_
#define MPC_EXAMPLES_FRAMEWORK_TYPES_HPP_

#include <Eigen/Dense>

#include <cstdint>
#include <initializer_list>

namespace mpc_examples::framework {

/**
 * @brief Fields that a reference sample may carry.
 *
 * Trajectory generators advertise which fields they produce via
 * ITrajectoryGenerator::providedReferenceFields(); controllers advertise which
 * fields they need via IController::requiredReferenceFields(). The
 * WaypointsSimulator compares both masks at construction and warns if the
 * generator does not cover all required fields. Missing fields stay at zero
 * in the ReferenceSample; the controller may degrade accordingly.
 */
enum class ReferenceField : std::uint32_t {
  kPosition     = 1u << 0,
  kVelocity     = 1u << 1,
  kAcceleration = 1u << 2,
};

using ReferenceFieldMask = std::uint32_t;

inline constexpr ReferenceFieldMask makeMask(std::initializer_list<ReferenceField> fields) {
  ReferenceFieldMask mask = 0u;
  for (ReferenceField f : fields) {
    mask |= static_cast<ReferenceFieldMask>(f);
  }
  return mask;
}

inline constexpr bool hasField(ReferenceFieldMask mask, ReferenceField f) {
  return (mask & static_cast<ReferenceFieldMask>(f)) != 0u;
}

/**
 * @brief Reference sample at a single point in time along the prediction horizon.
 *
 * All fields are expressed in the world frame.
 *   - position     [m]
 *   - velocity     [m/s]   (0 when the generator only produces positions)
 *   - acceleration [m/s^2] (0 when the generator only produces positions)
 *   - yaw          [rad]   (absolute heading; path-facing handled by the generator)
 */
struct ReferenceSample {
  Eigen::Vector3d position     = Eigen::Vector3d::Zero();
  Eigen::Vector3d velocity     = Eigen::Vector3d::Zero();
  Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
  double yaw                   = 0.0;
};

/**
 * @brief Control command produced by a controller at the outer loop rate.
 *
 * The WaypointsSimulator forwards this command to mav_simulator::Simulator in
 * RATES mode.
 */
struct ControlCommand {
  double thrust_n              = 0.0;                      //!< Collective thrust [N]
  Eigen::Vector3d angular_rate = Eigen::Vector3d::Zero();  //!< Body rates [rad/s]
};

}  // namespace mpc_examples::framework

#endif  // MPC_EXAMPLES_FRAMEWORK_TYPES_HPP_
