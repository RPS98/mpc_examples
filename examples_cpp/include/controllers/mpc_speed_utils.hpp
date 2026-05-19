// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file mpc_speed_utils.hpp
 *
 * Helpers that derive the MPC reference speed from the YAML's soft `‖v‖²`
 * upper bound (`constraints.uh[0]`) and apply it back to the solver. Matches
 * the aerostack2 `as2_position_mpc_plugin` convention: the YAML `uh` encodes
 * `max_speed²` in m²/s², and `max_vel_percentage ∈ (0, 1]` is the safety
 * knob that caps `v_ref` below that bound.
 *
 * The helpers are inline so each controller's translation unit instantiates
 * them against its own `acados_mpc::MPC` (position and trajectory libs
 * compile separate copies of that type).
 */

#ifndef MPC_EXAMPLES_ADAPTERS_MPC_SPEED_UTILS_HPP_
#define MPC_EXAMPLES_ADAPTERS_MPC_SPEED_UTILS_HPP_

#include <cmath>
#include <stdexcept>
#include <string>

#include "acados_mpc/acados_mpc.hpp"

namespace mpc_examples::adapters::speed_utils {

/// Reads `constraints.uh[0]` from the YAML-configured MPC. Returns 0 when
/// the OCP has no `‖v‖²` constraint compiled in (Nh==0) — callers can use 0
/// as a sentinel for "the solver has no soft speed bound, skip the v_ref
/// derivation". Throws when Nh>0 but the value is non-positive (the
/// convention is that the YAML encodes `max_speed²` there, not a placeholder).
inline double readUhDefault(acados_mpc::MPC& mpc, const std::string& who) {
  if constexpr (acados_mpc::NonlinearConstraintBounds::Nh == 0) {
    (void)mpc;
    (void)who;
    return 0.0;
  } else {
    const auto uh = mpc.getNonlinearConstraintBounds()->getUhArray();
    if (uh[0] <= 0.0) {
      throw std::invalid_argument(
          who + ": constraints.uh[0] must be > 0 in the YAML (it encodes max_speed²). "
                "Got uh=" + std::to_string(uh[0]) + ".");
    }
    return uh[0];
  }
}

/// Computes `v_ref = sqrt(uh_default) * max_vel_percentage`. Validates the
/// percentage lies in `(0, 1]`. Returns 0 when `uh_default` is 0 (no
/// constraint available).
inline double deriveVRef(double uh_default, double max_vel_percentage,
                         const std::string& who) {
  if (max_vel_percentage <= 0.0 || max_vel_percentage > 1.0) {
    throw std::invalid_argument(who + ": max_vel_percentage must be in (0, 1].");
  }
  return std::sqrt(uh_default) * max_vel_percentage;
}

/// Sets the solver's runtime `uh = v_ref²` so the soft penalty matches the
/// reference speed used by the ramp / velocity feed-forward. No-op when the
/// OCP has no `‖v‖²` constraint.
inline void updateSpeedConstraint(acados_mpc::MPC& mpc, double v_ref) {
  constexpr std::size_t kNh = acados_mpc::NonlinearConstraintBounds::Nh;
  if constexpr (kNh > 0) {
    std::array<double, kNh> uh{};
    uh[0] = v_ref * v_ref;
    mpc.getNonlinearConstraintBounds()->setUh(uh);
    mpc.updateNonlinearConstraintBounds();
  } else {
    (void)mpc;
    (void)v_ref;
  }
}

}  // namespace mpc_examples::adapters::speed_utils

#endif  // MPC_EXAMPLES_ADAPTERS_MPC_SPEED_UTILS_HPP_
