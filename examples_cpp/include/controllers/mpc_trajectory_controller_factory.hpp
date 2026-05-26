// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file mpc_trajectory_controller_factory.hpp
 *
 * Header-only factory declaration that hides the inclusion of
 * `mpc_trajectory_controller.hpp` (which pulls
 * `acados_mpc/acados_mpc.hpp` → trajectory variant of `acados_mpc::MPC`).
 * Paired with `mpcc_controller_factory.hpp`: each factory's
 * implementation lives in an isolated TU so a single downstream TU can
 * dispatch to both without ODR collisions on `acados_mpc::MPC`.
 */

#ifndef MPC_EXAMPLES_ADAPTERS_MPC_TRAJECTORY_CONTROLLER_FACTORY_HPP_
#define MPC_EXAMPLES_ADAPTERS_MPC_TRAJECTORY_CONTROLLER_FACTORY_HPP_

#include <memory>
#include <string>

#include "framework/controller_base.hpp"

namespace mpc_examples::adapters {

std::unique_ptr<framework::IController> makeMpcTrajectoryController(
    const std::string& config_path);

}  // namespace mpc_examples::adapters

#endif  // MPC_EXAMPLES_ADAPTERS_MPC_TRAJECTORY_CONTROLLER_FACTORY_HPP_
