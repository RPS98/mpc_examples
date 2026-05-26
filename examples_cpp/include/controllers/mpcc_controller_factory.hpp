// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file mpcc_controller_factory.hpp
 *
 * Header-only factory declaration. Hides the inclusion of
 * `mpcc_controller.hpp` (which transitively pulls
 * `mpcc_acados/acados_mpc.hpp` → `acados_mpc::MPC`) so that translation
 * units that ALSO need the legacy `acados_mpc/acados_mpc.hpp` MPC class
 * (e.g. `factories_circuit.cpp` which dispatches both MPCC and
 * mpc_trajectory) do not pick up two conflicting definitions of the
 * `acados_mpc::MPC` symbol.
 *
 * The implementation lives in mpcc_controller_factory.cpp, where the
 * mpcc-side header is included in isolation.
 */

#ifndef MPC_EXAMPLES_ADAPTERS_MPCC_CONTROLLER_FACTORY_HPP_
#define MPC_EXAMPLES_ADAPTERS_MPCC_CONTROLLER_FACTORY_HPP_

#include <memory>
#include <string>

#include "framework/controller_base.hpp"

namespace mpc_examples::adapters {

std::unique_ptr<framework::IController> makeMpccController(const std::string& config_path);

}  // namespace mpc_examples::adapters

#endif  // MPC_EXAMPLES_ADAPTERS_MPCC_CONTROLLER_FACTORY_HPP_
