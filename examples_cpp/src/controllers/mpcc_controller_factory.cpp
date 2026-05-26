// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file mpcc_controller_factory.cpp
 *
 * Isolated TU for instantiating `MpccController` so the mpcc_acados
 * `acados_mpc::MPC` definition stays out of scope of factory TUs that
 * also see the legacy `acados_mpc/acados_mpc.hpp`. See
 * mpcc_controller_factory.hpp for the rationale.
 */

#include "controllers/mpcc_controller_factory.hpp"

#include "controllers/mpcc_controller.hpp"

namespace mpc_examples::adapters {

std::unique_ptr<framework::IController> makeMpccController(const std::string& config_path) {
  auto cfg = MpccController::loadConfigFromYaml(config_path);
  return std::make_unique<MpccController>(cfg);
}

}  // namespace mpc_examples::adapters
