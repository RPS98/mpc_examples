// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file mpc_trajectory_controller_factory.cpp
 *
 * Isolated TU for instantiating `MpcTrajectoryController`. See
 * mpc_trajectory_controller_factory.hpp for the rationale.
 */

#include "controllers/mpc_trajectory_controller_factory.hpp"

#include "controllers/mpc_trajectory_controller.hpp"

namespace mpc_examples::adapters {

std::unique_ptr<framework::IController> makeMpcTrajectoryController(
    const std::string& config_path) {
  auto cfg = MpcTrajectoryController::loadConfigFromYaml(config_path);
  return std::make_unique<MpcTrajectoryController>(cfg);
}

}  // namespace mpc_examples::adapters
