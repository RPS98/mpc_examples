// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file factories_circuit_mpc.cpp
 *
 * Factory TU linked only into the ``circuit_examples_mpc`` executable
 * (MPC-trajectory scope). Dispatches mpc_trajectory + circuit. MPCC and
 * MPC-position are intentionally excluded so the legacy acados_mpc
 * library does not collide on the ``acados_mpc::`` symbols with
 * mpcc_acados or the position variant. The mpcc case lives in the
 * sibling `factories_circuit.cpp` → `circuit_examples` binary.
 */

#include "framework/factories.hpp"

#include <stdexcept>

#include "controllers/mpc_trajectory_controller_factory.hpp"
#include "generators/circuit_generator.hpp"

namespace mpc_examples::framework {

std::string defaultControllerConfigPath(const std::string& name) {
  if (name == ControllerKeys::kMpcTrajectory)
    return "configs/controllers/config_mpc_trajectory.yaml";
  throw std::invalid_argument("factories_circuit_mpc: unsupported controller '" + name +
                              "' (this binary only exposes 'mpc_trajectory'; "
                              "use circuit_examples for mpcc).");
}

std::string defaultGeneratorConfigPath(const std::string& name) {
  if (name == GeneratorKeys::kCircuit) return "configs/generators/config_circuit.yaml";
  throw std::invalid_argument("factories_circuit_mpc: unsupported generator '" + name + "'.");
}

std::unique_ptr<IController> makeController(const std::string& name,
                                            const std::string& config_path,
                                            bool /*is_trajectory_scope*/) {
  const std::string path = config_path.empty() ? defaultControllerConfigPath(name) : config_path;
  if (name == ControllerKeys::kMpcTrajectory) {
    return adapters::makeMpcTrajectoryController(path);
  }
  throw std::invalid_argument("factories_circuit_mpc: unsupported controller '" + name + "'.");
}

std::unique_ptr<ITrajectoryGenerator> makeGenerator(const std::string& name,
                                                    const std::string& config_path) {
  const std::string path = config_path.empty() ? defaultGeneratorConfigPath(name) : config_path;
  if (name == GeneratorKeys::kCircuit) {
    auto cfg = adapters::CircuitGenerator::loadConfigFromYaml(path);
    return std::make_unique<adapters::CircuitGenerator>(cfg);
  }
  throw std::invalid_argument("factories_circuit_mpc: unsupported generator '" + name + "'.");
}

}  // namespace mpc_examples::framework
