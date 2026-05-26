// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file factories_circuit.cpp
 *
 * Factory TU linked only into the ``circuit_examples`` executable
 * (MPCC scope). Dispatches the MPCC controller and the closed-loop
 * circuit generator.
 *
 * MPC-trajectory and MPC-position are intentionally excluded so the
 * mpcc_acados library does not collide on the ``acados_mpc::`` symbols
 * with the legacy ``acados_mpc`` library (both expose
 * ``acados_mpc::MPC`` / ``acados_mpc::configureMpcFromYaml`` with
 * different bodies). The mpc_trajectory case lives in a sibling
 * `factories_circuit_mpc.cpp` linked into the separate
 * `circuit_examples_mpc` binary.
 */

#include "framework/factories.hpp"

#include <stdexcept>

#include "controllers/mpcc_controller_factory.hpp"
#include "generators/circuit_generator.hpp"

namespace mpc_examples::framework {

std::string defaultControllerConfigPath(const std::string& name) {
  if (name == ControllerKeys::kMpcc)
    return "configs/controllers/config_mpcc.yaml";
  throw std::invalid_argument("factories_circuit: unsupported controller '" + name +
                              "' (this binary only exposes 'mpcc'; "
                              "use circuit_examples_mpc for mpc_trajectory).");
}

std::string defaultGeneratorConfigPath(const std::string& name) {
  if (name == GeneratorKeys::kCircuit) return "configs/generators/config_circuit.yaml";
  throw std::invalid_argument("factories_circuit: unsupported generator '" + name + "'.");
}

std::unique_ptr<IController> makeController(const std::string& name,
                                            const std::string& config_path,
                                            bool /*is_trajectory_scope*/) {
  const std::string path = config_path.empty() ? defaultControllerConfigPath(name) : config_path;
  if (name == ControllerKeys::kMpcc) {
    return adapters::makeMpccController(path);
  }
  throw std::invalid_argument("factories_circuit: unsupported controller '" + name + "'.");
}

std::unique_ptr<ITrajectoryGenerator> makeGenerator(const std::string& name,
                                                    const std::string& config_path) {
  const std::string path = config_path.empty() ? defaultGeneratorConfigPath(name) : config_path;
  if (name == GeneratorKeys::kCircuit) {
    auto cfg = adapters::CircuitGenerator::loadConfigFromYaml(path);
    return std::make_unique<adapters::CircuitGenerator>(cfg);
  }
  throw std::invalid_argument("factories_circuit: unsupported generator '" + name + "'.");
}

}  // namespace mpc_examples::framework
