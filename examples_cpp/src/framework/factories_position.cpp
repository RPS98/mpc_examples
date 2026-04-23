// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file factories_position.cpp
 *
 * Factory translation unit linked only into the ``position_examples``
 * executable. Dispatches PID and MPC-Position controllers plus all four
 * trajectory generators. Intentionally does NOT reference the
 * ``MpcTrajectoryController`` so ``acados_trajectory_mpc`` is kept out of
 * this binary (both acados libs expose the same ``acados_mpc::MPC`` symbol
 * and would violate ODR if linked together).
 */

#include "framework/factories.hpp"

#include <stdexcept>

#include "controllers/mpc_position_controller.hpp"
#include "controllers/pid_geometric_controller.hpp"
#include "generators/dynamic_trajectory_generator.hpp"
#include "generators/gcopter_generator.hpp"
#include "generators/jerk_limited_generator.hpp"
#include "generators/waypoint_reference_generator.hpp"

namespace mpc_examples::framework {

std::string defaultControllerConfigPath(const std::string& name) {
  if (name == ControllerKeys::kPid)         return "configs/controllers/config_pid.yaml";
  if (name == ControllerKeys::kMpcPosition) return "configs/controllers/config_mpc.yaml";
  throw std::invalid_argument(
      "factories_position: unsupported controller '" + name +
      "' (position_examples only links pid and mpc_position).");
}

std::string defaultGeneratorConfigPath(const std::string& name) {
  if (name == GeneratorKeys::kWaypoints)   return "configs/generators/config_waypoints.yaml";
  if (name == GeneratorKeys::kJerkLimited) return "configs/generators/config_jerk_limited.yaml";
  if (name == GeneratorKeys::kGcopter)     return "configs/generators/config_gcopter.yaml";
  if (name == GeneratorKeys::kDynamic)     return "configs/generators/config_dynamic.yaml";
  throw std::invalid_argument("Unknown generator name: '" + name + "'.");
}

std::unique_ptr<IController> makeController(const std::string& name,
                                            const std::string& config_path) {
  const std::string path =
      config_path.empty() ? defaultControllerConfigPath(name) : config_path;

  if (name == ControllerKeys::kPid) {
    auto cfg = adapters::PidGeometricController::loadConfigFromYaml(path);
    return std::make_unique<adapters::PidGeometricController>(cfg);
  }
  if (name == ControllerKeys::kMpcPosition) {
    auto cfg = adapters::MpcPositionController::loadConfigFromYaml(path);
    return std::make_unique<adapters::MpcPositionController>(cfg);
  }
  throw std::invalid_argument(
      "factories_position: unsupported controller '" + name + "'.");
}

std::unique_ptr<ITrajectoryGenerator> makeGenerator(const std::string& name,
                                                    const std::string& config_path) {
  const std::string path =
      config_path.empty() ? defaultGeneratorConfigPath(name) : config_path;

  if (name == GeneratorKeys::kWaypoints) {
    auto cfg = adapters::WaypointReferenceGenerator::loadConfigFromYaml(path);
    return std::make_unique<adapters::WaypointReferenceGenerator>(cfg);
  }
  if (name == GeneratorKeys::kJerkLimited) {
    auto cfg = adapters::JerkLimitedGenerator::loadConfigFromYaml(path);
    return std::make_unique<adapters::JerkLimitedGenerator>(cfg);
  }
  if (name == GeneratorKeys::kGcopter) {
    auto cfg = adapters::GcopterGenerator::loadConfigFromYaml(path);
    return std::make_unique<adapters::GcopterGenerator>(cfg);
  }
  if (name == GeneratorKeys::kDynamic) {
    auto cfg = adapters::DynamicTrajectoryGenerator::loadConfigFromYaml(path);
    return std::make_unique<adapters::DynamicTrajectoryGenerator>(cfg);
  }
  throw std::invalid_argument("Unknown generator name: '" + name + "'.");
}

}  // namespace mpc_examples::framework
