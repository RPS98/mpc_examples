// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file factories_trajectory.cpp
 *
 * Factory translation unit linked only into the ``trajectory_examples``
 * executable. Dispatches PID and MPC-Trajectory controllers plus all five
 * trajectory generators. Intentionally does NOT reference the
 * ``MpcPositionController`` so ``acados_position_mpc`` is kept out of this
 * binary (both acados libs expose the same ``acados_mpc::MPC`` symbol and
 * would violate ODR if linked together).
 */

#include "framework/factories.hpp"

#include <stdexcept>

#include "controllers/mpc_trajectory_controller.hpp"
#include "controllers/pid_position_geometric_controller.hpp"
#include "controllers/pid_trajectory_geometric_controller.hpp"
#include "generators/dynamic_trajectory_generator.hpp"
#include "generators/gcopter_generator.hpp"
#include "generators/jerk_limited_generator.hpp"
#include "generators/mav_traj_gen_generator.hpp"
#include "generators/waypoint_reference_generator.hpp"

namespace mpc_examples::framework {

std::string defaultControllerConfigPath(const std::string& name) {
  if (name == ControllerKeys::kPid) return "configs/controllers/config_pid.yaml";
  if (name == ControllerKeys::kMpcTrajectory)
    return "configs/controllers/config_mpc_trajectory.yaml";
  throw std::invalid_argument("factories_trajectory: unsupported controller '" + name +
                              "' (trajectory_examples only links pid and mpc_trajectory).");
}

std::string defaultGeneratorConfigPath(const std::string& name) {
  if (name == GeneratorKeys::kWaypoints) return "configs/generators/config_waypoints.yaml";
  if (name == GeneratorKeys::kJerkLimited) return "configs/generators/config_jerk_limited.yaml";
  if (name == GeneratorKeys::kGcopter) return "configs/generators/config_gcopter.yaml";
  if (name == GeneratorKeys::kDynamic) return "configs/generators/config_dynamic.yaml";
  if (name == GeneratorKeys::kMavTrajGen) return "configs/generators/config_mav_traj_gen.yaml";
  throw std::invalid_argument("Unknown generator name: '" + name + "'.");
}

std::unique_ptr<IController> makeController(const std::string& name,
                                            const std::string& config_path,
                                            bool is_trajectory_scope) {
  const std::string path = config_path.empty() ? defaultControllerConfigPath(name) : config_path;

  if (name == ControllerKeys::kPid) {
    if (is_trajectory_scope) {
      auto cfg = adapters::PidTrajectoryGeometricController::loadConfigFromYaml(path);
      return std::make_unique<adapters::PidTrajectoryGeometricController>(cfg);
    } else {
      auto cfg = adapters::PidPositionGeometricController::loadConfigFromYaml(path);
      return std::make_unique<adapters::PidPositionGeometricController>(cfg);
    }
  }
  if (name == ControllerKeys::kMpcTrajectory) {
    auto cfg = adapters::MpcTrajectoryController::loadConfigFromYaml(path);
    return std::make_unique<adapters::MpcTrajectoryController>(cfg);
  }
  throw std::invalid_argument("factories_trajectory: unsupported controller '" + name + "'.");
}

std::unique_ptr<ITrajectoryGenerator> makeGenerator(const std::string& name,
                                                    const std::string& config_path) {
  const std::string path = config_path.empty() ? defaultGeneratorConfigPath(name) : config_path;

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
  if (name == GeneratorKeys::kMavTrajGen) {
    auto cfg = adapters::MavTrajGenGenerator::loadConfigFromYaml(path);
    return std::make_unique<adapters::MavTrajGenGenerator>(cfg);
  }
  throw std::invalid_argument("Unknown generator name: '" + name + "'.");
}

}  // namespace mpc_examples::framework
