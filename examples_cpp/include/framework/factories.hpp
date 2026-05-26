// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file factories.hpp
 *
 * String-keyed factories that map a run spec (controller name, generator
 * name, YAML path) to instances of IController / ITrajectoryGenerator.
 *
 * Used by the unified runner to avoid per-combination main() boilerplate.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_FRAMEWORK_FACTORIES_HPP_
#define MPC_EXAMPLES_FRAMEWORK_FACTORIES_HPP_

#include <memory>
#include <string>

#include "framework/controller_base.hpp"
#include "framework/trajectory_generator_base.hpp"

namespace mpc_examples::framework {

/** @brief Canonical controller keys accepted by makeController(). */
struct ControllerKeys {
  static constexpr const char* kPid           = "pid";
  static constexpr const char* kMpcPosition   = "mpc_position";
  static constexpr const char* kMpcTrajectory = "mpc_trajectory";
  static constexpr const char* kMpcc          = "mpcc";
};

/** @brief Canonical generator keys accepted by makeGenerator(). */
struct GeneratorKeys {
  static constexpr const char* kWaypoints   = "waypoints";
  static constexpr const char* kJerkLimited = "jerk_limited";
  static constexpr const char* kGcopter     = "gcopter";
  static constexpr const char* kDynamic     = "dynamic";
  static constexpr const char* kMavTrajGen  = "mav_traj_gen";
  static constexpr const char* kCircuit     = "circuit";
};

/**
 * @brief Build a controller from its string key and YAML path.
 *
 * @param name                 Controller key (see ControllerKeys). Case-sensitive.
 * @param config_path          Absolute or relative path to the controller YAML. If
 *                             empty, the factory falls back to the default path under
 *                             `configs/controllers/`.
 * @param is_trajectory_scope  For the PID controller, determines whether to use
 *                             PidTrajectoryGeometricController (true) or
 *                             PidPositionGeometricController (false). Ignored for
 *                             MPC controllers. Defaults to false.
 * @throws std::invalid_argument if @p name is not recognised.
 */
std::unique_ptr<IController> makeController(const std::string& name,
                                            const std::string& config_path,
                                            bool is_trajectory_scope = false);

/**
 * @brief Build a trajectory generator from its string key and YAML path.
 *
 * @param name        Generator key (see GeneratorKeys). Case-sensitive.
 * @param config_path Absolute or relative path to the generator YAML. If
 *                    empty, the factory falls back to the default path under
 *                    `configs/generators/`.
 * @throws std::invalid_argument if @p name is not recognised.
 */
std::unique_ptr<ITrajectoryGenerator> makeGenerator(const std::string& name,
                                                    const std::string& config_path);

/** @return Default YAML path (relative to repo root) for a controller name. */
std::string defaultControllerConfigPath(const std::string& name);

/** @return Default YAML path (relative to repo root) for a generator name. */
std::string defaultGeneratorConfigPath(const std::string& name);

}  // namespace mpc_examples::framework

#endif  // MPC_EXAMPLES_FRAMEWORK_FACTORIES_HPP_
