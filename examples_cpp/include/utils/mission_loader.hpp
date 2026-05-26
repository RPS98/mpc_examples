// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file mission_loader.hpp
 *
 * Load a gate-based mission as a list of absolute waypoints. Single
 * source of truth (C++) for any race-circuit mission that follows the
 * per-gate-frame schema (closed loop of gate centres + per-row modifiers
 * that resolve to absolute world poses).
 *
 * Mirrors the Python module
 * ``examples_py/examples_py/utils/mission_loader.py``. Supported
 * modifiers: ``wo`` (vertical offset), ``vt`` (tangent velocity),
 * ``vn`` (velocity to next), ``ny`` (no-yaw flag), ``sy`` (custom face
 * point), ``hs``/``av``/``ls`` (speed selectors), ``gp``/``rt`` (passive,
 * carried through verbatim).
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_UTILS_MISSION_LOADER_HPP_
#define MPC_EXAMPLES_UTILS_MISSION_LOADER_HPP_

#include <Eigen/Dense>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mpc_examples::utils::mission_loader {

/// Pose of a gate in world frame: (x, y, z) [m] + yaw [rad] about world Z.
struct GatePose {
  double x   = 0.0;
  double y   = 0.0;
  double z   = 0.0;
  double yaw = 0.0;
};

/// `params:` block of the mission YAML.
struct MissionParams {
  int num_laps = 1;
  double ls    = 1.0;
  double av    = 1.0;
  double hs    = 1.0;
  double wo    = 0.0;
};

/// One waypoint, position/velocity/face_point expressed in WORLD frame.
struct Waypoint {
  std::string frame_id;
  std::string modifiers;
  Eigen::Vector3d position   = Eigen::Vector3d::Zero();
  Eigen::Vector3d velocity   = Eigen::Vector3d::Zero();
  Eigen::Vector3d face_point = Eigen::Vector3d::Zero();
  bool has_velocity          = false;
  bool has_face_point        = false;
};

/// All data parsed from mission.yaml + gates_config.yaml.
struct MissionData {
  std::unordered_map<std::string, GatePose> gates;
  MissionParams params;
  std::vector<Waypoint> takeoff;
  std::vector<Waypoint> fly;
};

/**
 * @brief Parse gates + mission YAMLs into a fully-resolved MissionData.
 *
 * After this call, every `Waypoint` in `takeoff` / `fly` has its
 * position, velocity and face_point already expressed in world frame
 * (gate-frame transforms + `wo` offset + `vt`/`vn` velocity resolution +
 * `hs`/`av`/`ls` speed scaling all applied in that order).
 *
 * @throws std::runtime_error if a required key is missing, a row is
 *         malformed, a `vn` modifier cannot resolve, or a waypoint
 *         references an unknown gate.
 */
MissionData load(const std::string& mission_yaml,
                 const std::string& gates_yaml);

/**
 * @brief Canonical path of the demo mission shipped with the repo.
 *
 * Returns (mission_yaml, gates_yaml) under
 * ``<repo_root>/configs/missions/demo_{circuit,gates}.yaml``. The repo
 * root is resolved relative to this translation unit so the function
 * works regardless of the caller's CWD.
 */
std::pair<std::string, std::string> defaultPaths();

}  // namespace mpc_examples::utils::mission_loader

#endif  // MPC_EXAMPLES_UTILS_MISSION_LOADER_HPP_
