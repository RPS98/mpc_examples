// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file mission_loader.cpp
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "utils/mission_loader.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>

namespace mpc_examples::utils::mission_loader {

namespace {

constexpr double kVnEpsilon = 1.0e-6;

YAML::Node loadYaml(const std::string& path) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("mission_loader: file not found: " + path);
  }
  return YAML::LoadFile(path);
}

std::unordered_map<std::string, GatePose> parseGates(const YAML::Node& gates_root) {
  const YAML::Node node = gates_root["gates_poses"];
  if (!node || !node.IsMap()) {
    throw std::runtime_error("mission_loader: missing or invalid 'gates_poses' map.");
  }
  std::unordered_map<std::string, GatePose> out;
  out.reserve(node.size());
  for (auto it = node.begin(); it != node.end(); ++it) {
    const std::string name = it->first.as<std::string>();
    const auto vals        = it->second.as<std::vector<double>>();
    if (vals.size() != 4) {
      throw std::runtime_error("mission_loader: gate '" + name +
                               "' must be [x, y, z, yaw], got size " +
                               std::to_string(vals.size()));
    }
    out[name] = GatePose{vals[0], vals[1], vals[2], vals[3]};
  }
  return out;
}

MissionParams parseParams(const YAML::Node& mission_root) {
  MissionParams p;
  const YAML::Node node = mission_root["params"];
  if (!node || !node.IsMap()) {
    return p;
  }
  if (node["num_laps"]) p.num_laps = node["num_laps"].as<int>();
  if (node["ls"])       p.ls       = node["ls"].as<double>();
  if (node["av"])       p.av       = node["av"].as<double>();
  if (node["hs"])       p.hs       = node["hs"].as<double>();
  if (node["wo"])       p.wo       = node["wo"].as<double>();
  return p;
}

std::vector<Waypoint> parseWaypoints(const YAML::Node& section,
                                     const std::string& section_name) {
  std::vector<Waypoint> out;
  if (!section) {
    return out;
  }
  if (!section.IsSequence()) {
    throw std::runtime_error("mission_loader: '" + section_name + "' must be a sequence.");
  }
  out.reserve(section.size());
  for (std::size_t i = 0; i < section.size(); ++i) {
    const YAML::Node row = section[i];
    if (!row.IsSequence() || row.size() < 3) {
      throw std::runtime_error(section_name + "[" + std::to_string(i) +
                               "] must have at least 3 elements "
                               "[frame_id, modifiers, [x,y,z]].");
    }
    Waypoint wp;
    wp.frame_id  = row[0].as<std::string>();
    wp.modifiers = row[1].as<std::string>();

    const YAML::Node pos_node = row[2];
    if (!pos_node.IsSequence() || pos_node.size() != 3) {
      throw std::runtime_error(section_name + "[" + std::to_string(i) +
                               "]: position must be [x, y, z].");
    }
    wp.position = Eigen::Vector3d(pos_node[0].as<double>(),
                                  pos_node[1].as<double>(),
                                  pos_node[2].as<double>());

    if (row.size() >= 4 && row[3] && row[3].IsSequence() && row[3].size() == 3) {
      wp.velocity     = Eigen::Vector3d(row[3][0].as<double>(),
                                        row[3][1].as<double>(),
                                        row[3][2].as<double>());
      wp.has_velocity = true;
    }
    if (row.size() >= 5 && row[4] && row[4].IsSequence() && row[4].size() == 3) {
      wp.face_point     = Eigen::Vector3d(row[4][0].as<double>(),
                                          row[4][1].as<double>(),
                                          row[4][2].as<double>());
      wp.has_face_point = true;
    }
    out.push_back(std::move(wp));
  }
  return out;
}

/// Apply (translate, yaw-rotate-about-Z) transform from gate frame to world.
/// If @p translate is false, the gate translation is skipped (e.g. for
/// velocity/face_point vectors that are themselves directions).
Eigen::Vector3d gateToWorld(const GatePose& gate,
                            const Eigen::Vector3d& vec_local,
                            bool translate) {
  const double c  = std::cos(gate.yaw);
  const double s  = std::sin(gate.yaw);
  const double rx = c * vec_local.x() - s * vec_local.y();
  const double ry = s * vec_local.x() + c * vec_local.y();
  const double rz = vec_local.z();
  if (translate) {
    return {gate.x + rx, gate.y + ry, gate.z + rz};
  }
  return {rx, ry, rz};
}

/// Resolve a single waypoint's pose/velocity/face_point to world frame.
Waypoint resolveToWorld(const Waypoint& wp,
                        const std::unordered_map<std::string, GatePose>& gates) {
  Waypoint out;
  out.frame_id       = "map";
  out.modifiers      = wp.modifiers;
  out.has_velocity   = wp.has_velocity;
  out.has_face_point = wp.has_face_point;

  if (wp.frame_id == "drone0/map" || wp.frame_id == "map") {
    out.position   = wp.position;
    out.velocity   = wp.velocity;
    out.face_point = wp.face_point;
    return out;
  }
  const auto it = gates.find(wp.frame_id);
  if (it == gates.end()) {
    throw std::runtime_error("mission_loader: waypoint frame_id '" + wp.frame_id +
                             "' not present in gates_config.yaml.");
  }
  const GatePose& gate = it->second;
  out.position         = gateToWorld(gate, wp.position, /*translate=*/true);
  out.velocity         = wp.has_velocity
                             ? gateToWorld(gate, wp.velocity, /*translate=*/false)
                             : Eigen::Vector3d::Zero();
  out.face_point       = wp.has_face_point
                             ? gateToWorld(gate, wp.face_point, /*translate=*/true)
                             : Eigen::Vector3d::Zero();
  return out;
}

/// Concatenate ``fly`` ``laps`` times, dropping consecutive duplicates.
/// Mirrors mission_loader.py::_expand_laps so the C++ and Python paths
/// produce the same list when downstream consumers (circuit_generator,
/// trajectory MPC) replay the lap N>=2 times — without dedup the closing
/// waypoint of lap N collides with the opening waypoint of lap N+1 and
/// the polynomial trajectory generator asserts segment_time > 0.
std::vector<Waypoint> expandLaps(const std::vector<Waypoint>& fly, int num_laps) {
  if (fly.empty()) {
    return {};
  }
  const int laps = std::max(num_laps, 1);
  std::vector<Waypoint> expanded;
  expanded.reserve(static_cast<std::size_t>(laps) * fly.size());
  constexpr double kDuplicateToleranceM = 1.0e-3;
  for (int lap = 0; lap < laps; ++lap) {
    for (const auto& wp : fly) {
      if (!expanded.empty() &&
          (wp.position - expanded.back().position).cwiseAbs().maxCoeff() <=
              kDuplicateToleranceM) {
        continue;
      }
      expanded.push_back(wp);
    }
  }
  return expanded;
}

/// Resolve `vt`/`vn`, then apply `hs`/`av`/`ls` speed scaling. Mutates @p wps.
/// Must run AFTER positions have been moved to world frame.
void applyVelocityModifiers(std::vector<Waypoint>& wps,
                            const std::vector<Waypoint>& raw_section,
                            const MissionParams& params,
                            const std::unordered_map<std::string, GatePose>& gates) {
  const std::size_t n = wps.size();
  for (std::size_t i = 0; i < n; ++i) {
    if (!wps[i].has_velocity) {
      if (wps[i].modifiers.find("vt") != std::string::npos) {
        const std::string& raw_frame = raw_section[i].frame_id;
        const auto it                = gates.find(raw_frame);
        const Eigen::Vector3d unit_x(1.0, 0.0, 0.0);
        wps[i].velocity =
            (it != gates.end()) ? gateToWorld(it->second, unit_x, /*translate=*/false) : unit_x;
      } else if (wps[i].modifiers.find("vn") != std::string::npos && i + 1 < n) {
        Eigen::Vector3d direction = wps[i + 1].position - wps[i].position;
        if (direction.norm() < kVnEpsilon) {
          throw std::runtime_error("mission_loader: waypoint " + std::to_string(i) +
                                   " shares position with next; cannot resolve 'vn'.");
        }
        wps[i].velocity = direction;
      }
    }

    double speed = params.av;
    if (wps[i].modifiers.find("hs") != std::string::npos) {
      speed = params.hs;
    } else if (wps[i].modifiers.find("ls") != std::string::npos) {
      speed = params.ls;
    }

    const double nrm = wps[i].velocity.norm();
    if (nrm > kVnEpsilon) {
      wps[i].velocity = wps[i].velocity / nrm * speed;
    }
  }
}

}  // namespace

MissionData load(const std::string& mission_yaml, const std::string& gates_yaml) {
  const YAML::Node gates_root   = loadYaml(gates_yaml);
  const YAML::Node mission_root = loadYaml(mission_yaml);

  MissionData out;
  out.gates  = parseGates(gates_root);
  out.params = parseParams(mission_root);

  const auto takeoff_raw =
      parseWaypoints(mission_root["takeoff_waypoints"], "takeoff_waypoints");
  const auto fly_raw = parseWaypoints(mission_root["fly_waypoints"], "fly_waypoints");

  out.takeoff.reserve(takeoff_raw.size());
  for (const auto& wp : takeoff_raw) {
    out.takeoff.push_back(resolveToWorld(wp, out.gates));
  }
  out.fly.reserve(fly_raw.size());
  for (const auto& wp : fly_raw) {
    out.fly.push_back(resolveToWorld(wp, out.gates));
  }

  // `wo` vertical offset applied before velocity resolution (matches Python).
  for (auto* section : {&out.takeoff, &out.fly}) {
    for (auto& wp : *section) {
      if (wp.modifiers.find("wo") != std::string::npos) {
        wp.position.z() += out.params.wo;
      }
    }
  }

  applyVelocityModifiers(out.takeoff, takeoff_raw, out.params, out.gates);
  applyVelocityModifiers(out.fly, fly_raw, out.params, out.gates);

  // Replicate the lap by params.num_laps and drop consecutive duplicates
  // so the closing waypoint of lap N != opening waypoint of lap N+1.
  out.fly = expandLaps(out.fly, out.params.num_laps);
  return out;
}

std::pair<std::string, std::string> defaultPaths() {
  // mission_loader.cpp lives at
  //   <repo_root>/examples_cpp/src/utils/mission_loader.cpp
  // so parents[3] resolves to the mav_examples repo root.
  const std::filesystem::path self(__FILE__);
  const std::filesystem::path base =
      self.parent_path().parent_path().parent_path().parent_path() / "configs" / "missions";
  return {(base / "demo_circuit.yaml").string(), (base / "demo_gates.yaml").string()};
}

}  // namespace mpc_examples::utils::mission_loader
