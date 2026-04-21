// Copyright 2025 Universidad Politécnica de Madrid
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

/**
 * @file example_config_utils.hpp
 *
 * Common YAML utilities shared by all integrated controller examples.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_EXAMPLE_CONFIG_UTILS_HPP_
#define MPC_EXAMPLES_EXAMPLE_CONFIG_UTILS_HPP_

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mpc_examples {

struct ExampleConfig {
  double sim_time      = 0.0;
  double model_dt      = 0.0;
  double controller_dt = 0.0;
  double mpc_dt        = 0.0;
  double pid_dt        = 0.0;
  double max_speed     = 0.0;
  double hover_time    = 0.0;
  bool path_facing     = true;
  bool benchmark       = false;  ///< Skip CSV logging for performance measurement
  bool silent = false;  ///< Suppress in-loop console output (progress bar, waypoint messages)
  std::vector<Eigen::Vector3d> waypoints;
};

namespace detail {

inline bool fileExists(const std::string& path) {
  std::ifstream file(path.c_str());
  return file.good();
}

inline std::string normalizeOutputPath(const std::string& output_path) {
  std::filesystem::path path(output_path);
  if (!path.has_parent_path()) {
    path = std::filesystem::path("simulator_logs") / path;
  }
  return path.string();
}

template <int N>
Eigen::Matrix<double, N, 1> readVector(const YAML::Node& node, const std::string& name) {
  if (!node || !node.IsSequence() || static_cast<int>(node.size()) != N) {
    throw std::invalid_argument(name + " must be a sequence with " + std::to_string(N) +
                                " elements.");
  }

  Eigen::Matrix<double, N, 1> value;
  for (int index = 0; index < N; ++index) {
    try {
      value(index) = node[index].as<double>();
    } catch (const YAML::Exception&) {
      throw std::invalid_argument(name + " must contain numeric values.");
    }
  }
  return value;
}

inline YAML::Node loadYamlRoot(const std::string& path) {
  if (!fileExists(path)) {
    throw std::invalid_argument("Config file not found at " +
                                std::filesystem::absolute(path).string() + ".");
  }

  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (const YAML::Exception& exception) {
    throw std::invalid_argument("Failed to parse YAML file '" + path + "': " + exception.what());
  }

  if (!root || !root.IsMap()) {
    throw std::invalid_argument("Root YAML node must be a mapping.");
  }
  return root;
}

inline double readDoubleRequired(const YAML::Node& node, const std::string& path) {
  if (!node) {
    throw std::invalid_argument("Missing required configuration key: " + path);
  }
  try {
    return node.as<double>();
  } catch (const YAML::Exception&) {
    throw std::invalid_argument(path + " must be a numeric value.");
  }
}

inline bool readBoolRequired(const YAML::Node& node, const std::string& path) {
  if (!node) {
    throw std::invalid_argument("Missing required configuration key: " + path);
  }
  try {
    return node.as<bool>();
  } catch (const YAML::Exception&) {
    throw std::invalid_argument(path + " must be a boolean value.");
  }
}

inline bool readBoolOptional(const YAML::Node& node,
                             const std::string& path,
                             const bool default_value) {
  if (!node) {
    return default_value;
  }
  try {
    return node.as<bool>();
  } catch (const YAML::Exception&) {
    throw std::invalid_argument(path + " must be a boolean value.");
  }
}

inline void validateDtDivisibility(const ExampleConfig& config) {
  if (config.model_dt <= 0.0) {
    throw std::invalid_argument("sim_config.model_dt must be greater than zero.");
  }
  if (config.controller_dt <= 0.0) {
    throw std::invalid_argument("sim_config.controller_dt must be greater than zero.");
  }
  if (config.mpc_dt <= 0.0) {
    throw std::invalid_argument("sim_config.mpc_dt must be greater than zero.");
  }
  if (config.pid_dt <= 0.0) {
    throw std::invalid_argument("sim_config.pid_dt must be greater than zero.");
  }

  const double model_to_controller = config.controller_dt / config.model_dt;
  if (model_to_controller < 1.0 ||
      std::abs(model_to_controller - std::round(model_to_controller)) > 1e-9) {
    throw std::invalid_argument(
        "sim_config.model_dt must divide sim_config.controller_dt exactly.");
  }

  const double controller_to_mpc = config.mpc_dt / config.controller_dt;
  if (controller_to_mpc < 1.0 ||
      std::abs(controller_to_mpc - std::round(controller_to_mpc)) > 1e-9) {
    throw std::invalid_argument("sim_config.controller_dt must divide sim_config.mpc_dt exactly.");
  }

  const double controller_to_pid = config.pid_dt / config.controller_dt;
  if (controller_to_pid < 1.0 ||
      std::abs(controller_to_pid - std::round(controller_to_pid)) > 1e-9) {
    throw std::invalid_argument("sim_config.controller_dt must divide sim_config.pid_dt exactly.");
  }
}

}  // namespace detail

inline ExampleConfig loadExampleConfig(const std::string& path) {
  const YAML::Node root = detail::loadYamlRoot(path);
  const YAML::Node sim  = root["sim_config"];
  if (!sim || !sim.IsMap()) {
    throw std::invalid_argument("sim_config must be a mapping.");
  }

  ExampleConfig config;
  config.sim_time = detail::readDoubleRequired(sim["sim_time"], "sim_config.sim_time");
  config.model_dt = detail::readDoubleRequired(sim["model_dt"], "sim_config.model_dt");
  config.controller_dt =
      detail::readDoubleRequired(sim["controller_dt"], "sim_config.controller_dt");
  config.mpc_dt      = detail::readDoubleRequired(sim["mpc_dt"], "sim_config.mpc_dt");
  config.pid_dt      = detail::readDoubleRequired(sim["pid_dt"], "sim_config.pid_dt");
  config.max_speed   = detail::readDoubleRequired(sim["max_speed"], "sim_config.max_speed");
  config.hover_time  = detail::readDoubleRequired(sim["hover_time"], "sim_config.hover_time");
  config.path_facing = detail::readBoolRequired(sim["path_facing"], "sim_config.path_facing");
  config.benchmark   = detail::readBoolOptional(sim["benchmark"], "sim_config.benchmark", false);
  config.silent      = detail::readBoolOptional(sim["silent"], "sim_config.silent", false);

  const YAML::Node waypoints = sim["waypoints"];
  if (!waypoints || !waypoints.IsSequence()) {
    throw std::invalid_argument("sim_config.waypoints must be a list.");
  }

  config.waypoints.clear();
  for (std::size_t index = 0; index < waypoints.size(); ++index) {
    config.waypoints.push_back(detail::readVector<3>(
        waypoints[index], "sim_config.waypoints[" + std::to_string(index) + "]"));
  }

  if (config.waypoints.empty()) {
    throw std::invalid_argument("sim_config.waypoints must contain at least one waypoint.");
  }

  detail::validateDtDivisibility(config);
  return config;
}

}  // namespace mpc_examples

#endif  // MPC_EXAMPLES_EXAMPLE_CONFIG_UTILS_HPP_
