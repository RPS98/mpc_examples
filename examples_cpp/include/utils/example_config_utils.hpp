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

/**
 * @brief Whether the controller/generator delay applied to the simulator is
 *        taken from the measured wall-clock per iteration or from a fixed
 *        value declared in YAML.
 *
 * @see ExampleConfig::controller_delay_mode
 * @see ExampleConfig::generator_delay_mode
 */
enum class DelayMode {
  kMeasured,
  kFixed,
};

/**
 * @brief Spec of one controller/generator combination to run.
 *
 * The unified executable loops over `ExampleConfig::runs` and for each
 * enabled entry instantiates the matching adapters via the factories and
 * executes a single mission through `WaypointsSimulator`.
 *
 * @p controller_config and @p generator_config are filesystem paths relative
 * to the repo root (or absolute). When empty, the factory uses the default
 * path under `configs/controllers/` or `configs/generators/`.
 */
struct RunSpec {
  std::string controller;  ///< pid | mpc_position | mpc_trajectory
  std::string generator;   ///< waypoints | jerk_limited | gcopter | dynamic
  bool enabled = true;
  std::string controller_config;  ///< Optional override for the controller YAML path.
  std::string generator_config;   ///< Optional override for the generator YAML path.
};

struct ExampleConfig {
  double sim_time           = 0.0;
  double model_dt           = 0.0;
  double controller_dt      = 0.0;
  double mpc_dt             = 0.0;
  double pid_dt             = 0.0;
  double max_speed          = 0.0;
  double hover_time         = 0.0;
  double settle_margin_s    = 2.0;  ///< Margin added to each waypoint hop (s).
  bool path_facing          = true;
  std::string output_format = "mcap";  ///< Output format: "mcap" or "csv".
  bool benchmark            = false;   ///< Skip CSV logging for performance measurement.
  bool silent   = false;  ///< Suppress in-loop console output (progress bar, waypoint messages).
  bool parallel = false;  ///< Run all enabled cases concurrently (one worker per run).

  // Compute delay model applied by WaypointsSimulator when forwarding
  // references (generator) and commands (controller) to the high-frequency
  // physics/INDI loops. `kMeasured` uses the wall-clock time of the most
  // recent generator/controller step as the delay; `kFixed` ignores the
  // measurement and always applies the corresponding *_delay_fixed_s value.
  DelayMode controller_delay_mode = DelayMode::kMeasured;
  double controller_delay_fixed_s = 0.0;
  DelayMode generator_delay_mode  = DelayMode::kMeasured;
  double generator_delay_fixed_s  = 0.0;

  std::vector<Eigen::Vector3d> waypoints;
  std::vector<RunSpec> runs;
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

inline double readDoubleOptional(const YAML::Node& node,
                                 const std::string& path,
                                 const double default_value) {
  if (!node) {
    return default_value;
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

inline std::string readStringOptional(const YAML::Node& node,
                                      const std::string& path,
                                      const std::string& default_value) {
  if (!node) {
    return default_value;
  }
  try {
    return node.as<std::string>();
  } catch (const YAML::Exception&) {
    throw std::invalid_argument(path + " must be a string value.");
  }
}

inline DelayMode parseDelayMode(const std::string& value, const std::string& path) {
  if (value == "measured") {
    return DelayMode::kMeasured;
  }
  if (value == "fixed") {
    return DelayMode::kFixed;
  }
  throw std::invalid_argument(path + " must be 'measured' or 'fixed' (got '" + value + "').");
}

inline void validateDtDivisibility(const ExampleConfig& config) {
  if (config.sim_time <= 0.0) {
    throw std::invalid_argument("sim_config.sim_time must be greater than zero.");
  }
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

inline void loadRunsFromNode(const YAML::Node& runs_node, std::vector<RunSpec>& out) {
  if (!runs_node) {
    return;
  }
  if (!runs_node.IsSequence()) {
    throw std::invalid_argument("sim_config.runs must be a sequence of run specs.");
  }
  for (std::size_t i = 0; i < runs_node.size(); ++i) {
    const YAML::Node& item   = runs_node[i];
    const std::string prefix = "sim_config.runs[" + std::to_string(i) + "]";
    if (!item || !item.IsMap()) {
      throw std::invalid_argument(prefix + " must be a mapping.");
    }
    RunSpec spec;
    if (!item["controller"]) {
      throw std::invalid_argument(prefix + ".controller is required.");
    }
    if (!item["generator"]) {
      throw std::invalid_argument(prefix + ".generator is required.");
    }
    spec.controller = item["controller"].as<std::string>();
    spec.generator  = item["generator"].as<std::string>();
    spec.enabled    = readBoolOptional(item["enabled"], prefix + ".enabled", true);
    spec.controller_config =
        readStringOptional(item["controller_config"], prefix + ".controller_config", std::string());
    spec.generator_config =
        readStringOptional(item["generator_config"], prefix + ".generator_config", std::string());
    out.push_back(std::move(spec));
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
  config.mpc_dt     = detail::readDoubleRequired(sim["mpc_dt"], "sim_config.mpc_dt");
  config.pid_dt     = detail::readDoubleRequired(sim["pid_dt"], "sim_config.pid_dt");
  config.max_speed  = detail::readDoubleRequired(sim["max_speed"], "sim_config.max_speed");
  config.hover_time = detail::readDoubleRequired(sim["hover_time"], "sim_config.hover_time");
  config.settle_margin_s =
      detail::readDoubleOptional(sim["settle_margin_s"], "sim_config.settle_margin_s", 2.0);
  config.path_facing = detail::readBoolRequired(sim["path_facing"], "sim_config.path_facing");
  config.output_format =
      detail::readStringOptional(sim["output_format"], "sim_config.output_format", "mcap");
  config.benchmark = detail::readBoolOptional(sim["benchmark"], "sim_config.benchmark", false);
  config.silent    = detail::readBoolOptional(sim["silent"], "sim_config.silent", false);
  config.parallel  = detail::readBoolOptional(sim["parallel"], "sim_config.parallel", false);

  // Delay configuration -------------------------------------------------------
  const std::string ctrl_delay_str = detail::readStringOptional(
      sim["controller_delay_mode"], "sim_config.controller_delay_mode", "measured");
  config.controller_delay_mode =
      detail::parseDelayMode(ctrl_delay_str, "sim_config.controller_delay_mode");
  config.controller_delay_fixed_s = detail::readDoubleOptional(
      sim["controller_delay_fixed_s"], "sim_config.controller_delay_fixed_s", 0.0);

  const std::string gen_delay_str = detail::readStringOptional(
      sim["generator_delay_mode"], "sim_config.generator_delay_mode", "measured");
  config.generator_delay_mode =
      detail::parseDelayMode(gen_delay_str, "sim_config.generator_delay_mode");
  config.generator_delay_fixed_s = detail::readDoubleOptional(
      sim["generator_delay_fixed_s"], "sim_config.generator_delay_fixed_s", 0.0);

  // Waypoints -----------------------------------------------------------------
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

  // Runs ----------------------------------------------------------------------
  detail::loadRunsFromNode(sim["runs"], config.runs);

  detail::validateDtDivisibility(config);
  return config;
}

}  // namespace mpc_examples

#endif  // MPC_EXAMPLES_EXAMPLE_CONFIG_UTILS_HPP_
