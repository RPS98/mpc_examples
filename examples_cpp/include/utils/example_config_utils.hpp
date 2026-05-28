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
  /// Optional scope hint. ``"position"`` restricts the entry to
  /// ``run_position_examples``; ``"trajectory"`` restricts it to
  /// ``run_trajectory_examples``. Empty (default) means both binaries
  /// honour the legacy generator-based scope check. Use the explicit
  /// scope only when the same (controller, generator) pair is meant to
  /// run with **different** controller_config files in each binary
  /// (e.g. ``pid + gcopter`` runs cascade in the position scope and
  /// parallel in the trajectory scope — see the paper's moving_path
  /// configuration).
  std::string scope;
};

struct ExampleConfig {
  double sim_time        = 0.0;
  double model_dt        = 0.0;
  double controller_dt   = 0.0;
  double mpc_dt          = 0.0;
  double pid_dt          = 0.0;
  double max_speed       = 0.0;
  double hover_time      = 0.0;
  double settle_margin_s = 2.0;  ///< Margin added to each waypoint hop (s).
  /// Effective-speed factor used by WaypointScheduler when budgeting time
  /// per segment. The heuristic assumes the drone travels at
  /// max_speed * scheduler_speed_factor on average. Bell-shaped (gcopter,
  /// mav_traj_gen) and trapezoidal (jerk_limited) generators never
  /// sustain max_speed during the whole hop, so a factor < 1.0 buys the
  /// scheduler enough time for the drone to settle before the next
  /// waypoint switch. Must lie in (0, 1]. Default 1.0 keeps the legacy
  /// distance/max_speed heuristic.
  double scheduler_speed_factor = 1.0;
  /// Mission scheduling mode. "stepwise" (default) keeps the legacy
  /// behaviour: the WaypointScheduler waits ``settle_margin_s`` between
  /// hops, so the drone reaches a near-zero velocity at every waypoint.
  /// "continuous" overrides ``settle_margin_s`` to 0 and
  /// ``scheduler_speed_factor`` to 1, producing a chained flow through
  /// all waypoints with no idle hold — this mirrors aerostack2's
  /// ``mission_moving_path.py`` (single follow_reference goal vs.
  /// per-waypoint go_to). The flag also propagates to the
  /// ``experiment_active`` toggle so the active window covers the whole
  /// chain rather than starting at ``waypoint_index >= 1``.
  std::string mission_mode      = "stepwise";
  /// follow_reference emulation parameters (only used when
  /// ``mission_mode == "continuous"``). Mirror aerostack2's
  /// ``mission_moving_path.py`` + ``follow_reference_plugin_trajectory``
  /// chain: a target plan is pre-fitted with gcopter on all waypoints,
  /// then the local generator is replanned (``onWaypointChanged``) every
  /// ``target_modify_period_s`` seconds whenever the moving target has
  /// shifted more than ``target_modify_threshold_m`` from the last
  /// published goal.
  double target_modify_period_s    = 0.2;    ///< Replan at most 5 Hz (rate-limit gate).
  double target_modify_threshold_m = 1.0;    ///< Replan only when target shifted >1.0 m
                                             ///< from last published goal. Higher than the
                                             ///< aerostack2 plugin default (0.01 m) to avoid
                                             ///< LBFGS degeneracies — gcopter solver fails on
                                             ///< near-zero-distance start→end pairs. Tuned to
                                             ///< let mpc_trajectory × gcopter converge while
                                             ///< still producing a moving target reference.
  double target_start_delay_s      = 0.0;    ///< Seconds the target stays parked at waypoint[0].
  /// Park / degenerate-hold threshold (m). When the moving target sits
  /// closer than this to the vehicle, the local generator is bypassed and
  /// the controller receives a static reference horizon latched to the
  /// target. Mirrors aerostack2's `kDegenerateDistanceM` constant in
  /// `generate_polynomial_trajectory_base.hpp`. <= 0 disables the gate.
  double degenerate_distance_m  = 0.05;
  /// Synthetic takeoff height (m) prepended as the first waypoint of the
  /// mission. Mirrors aerostack2's `takeoff_behavior` so both backends
  /// record the climb-to-cruise phase in the mcap. Set to 0 to keep the
  /// legacy behaviour (drone starts at the first user waypoint).
  double takeoff_altitude_m     = 0.0;
  /// Append a synthetic landing waypoint at the end of the mission. The
  /// land target is (last_wp.x, last_wp.y, 0). Mirrors aerostack2's
  /// `land_behavior`.
  bool land_at_end              = false;
  bool path_facing              = true;
  /// Publication frequency of the mission pose reference topic
  /// (`/drone0/debug/mission/reference/pose`). Matches the rate
  /// aerostack2's mission scripts use: 10 Hz for the stepwise triangle
  /// mission (constant `REPUBLISH_RATE_HZ` in `mission.py`) and the
  /// mission.yaml `execution.broadcaster_rate_hz` for the continuous
  /// moving_path mission. Set to <= 0 to suppress the topic entirely.
  double mission_pose_ref_freq  = 10.0;
  std::string output_format     = "mcap";  ///< Output format: "mcap" or "csv".
  bool benchmark                = false;   ///< Skip CSV logging for performance measurement.
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

  /// Optional initial pose applied to the simulator before arming. When
  /// `has_initial_state` is false, the simulator keeps its default state
  /// (`(0, 0, 0)` with identity orientation). When true, the simulator is
  /// reset to `initial_position` + RPY-derived orientation. Translates the
  /// project-level `vehicle_initial_pose` into the framework so the closed
  /// loop matches the standalone acados examples and the MPCC adapter
  /// constructs its spline around the actual start pose.
  bool has_initial_state = false;
  std::array<double, 3> initial_position {0.0, 0.0, 0.0};
  std::array<double, 3> initial_rpy      {0.0, 0.0, 0.0};

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

/// Resolve a single waypoint coordinate that may be expressed as a literal
/// number or one of the symbolic tokens "D", "H", "-D", "-H".
///
/// `D` (horizontal distance) and `H` (vertical step) are configured
/// through `sim_config.evaluate.{distance,height}`. When neither token
/// applies and the YAML scalar is numeric, the value is returned as-is.
inline double resolveWaypointToken(const YAML::Node& node,
                                   const std::string& name,
                                   const double distance,
                                   const double height) {
  if (!node || !node.IsScalar()) {
    throw std::invalid_argument(name + " must be a scalar (number or token).");
  }
  // Try numeric first.
  try {
    return node.as<double>();
  } catch (const YAML::Exception&) {
    // fall through to token parsing
  }
  std::string s = node.as<std::string>();
  // Trim leading/trailing spaces.
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(0, 1);
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
  double sign = 1.0;
  if (!s.empty() && s.front() == '-') {
    sign = -1.0;
    s.erase(0, 1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(0, 1);
  }
  if (s == "D") return sign * distance;
  if (s == "H") return sign * height;
  throw std::invalid_argument(name + " has unsupported token '" + node.as<std::string>() +
                              "'; allowed values are D, H, -D, -H, or a numeric literal.");
}

/// Read a 3-element waypoint that may use D/H tokens. The takeoff offset
/// is added to the third (z) component so token "0" translates to
/// `takeoff_height` and "H" to `takeoff_height + height`. If both
/// `distance` and `height` are <= 0, the function reads the waypoint
/// as a plain numeric vector and ignores the takeoff offset (legacy
/// behaviour).
inline Eigen::Vector3d readWaypoint(const YAML::Node& node,
                                    const std::string& name,
                                    const double distance,
                                    const double height,
                                    const double takeoff_height) {
  if (!node || !node.IsSequence() || node.size() != 3) {
    throw std::invalid_argument(name + " must be a sequence with 3 elements.");
  }
  if (distance <= 0.0 && height <= 0.0) {
    // Legacy: numeric vector, no takeoff offset.
    Eigen::Vector3d v;
    for (int i = 0; i < 3; ++i) {
      try {
        v(i) = node[i].as<double>();
      } catch (const YAML::Exception&) {
        throw std::invalid_argument(name + " must contain numeric values.");
      }
    }
    return v;
  }
  const double x = resolveWaypointToken(node[0], name + "[x]", distance, height);
  const double y = resolveWaypointToken(node[1], name + "[y]", distance, height);
  const double z = resolveWaypointToken(node[2], name + "[z]", distance, height);
  return Eigen::Vector3d(x, y, takeoff_height + z);
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
    spec.scope = readStringOptional(item["scope"], prefix + ".scope", std::string());
    if (!spec.scope.empty() && spec.scope != "position" && spec.scope != "trajectory") {
      throw std::invalid_argument(prefix + ".scope must be 'position', 'trajectory' or empty (got '" +
                                  spec.scope + "').");
    }
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
  config.scheduler_speed_factor = detail::readDoubleOptional(
      sim["scheduler_speed_factor"], "sim_config.scheduler_speed_factor", 1.0);
  if (config.scheduler_speed_factor <= 0.0 || config.scheduler_speed_factor > 1.0) {
    throw std::runtime_error("sim_config.scheduler_speed_factor must lie in (0, 1] (got " +
                             std::to_string(config.scheduler_speed_factor) + ")");
  }
  config.mission_mode = detail::readStringOptional(
      sim["mission_mode"], "sim_config.mission_mode", "stepwise");
  if (config.mission_mode != "stepwise" && config.mission_mode != "continuous") {
    throw std::invalid_argument(
        "sim_config.mission_mode must be either 'stepwise' or 'continuous' (got '" +
        config.mission_mode + "').");
  }
  config.target_modify_period_s = detail::readDoubleOptional(
      sim["target_modify_period_s"], "sim_config.target_modify_period_s", 0.05);
  if (config.target_modify_period_s < 0.0) {
    throw std::runtime_error(
        "sim_config.target_modify_period_s must be >= 0 (got " +
        std::to_string(config.target_modify_period_s) + ")");
  }
  config.target_modify_threshold_m = detail::readDoubleOptional(
      sim["target_modify_threshold_m"], "sim_config.target_modify_threshold_m", 0.01);
  if (config.target_modify_threshold_m < 0.0) {
    throw std::runtime_error(
        "sim_config.target_modify_threshold_m must be >= 0 (got " +
        std::to_string(config.target_modify_threshold_m) + ")");
  }
  config.target_start_delay_s = detail::readDoubleOptional(
      sim["target_start_delay_s"], "sim_config.target_start_delay_s", 0.0);
  if (config.target_start_delay_s < 0.0) {
    throw std::runtime_error(
        "sim_config.target_start_delay_s must be >= 0 (got " +
        std::to_string(config.target_start_delay_s) + ")");
  }
  config.degenerate_distance_m = detail::readDoubleOptional(
      sim["degenerate_distance_m"], "sim_config.degenerate_distance_m", 0.05);
  config.takeoff_altitude_m = detail::readDoubleOptional(
      sim["takeoff_altitude_m"], "sim_config.takeoff_altitude_m", 0.0);
  if (config.takeoff_altitude_m < 0.0) {
    throw std::runtime_error(
        "sim_config.takeoff_altitude_m must be >= 0 (got " +
        std::to_string(config.takeoff_altitude_m) + ")");
  }
  config.land_at_end = detail::readBoolOptional(
      sim["land_at_end"], "sim_config.land_at_end", false);

  // Initial pose (optional): overrides the simulator default state when
  // present. RPY is XYZ-intrinsic, matches tf2 / vehicle_initial_pose.py.
  const YAML::Node initial_state_node = sim["initial_state"];
  if (initial_state_node && initial_state_node.IsMap()) {
    const auto pos = detail::readVector<3>(initial_state_node["position"],
                                           "sim_config.initial_state.position");
    const auto rpy = detail::readVector<3>(initial_state_node["rpy"],
                                           "sim_config.initial_state.rpy");
    config.has_initial_state = true;
    config.initial_position  = {pos(0), pos(1), pos(2)};
    config.initial_rpy       = {rpy(0), rpy(1), rpy(2)};
  }
  config.path_facing = detail::readBoolRequired(sim["path_facing"], "sim_config.path_facing");
  config.mission_pose_ref_freq = detail::readDoubleOptional(
      sim["mission_pose_ref_freq"], "sim_config.mission_pose_ref_freq", 10.0);
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
  // Optional `sim_config.evaluate.{distance,height}` enables symbolic
  // tokens "D"/"H"/"-D"/"-H" inside the waypoint list (and adds
  // `takeoff_height` to the z component). When the section is absent
  // the legacy behaviour (waypoints are absolute metres in world frame)
  // is preserved.
  double evaluate_distance = 0.0;
  double evaluate_height = 0.0;
  double takeoff_height = 0.0;
  const YAML::Node evaluate_node = sim["evaluate"];
  if (evaluate_node && evaluate_node.IsMap()) {
    evaluate_distance = detail::readDoubleRequired(
        evaluate_node["distance"], "sim_config.evaluate.distance");
    evaluate_height = detail::readDoubleOptional(
        evaluate_node["height"], "sim_config.evaluate.height", evaluate_distance);
    takeoff_height = detail::readDoubleOptional(
        sim["takeoff_height"], "sim_config.takeoff_height", 0.0);
    if (evaluate_distance <= 0.0 || evaluate_height <= 0.0) {
      throw std::invalid_argument(
          "sim_config.evaluate.distance and evaluate.height must be > 0.");
    }
  }

  const YAML::Node waypoints = sim["waypoints"];
  if (!waypoints || !waypoints.IsSequence()) {
    throw std::invalid_argument("sim_config.waypoints must be a list.");
  }

  config.waypoints.clear();
  for (std::size_t index = 0; index < waypoints.size(); ++index) {
    config.waypoints.push_back(detail::readWaypoint(
        waypoints[index], "sim_config.waypoints[" + std::to_string(index) + "]",
        evaluate_distance, evaluate_height, takeoff_height));
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
