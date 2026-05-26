// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file mpcc_controller.cpp
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "controllers/mpcc_controller.hpp"

#include <yaml-cpp/yaml.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "generators/circuit_generator.hpp"
#include "mpcc_acados/acados_mpc_datatype.hpp"
#include "mpcc_acados/acados_mpc_yaml.hpp"
#include "utils/example_config_utils.hpp"
#include "utils/mission_loader.hpp"

namespace mpc_examples::adapters {

namespace ml = mpc_examples::utils::mission_loader;

namespace {

constexpr int    kMpccSplineKnots   = 20;
constexpr double kSoftWeightDefault = 2.0;
constexpr double kSoftWeightOg      = 1.0;
constexpr double kSearchRadiusM     = 1.0;
constexpr double kDsCoarse          = 0.10;
constexpr double kDsFine            = 0.01;

std::string envOrEmpty(const char* name) {
  const char* v = std::getenv(name);
  return (v != nullptr) ? std::string(v) : std::string();
}

/// Auto-locate the acados OCP json shipped alongside the MPCC YAML.
std::string resolveOcpJson(const std::string& explicit_path,
                           const std::string& mpc_yaml_path) {
  if (!explicit_path.empty() && std::filesystem::exists(explicit_path)) {
    return explicit_path;
  }
  std::vector<std::string> candidate_dirs;
  if (const char* d = std::getenv("MPCC_ACADOS_EXAMPLE_DIR")) {
    candidate_dirs.emplace_back(d);
  }
  if (const char* root = std::getenv("MPCC_REPO_ROOT")) {
    candidate_dirs.emplace_back(std::string(root) +
                                "/workspace/thirdparty_libs/mpcc/mpcc_acados_example");
  }
  // Sibling of the mpc_yaml_path (upstream mpcc layout, two levels up).
  candidate_dirs.push_back(
      std::filesystem::absolute(mpc_yaml_path).parent_path().parent_path().string());
  for (const auto& cand_dir : candidate_dirs) {
    if (cand_dir.empty()) continue;
    const std::filesystem::path cand =
        std::filesystem::path(cand_dir) / "mpcc_interface" / "mpc_generated_code" /
        "acados_ocp.json";
    if (std::filesystem::exists(cand)) {
      return cand.string();
    }
  }
  throw std::invalid_argument(
      "MpccController: cannot locate acados ocp_json_file_path (tried explicit + "
      "MPCC_ACADOS_EXAMPLE_DIR + MPCC_REPO_ROOT + sibling of " + mpc_yaml_path + ").");
}

/// Coarse + fine search for the spline arc-length closest to @p drone_pos.
double findClosestS(const Eigen::Vector3d& drone_pos,
                    const spline::arc_length_reparam::ArcLengthReparametrization& window,
                    double s_min, double s_max) {
  double best_s = s_min;
  double min_d  = std::numeric_limits<double>::infinity();
  for (double s = s_min; s <= s_max; s += kDsCoarse) {
    const auto ev = spline::arc_length_reparam::evaluateArcLengthSpline(s, window);
    if (ev.position.size() < 3) continue;
    const double dx = ev.position[0] - drone_pos.x();
    const double dy = ev.position[1] - drone_pos.y();
    const double dz = ev.position[2] - drone_pos.z();
    const double d  = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (d < min_d) {
      min_d  = d;
      best_s = s;
    }
  }
  const double s_lo = std::max(s_min, best_s - kDsCoarse);
  const double s_hi = std::min(s_max, best_s + kDsCoarse);
  for (double s = s_lo; s <= s_hi; s += kDsFine) {
    const auto ev = spline::arc_length_reparam::evaluateArcLengthSpline(s, window);
    if (ev.position.size() < 3) continue;
    const double dx = ev.position[0] - drone_pos.x();
    const double dy = ev.position[1] - drone_pos.y();
    const double dz = ev.position[2] - drone_pos.z();
    const double d  = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (d < min_d) {
      min_d  = d;
      best_s = s;
    }
  }
  return best_s;
}

}  // namespace

MpccController::MpccController(const Config& cfg) : cfg_(cfg) {
  if (cfg_.mpc_yaml_path.empty()) {
    throw std::invalid_argument("MpccController: mpc_yaml_path must be provided.");
  }
  if (cfg_.desired_speed <= 0.0) {
    throw std::invalid_argument("MpccController: desired_speed must be > 0.");
  }
}

MpccController::Config MpccController::loadConfigFromYaml(const std::string& path) {
  const YAML::Node root = mpc_examples::detail::loadYamlRoot(path);
  if (!root["mpcc"]) {
    throw std::invalid_argument(
        "mpcc config " + path +
        ": missing top-level 'mpcc:' block. (If porting an old config, rename `mpc:` -> `mpcc:`.)");
  }
  Config cfg;
  cfg.mpc_yaml_path = path;

  if (const YAML::Node ctrl = root["controller"]; ctrl && ctrl.IsMap()) {
    if (ctrl["ocp_json_file_path"]) {
      cfg.ocp_json_file_path = ctrl["ocp_json_file_path"].as<std::string>();
    }
  }
  if (const YAML::Node circuit = root["circuit"]; circuit && circuit.IsMap()) {
    if (circuit["mission_yaml"]) cfg.mission_yaml = circuit["mission_yaml"].as<std::string>();
    if (circuit["gates_yaml"])   cfg.gates_yaml   = circuit["gates_yaml"].as<std::string>();
    if (circuit["desired_speed"]) {
      cfg.desired_speed = circuit["desired_speed"].as<double>();
    }
    if (circuit["origin_offset_m"]) {
      cfg.origin_offset_m = circuit["origin_offset_m"].as<double>();
    }
    if (circuit["closing_exit_margin_m"]) {
      cfg.closing_exit_margin_m = circuit["closing_exit_margin_m"].as<double>();
    }
    if (circuit["target_segment_length"]) {
      cfg.target_segment_length = circuit["target_segment_length"].as<double>();
    }
    if (circuit["samples_per_segment"]) {
      cfg.samples_per_segment = circuit["samples_per_segment"].as<int>();
    }
  }
  return cfg;
}

std::string MpccController::rekeyMpccToMpc(const std::string& source_path) {
  const YAML::Node root = YAML::LoadFile(source_path);
  if (!root.IsMap() || !root["mpcc"]) {
    return source_path;
  }
  // Build a copy with mpcc: renamed to mpc:.
  YAML::Node rekeyed;
  for (auto it = root.begin(); it != root.end(); ++it) {
    const std::string key = it->first.as<std::string>();
    rekeyed[(key == "mpcc") ? "mpc" : key] = it->second;
  }

  // Write the temp file under $TMPDIR (or /tmp). Use mkstemp for a
  // unique name; the upstream loader only reads it once at startup so
  // it is safe to leave for the OS to reap.
  std::filesystem::path tmpdir =
      std::filesystem::temp_directory_path();  // honours TMPDIR
  std::string templ =
      (tmpdir / "mpcc_rekey_XXXXXX.yaml").string();
  std::vector<char> buf(templ.begin(), templ.end());
  buf.push_back('\0');
  const int fd = mkstemps(buf.data(), 5);  // suffix length: ".yaml" == 5
  if (fd < 0) {
    throw std::runtime_error("MpccController::rekeyMpccToMpc: mkstemps failed.");
  }
  const std::string tmp_path(buf.data());
  ::close(fd);

  std::ofstream out(tmp_path);
  if (!out) {
    throw std::runtime_error("MpccController::rekeyMpccToMpc: cannot open " + tmp_path);
  }
  out << "# AUTO-GENERATED by MpccController::rekeyMpccToMpc — do not edit by hand.\n";
  out << rekeyed;
  out.close();
  return tmp_path;
}

void MpccController::initialize(const mav_model::State& initial_state,
                                const ExampleConfig& example_cfg) {
  if (example_cfg.mpc_dt <= 0.0) {
    throw std::invalid_argument("MpccController: example_cfg.mpc_dt must be > 0.");
  }

  // MPCC solver + YAML re-key shim. `configureMpcFromYaml` upstream
  // reads root['mpc'] hardcoded; our YAMLs use `mpcc:` so we rewrite
  // to a temp file before invoking the loader.
  mpc_ = std::make_unique<acados_mpc::MPC>();
  const std::string rekeyed_yaml = rekeyMpccToMpc(cfg_.mpc_yaml_path);
  acados_mpc::configureMpcFromYaml(*mpc_, rekeyed_yaml);
  // Note: ocp_json_file_path is currently informational — the MPC
  // constructor loads the codegen .so via RUNPATH baked at build time,
  // not from the JSON. We still validate the json exists for parity
  // with the Python adapter and to fail fast if the user mis-configured.
  (void)resolveOcpJson(cfg_.ocp_json_file_path, cfg_.mpc_yaml_path);

  control_period_ = mpc_->getPredictionTimeStep();
  if (example_cfg.mpc_dt > 0.0) {
    control_period_ = example_cfg.mpc_dt;
  }

  // Mission + spline. Path precedence: env var > config YAML > demo defaults.
  const std::string env_mission = envOrEmpty("CIRCUIT_MISSION_YAML");
  const std::string env_gates   = envOrEmpty("CIRCUIT_GATES_YAML");
  const auto [demo_mission, demo_gates] = ml::defaultPaths();
  const std::string mission_yaml =
      !env_mission.empty() ? env_mission
                           : (!cfg_.mission_yaml.empty() ? cfg_.mission_yaml : demo_mission);
  const std::string gates_yaml =
      !env_gates.empty() ? env_gates
                         : (!cfg_.gates_yaml.empty() ? cfg_.gates_yaml : demo_gates);

  Eigen::Vector3d origin_pose = initial_state.getPositionVector();
  if (example_cfg.takeoff_altitude_m > 0.0) {
    origin_pose.z() = example_cfg.takeoff_altitude_m;
  }
  const auto setpoints = CircuitGenerator::buildSetpoints(
      origin_pose, mission_yaml, gates_yaml, cfg_.origin_offset_m, cfg_.closing_exit_margin_m);
  traj_ = std::make_unique<spline::TrajectoryGenerator>(
      setpoints, kMpccSplineKnots, cfg_.target_segment_length, cfg_.samples_per_segment);

  s_eval_                = 0.0;
  last_solve_us_         = 0.0;
  last_desired_velocity_ = Eigen::Vector3d::Zero();
}

framework::ControlCommand MpccController::computeCommand(
    const mav_model::State& state,
    const std::vector<framework::ReferenceSample>& /*references*/) {
  if (!mpc_ || !traj_) {
    throw std::runtime_error("MpccController: initialize() must be called first.");
  }
  const Eigen::Vector3d pos       = state.getPositionVector();
  const Eigen::Quaterniond q      = state.getOrientationVector();
  const Eigen::Vector3d vel       = state.getLinearVelocityVector();

  acados_mpc::MPCData* mpc_data = mpc_->getData();
  mpc_data->state.setPosition({pos.x(), pos.y(), pos.z()});
  mpc_data->state.setOrientation({q.w(), q.x(), q.y(), q.z()});
  mpc_data->state.setLinearVelocity({vel.x(), vel.y(), vel.z()});
  mpc_data->state.setTheta(s_eval_);

  auto [win, s_eval] = traj_->getTrajectoryWindow(s_eval_);
  s_eval_            = s_eval;

  // Pack the sliding window into the solver's spline parameters.
  std::array<double, kMpccSplineKnots * 3> spline_pts{};
  std::array<double, kMpccSplineKnots * 3> spline_tan{};
  std::array<double, kMpccSplineKnots * 3> spline_face{};
  std::array<double, kMpccSplineKnots>     s_lengths{};
  std::array<double, kMpccSplineKnots>     softc{};
  const std::size_t n_pts = std::min(win.path.position.size() / 3,
                                      static_cast<std::size_t>(kMpccSplineKnots));
  for (std::size_t i = 0; i < n_pts; ++i) {
    spline_pts[3 * i + 0] = win.path.position[3 * i + 0];
    spline_pts[3 * i + 1] = win.path.position[3 * i + 1];
    spline_pts[3 * i + 2] = win.path.position[3 * i + 2];
    spline_tan[3 * i + 0] = win.path.tangent[3 * i + 0];
    spline_tan[3 * i + 1] = win.path.tangent[3 * i + 1];
    spline_tan[3 * i + 2] = win.path.tangent[3 * i + 2];
    spline_face[3 * i + 0] = win.facing_points[3 * i + 0];
    spline_face[3 * i + 1] = win.facing_points[3 * i + 1];
    spline_face[3 * i + 2] = win.facing_points[3 * i + 2];
  }
  const std::size_t n_s = std::min(win.s_values.size(),
                                    static_cast<std::size_t>(kMpccSplineKnots));
  for (std::size_t i = 0; i < n_s; ++i) {
    s_lengths[i] = win.s_values[i];
  }
  softc.fill(kSoftWeightDefault);
  for (const auto& og : win.original_points) {
    if (og.index >= 0 && og.index < kMpccSplineKnots) {
      softc[og.index] = kSoftWeightOg;
    }
  }
  acados_mpc::OnlineParameters* params = mpc_->getParameters();
  params->setSplinePoints(spline_pts);
  params->setSplineTangents(spline_tan);
  params->setSplineFacePoints(spline_face);
  params->setSplineSoftconstraints(softc);
  params->setSLengths(s_lengths);

  // Solve.
  const auto t0    = std::chrono::high_resolution_clock::now();
  const int status = mpc_->solve();
  const auto t1    = std::chrono::high_resolution_clock::now();
  last_solve_us_   = std::chrono::duration<double>(t1 - t0).count() * 1e6;
  if (status != 0) {
    throw std::runtime_error("MpccController: solver returned status " +
                             std::to_string(status));
  }

  // Project the drone onto the spline (monotonic) for the next iteration.
  if (!win.s_values.empty()) {
    const double s_top   = win.s_values.back();
    const double s_lower = std::max(0.0, s_eval_ - kSearchRadiusM);
    const double s_upper = std::min(s_top, s_eval_ + kSearchRadiusM);
    double best_s        = findClosestS(pos, win, s_lower, s_upper);
    best_s               = std::max(s_eval_, best_s);
    s_eval_              = std::min(std::max(best_s, 0.0), s_top);
  }

  // Expose stage-1 predicted velocity (best proxy for v_des).
  if (win.path.tangent.size() >= 3) {
    last_desired_velocity_ = cfg_.desired_speed * Eigen::Vector3d(win.path.tangent[0],
                                                                  win.path.tangent[1],
                                                                  win.path.tangent[2]);
  }

  framework::ControlCommand cmd;
  cmd.thrust_n      = mpc_data->actuation.getThrust();
  const auto rates  = mpc_data->actuation.getAngularVelocity();
  cmd.angular_rate  = {rates[0], rates[1], rates[2]};
  return cmd;
}

}  // namespace mpc_examples::adapters
