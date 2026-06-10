// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file pid_solve_benchmark.cpp
 * @brief Google Benchmark of the cascaded PID position controller
 *        (`PidPositionGeometricController` pipeline).
 *
 * Methodology mirrors `BM_PmpcSolve`: an ideal plant-free closed loop. Each
 * tick runs the full PID cascade in the same order the showcase adapter does:
 *
 *   pos PID   -> v_des   (saturated to kMaxSpeed)
 *   vel PID   -> a_des
 *   geometric -> (thrust, body_rates)
 *
 * The state is integrated forward with a Euler step using a unit thrust-to-
 * acceleration assumption (``a = thrust / mass - g·ẑ``) so the controllers
 * see a non-trivial, slowly-varying trajectory: this exercises the PID
 * derivative filter, anti-windup and saturation paths in the same regime as a
 * real flight, instead of converging to a static fixed point after one tick.
 *
 * The carrot is held @ref kCarrotDistance ahead along +x to keep the steady
 * tracking error constant. The pipeline is sub-millisecond so it's reported
 * in microseconds for parity with the MPC solve benchmarks.
 *
 * No reference to the framework adapter (`PidPositionGeometricController`) is
 * needed — the benchmark binds the three underlying mav_controllers libraries
 * directly to keep the dependency surface ROS-2-free and tiny.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include <benchmark/benchmark.h>

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

#include "geometric_controller/geometric_controller.hpp"
#include "mav_benchmark/bench_common.hpp"
#include "pid_controllers/position_controller.hpp"
#include "pid_controllers/velocity_controller.hpp"

namespace {

using Vec3 = Eigen::Vector3d;
using Quat = Eigen::Quaterniond;

/// Reads a 3-vector field, throwing with the parent path on failure.
Vec3 readVec3(const YAML::Node& node, const std::string& path) {
  if (!node || !node.IsSequence() || node.size() != 3) {
    throw std::invalid_argument(path + ": expected a 3-element sequence.");
  }
  return Vec3(node[0].as<double>(), node[1].as<double>(), node[2].as<double>());
}

/// Accepts either a scalar (broadcast to all 3 axes) or a 3-vector.
Vec3 readVec3OrScalar(const YAML::Node& node, const std::string& path) {
  if (!node) {
    throw std::invalid_argument(path + " is required.");
  }
  if (node.IsSequence()) {
    return readVec3(node, path);
  }
  return Vec3::Constant(node.as<double>());
}

/// Translate the YAML PID block to a pid_controller::PIDParameters.
pid_controller::PIDParameters<double> parsePid(const YAML::Node& node, const std::string& section) {
  pid_controller::PIDParameters<double> params;
  params.Kp_gains = readVec3(node["kp"], section + ".kp");
  params.Ki_gains = readVec3(node["ki"], section + ".ki");
  params.Kd_gains = readVec3(node["kd"], section + ".kd");
  if (node["antiwindup_cte"]) {
    params.antiwindup_cte = readVec3OrScalar(node["antiwindup_cte"], section + ".antiwindup_cte");
  }
  if (node["alpha"]) {
    params.alpha = readVec3OrScalar(node["alpha"], section + ".alpha");
  }
  if (node["saturation_upper"] && node["saturation_lower"]) {
    params.upper_output_saturation =
        readVec3(node["saturation_upper"], section + ".saturation_upper");
    params.lower_output_saturation =
        readVec3(node["saturation_lower"], section + ".saturation_lower");
  }
  params.proportional_saturation_flag = true;
  return params;
}

struct PidCascade {
  pid_controllers::PositionController<double> pos_ctrl;
  pid_controllers::VelocityController<double> vel_ctrl;
  geometric_controller::GeometricController<double> geo_ctrl;
  double mass = 0.0;
  static constexpr double kGravity = 9.81;
};

PidCascade buildCascade(const std::string& yaml_path) {
  const YAML::Node root = YAML::LoadFile(yaml_path);
  const YAML::Node ctrl = root["controller"];
  if (!ctrl || !ctrl.IsMap()) {
    throw std::invalid_argument(yaml_path + ": 'controller' must be a mapping.");
  }

  pid_controllers::PositionControllerParameters<double> pos_p;
  pos_p.pid_parameters = parsePid(ctrl["position"], "controller.position");
  pid_controllers::VelocityControllerParameters<double> vel_p;
  vel_p.pid_parameters = parsePid(ctrl["velocity"], "controller.velocity");

  geometric_controller::AttitudeGeometricControllerParameters<double> att_p;
  att_p.vehicle_mass = ctrl["geometric"]["mass"].as<double>();
  geometric_controller::RatesGeometricControllerParameters<double> rate_p;
  rate_p.kp_rotation = readVec3(ctrl["geometric"]["rotation_kp"], "controller.geometric.rotation_kp");

  return PidCascade{
      pid_controllers::PositionController<double>(pos_p),
      pid_controllers::VelocityController<double>(vel_p),
      geometric_controller::GeometricController<double>(att_p, rate_p),
      att_p.vehicle_mass,
  };
}

Vec3 saturateVelocity(const Vec3& v, double v_max) {
  const double speed = v.norm();
  if (speed > v_max) {
    return (v / speed) * v_max;
  }
  return v;
}

void BM_PidSolve(benchmark::State& state) {
  PidCascade c = buildCascade(mav_benchmark::kPidYaml);
  const double dt = mav_benchmark::kPidDt;
  const double v_max = mav_benchmark::kMaxSpeed;

  Vec3 position = Vec3::Zero();
  Vec3 velocity = Vec3::Zero();
  Quat orientation = Quat::Identity();

  long iter = 0;
  for (auto _ : state) {
    // Carrot held kCarrotDistance ahead along +x; the heading is rotated
    // slightly each tick to defeat trivial steady state.
    const double yaw = 1e-4 * static_cast<double>(iter++);
    const Vec3 ref_pos = position +
                         Vec3(mav_benchmark::kCarrotDistance * std::cos(yaw),
                              mav_benchmark::kCarrotDistance * std::sin(yaw), 0.0);

    Vec3 v_des = c.pos_ctrl.positionToLinearVelocity(position, ref_pos, dt);
    v_des = saturateVelocity(v_des, v_max);
    const Vec3 a_des = c.vel_ctrl.linearVelocityToLinearAcceleration(velocity, v_des, dt);
    const auto [thrust, body_rates] = c.geo_ctrl.accelerationToRates(a_des, /*yaw=*/0.0, orientation);

    benchmark::DoNotOptimize(thrust);
    benchmark::DoNotOptimize(body_rates);

    // Advance the open-loop "plant" so the next tick sees a non-trivial input.
    // a_plant = thrust/m * R*ẑ_body - g·ẑ_world. With orientation == identity
    // it collapses to a vertical balance; we still ramp the position along the
    // desired velocity to keep the cascade saturated in steady state.
    position += v_des * dt;
    velocity += (a_des - PidCascade::kGravity * Vec3::UnitZ() + thrust / c.mass * Vec3::UnitZ()) * dt;
    // Keep the orientation drifting very slightly so the geometric stage sees
    // a non-identity quaternion in subsequent iterations.
    const Vec3 small_rot = body_rates * dt;
    orientation = orientation * Quat(Eigen::AngleAxisd(small_rot.norm() + 1e-9,
                                                       (small_rot + Vec3::UnitX() * 1e-9).normalized()));
    orientation.normalize();
  }
}

BENCHMARK(BM_PidSolve)->Unit(benchmark::kMicrosecond)->Threads(1)->Repetitions(10);

}  // namespace
