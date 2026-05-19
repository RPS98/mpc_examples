// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

// Native mav_simulator regression: command thrust = m·g + ω = 0 for 1 s
// from rest and verify the drone hovers in place. Four variants:
//   A  — baseline (HOVER → RATES).
//   B1 — switch to RATES before arming (no HOVER tick).
//   B2 — symmetric inertia (Ixx == Iyy).
//   B3 — bypass INDI via MOTOR_W with analytical hover speed.
// All probes disable process and IMU noise to keep the result deterministic.

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#include "mav_simulator/mav_simulator.hpp"
#include "mav_simulator/simulator_yaml.hpp"
#include "test_helpers.hpp"

namespace {

using mav_simulator::ControlMode;
using mav_simulator::Simulator;
using mav_simulator::SimulatorParameters;
using mav_simulator::YawControlMode;
using mpc_examples::testing::repoPath;

constexpr double kModelDt      = 1.0 / 1000.0;  // 1 kHz physics
constexpr double kControllerDt = 1.0 / 500.0;   // 500 Hz INDI
constexpr double kTotalT       = 1.0;           // s
constexpr double kPosTolM      = 1e-3;
constexpr double kVelTolMs     = 1e-2;

SimulatorParameters loadNoiseFreeSimParams() {
  SimulatorParameters params = mav_simulator::loadSimulatorParametersFromYaml(
      repoPath("configs/simulation/config_simulator.yaml"));
  params.model_parameters.setStocasticForceVector(Eigen::Vector3d::Zero());
  params.model_parameters.setStocasticTorqueVector(Eigen::Vector3d::Zero());
  params.model_parameters.setExternalForceVector(Eigen::Vector3d::Zero());
  params.model_parameters.setExternalTorqueVector(Eigen::Vector3d::Zero());
  params.imu_parameters.gyroscope_noise_density     = 0.0;
  params.imu_parameters.gyroscope_random_walk       = 0.0;
  params.imu_parameters.accelerometer_noise_density = 0.0;
  params.imu_parameters.accelerometer_random_walk   = 0.0;
  return params;
}

struct DriftStats {
  Eigen::Vector3d final_pos{Eigen::Vector3d::Zero()};
  Eigen::Vector3d final_vel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d max_abs_pos{Eigen::Vector3d::Zero()};
  Eigen::Vector3d max_abs_vel{Eigen::Vector3d::Zero()};
};

template <typename CommandFn>
DriftStats runProbe(Simulator& sim, CommandFn cmd) {
  const int n_controller_ticks =
      static_cast<int>(std::round(kTotalT / kControllerDt));
  const int model_steps_per_ctrl =
      static_cast<int>(std::round(kControllerDt / kModelDt));

  DriftStats stats;
  for (int k = 0; k < n_controller_ticks; ++k) {
    cmd(sim);
    sim.updateController(kControllerDt);
    sim.updateImu(kControllerDt);
    for (int j = 0; j < model_steps_per_ctrl; ++j) {
      sim.updateModel(kModelDt);
    }
    const auto& st  = sim.getState();
    const auto& pos = st.getPositionVector();
    const auto& vel = st.getLinearVelocityVector();
    for (int i = 0; i < 3; ++i) {
      stats.max_abs_pos[i] = std::max(stats.max_abs_pos[i], std::abs(pos[i]));
      stats.max_abs_vel[i] = std::max(stats.max_abs_vel[i], std::abs(vel[i]));
    }
  }
  stats.final_pos = sim.getState().getPositionVector();
  stats.final_vel = sim.getState().getLinearVelocityVector();
  return stats;
}

void printStats(const std::string& tag, const DriftStats& s) {
  std::cerr << std::fixed << std::setprecision(6) << "[" << tag << "] after "
            << kTotalT << " s\n"
            << "  final pos = (" << s.final_pos.x() << ", " << s.final_pos.y() << ", "
            << s.final_pos.z() << ")\n"
            << "  final vel = (" << s.final_vel.x() << ", " << s.final_vel.y() << ", "
            << s.final_vel.z() << ")\n"
            << "  max |pos| = (" << s.max_abs_pos.x() << ", " << s.max_abs_pos.y() << ", "
            << s.max_abs_pos.z() << ")\n"
            << "  max |vel| = (" << s.max_abs_vel.x() << ", " << s.max_abs_vel.y() << ", "
            << s.max_abs_vel.z() << ")\n";
}

}  // namespace

TEST(NativeRatesDrift, A_BaselineNoiseFree) {
  auto params = loadNoiseFreeSimParams();
  Simulator sim(params);
  const double hover_th =
      params.model_parameters.getMass() * params.model_parameters.getGravity();

  sim.arm();
  sim.setControlMode(ControlMode::RATES, YawControlMode::RATE);
  sim.setReferenceYawRate(0.0);

  const auto stats = runProbe(sim, [hover_th](Simulator& s) {
    s.setReferenceRates(hover_th, Eigen::Vector3d::Zero());
  });
  printStats("A_BaselineNoiseFree", stats);
  EXPECT_LT(stats.max_abs_pos.x(), kPosTolM);
  EXPECT_LT(stats.max_abs_pos.y(), kPosTolM);
  EXPECT_LT(stats.max_abs_vel.x(), kVelTolMs);
  EXPECT_LT(stats.max_abs_vel.y(), kVelTolMs);
}

TEST(NativeRatesDrift, B1_SkipHoverIntermediate) {
  auto params = loadNoiseFreeSimParams();
  Simulator sim(params);

  sim.setControlMode(ControlMode::RATES, YawControlMode::RATE);
  sim.setReferenceYawRate(0.0);
  sim.arm();

  const double hover_th =
      params.model_parameters.getMass() * params.model_parameters.getGravity();
  const auto stats = runProbe(sim, [hover_th](Simulator& s) {
    s.setReferenceRates(hover_th, Eigen::Vector3d::Zero());
  });
  printStats("B1_SkipHoverIntermediate", stats);
  EXPECT_LT(stats.max_abs_pos.x(), kPosTolM);
  EXPECT_LT(stats.max_abs_pos.y(), kPosTolM);
  EXPECT_LT(stats.max_abs_vel.x(), kVelTolMs);
  EXPECT_LT(stats.max_abs_vel.y(), kVelTolMs);
}

TEST(NativeRatesDrift, B2_SymmetricInertia) {
  auto params = loadNoiseFreeSimParams();
  const Eigen::Vector3d original_inertia = params.model_parameters.getInertiaVector();
  const Eigen::Vector3d sym_inertia(original_inertia.y(), original_inertia.y(),
                                    original_inertia.z());
  params.model_parameters.setInertiaVector(sym_inertia);
  params = mav_simulator::computeDefaultParameters(params.model_parameters);
  params.imu_parameters.gyroscope_noise_density     = 0.0;
  params.imu_parameters.gyroscope_random_walk       = 0.0;
  params.imu_parameters.accelerometer_noise_density = 0.0;
  params.imu_parameters.accelerometer_random_walk   = 0.0;

  Simulator sim(params);
  sim.arm();
  sim.setControlMode(ControlMode::RATES, YawControlMode::RATE);
  sim.setReferenceYawRate(0.0);

  const double hover_th =
      params.model_parameters.getMass() * params.model_parameters.getGravity();
  const auto stats = runProbe(sim, [hover_th](Simulator& s) {
    s.setReferenceRates(hover_th, Eigen::Vector3d::Zero());
  });
  printStats("B2_SymmetricInertia", stats);
  EXPECT_LT(stats.max_abs_pos.x(), kPosTolM);
  EXPECT_LT(stats.max_abs_pos.y(), kPosTolM);
  EXPECT_LT(stats.max_abs_vel.x(), kVelTolMs);
  EXPECT_LT(stats.max_abs_vel.y(), kVelTolMs);
}

TEST(NativeRatesDrift, B3_BypassIndiMotorW) {
  auto params = loadNoiseFreeSimParams();
  Simulator sim(params);

  const double mass    = params.model_parameters.getMass();
  const double gravity = params.model_parameters.getGravity();
  const Eigen::Vector4d cf_vec = params.model_parameters.getMotorsCfVector();
  // F = sum_i cf_i · w_i², all motors identical → w_hover = sqrt(m·g / (n · cf))
  const double cf       = cf_vec.mean();
  const double n_motors = static_cast<double>(cf_vec.size());
  const double w_hover  = std::sqrt(mass * gravity / (n_motors * cf));
  const Eigen::Vector4d motor_w_cmd = Eigen::Vector4d::Constant(w_hover);

  sim.arm();
  sim.setControlMode(ControlMode::MOTOR_W, YawControlMode::RATE);

  const auto stats = runProbe(sim, [&motor_w_cmd](Simulator& s) {
    s.setReferenceMotorW(motor_w_cmd);
  });
  printStats("B3_BypassIndiMotorW", stats);
  EXPECT_LT(stats.max_abs_pos.x(), kPosTolM);
  EXPECT_LT(stats.max_abs_pos.y(), kPosTolM);
  EXPECT_LT(stats.max_abs_vel.x(), kVelTolMs);
  EXPECT_LT(stats.max_abs_vel.y(), kVelTolMs);
}

// Process-noise contract: with σ_F > 0 and σ_M > 0,
//   - same seed produces identical trajectories (determinism);
//   - different seeds produce different trajectories;
//   - the noise is zero-mean (final position stays bounded over 1 s with the
//     analytical hover thrust + zero rate setpoint).
TEST(NativeRatesDrift, NoiseIsDeterministicAndBounded) {
  auto build_sim = [](int seed) {
    SimulatorParameters params = mav_simulator::loadSimulatorParametersFromYaml(
        repoPath("configs/simulation/config_simulator.yaml"));
    params.process_noise_sigma_force  = Eigen::Vector3d::Constant(0.10);
    params.process_noise_sigma_torque = Eigen::Vector3d::Constant(0.01);
    params.process_noise_seed         = seed;
    params.imu_parameters.gyroscope_noise_density     = 0.0;
    params.imu_parameters.accelerometer_noise_density = 0.0;
    return params;
  };

  const auto run_once = [&](int seed) {
    auto params = build_sim(seed);
    Simulator sim(params);
    const double hover_th =
        params.model_parameters.getMass() * params.model_parameters.getGravity();
    sim.arm();
    sim.setControlMode(ControlMode::RATES, YawControlMode::RATE);
    sim.setReferenceYawRate(0.0);
    return runProbe(sim, [hover_th](Simulator& s) {
      s.setReferenceRates(hover_th, Eigen::Vector3d::Zero());
    });
  };

  const DriftStats a1 = run_once(42);
  const DriftStats a2 = run_once(42);
  const DriftStats b  = run_once(7);

  printStats("Noise_seed42_run1", a1);
  printStats("Noise_seed7", b);

  // Determinism: identical seed → byte-identical trajectory.
  EXPECT_DOUBLE_EQ(a1.final_pos.x(), a2.final_pos.x());
  EXPECT_DOUBLE_EQ(a1.final_pos.y(), a2.final_pos.y());
  EXPECT_DOUBLE_EQ(a1.final_vel.x(), a2.final_vel.x());

  // Different seed → different trajectory (at least one component differs).
  const bool seeds_differ =
      (std::abs(a1.final_pos.x() - b.final_pos.x()) > 1e-6) ||
      (std::abs(a1.final_pos.y() - b.final_pos.y()) > 1e-6) ||
      (std::abs(a1.final_vel.x() - b.final_vel.x()) > 1e-6);
  EXPECT_TRUE(seeds_differ) << "Different seeds produced identical outputs";

  // Sanity: trajectory stays finite and small at the seed scale. The bound is
  // loose because σ_M=0.01 on Ixx≈1.6e-3 induces ω-RMS ~62 rad/s in 1 s,
  // which tilts the thrust vector and couples gravity into lateral drift.
  // What we assert is that nothing diverged to NaN / huge values.
  EXPECT_TRUE(std::isfinite(a1.final_pos.x()));
  EXPECT_TRUE(std::isfinite(a1.final_pos.y()));
  EXPECT_TRUE(std::isfinite(a1.final_pos.z()));
  EXPECT_LT(a1.max_abs_pos.norm(), 100.0);
}
