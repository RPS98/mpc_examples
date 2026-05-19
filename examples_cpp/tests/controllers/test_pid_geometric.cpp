// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "controllers/pid_position_geometric_controller.hpp"
#include "framework/controller_base.hpp"
#include "test_helpers.hpp"
#include "utils/utils.hpp"  // quaternionToEuler

using mpc_examples::adapters::PidPositionGeometricController;
using mpc_examples::quaternionToEuler;
using mpc_examples::testing::horizonAtPosition;
using mpc_examples::testing::hoverStateAt;
using mpc_examples::testing::loadTestSimConfig;
using mpc_examples::testing::repoPath;

class PidGeometricControllerTest : public ::testing::Test {
protected:
  mpc_examples::ExampleConfig sim_cfg_ = loadTestSimConfig();
  mav_model::State state_              = hoverStateAt({0.0, 0.0, 10.0});
  std::unique_ptr<PidPositionGeometricController> ctrl_;

  void SetUp() override {
    // Loads the real controller YAML shipped under configs/controllers/.
    auto cfg = PidPositionGeometricController::loadConfigFromYaml(
        repoPath("configs/controllers/config_pid.yaml"));
    ctrl_ = std::make_unique<PidPositionGeometricController>(cfg);
    ctrl_->initialize(state_, sim_cfg_);
  }
};

TEST_F(PidGeometricControllerTest, HorizonShapeIsSingleStep) {
  EXPECT_EQ(ctrl_->referenceHorizonSize(), 1);
  EXPECT_GT(ctrl_->controlPeriod(), 0.0);
}

TEST_F(PidGeometricControllerTest, ProducesFiniteCommandOverBriefSimulation) {
  const auto refs = horizonAtPosition({5.0, 0.0, 10.0}, ctrl_->referenceHorizonSize());
  for (int k = 0; k < 50; ++k) {
    const auto cmd = ctrl_->computeCommand(state_, refs);
    ASSERT_TRUE(std::isfinite(cmd.thrust_n)) << "step " << k;
    ASSERT_TRUE(cmd.angular_rate.allFinite()) << "step " << k;
    EXPECT_GT(cmd.thrust_n, 0.0) << "thrust should be positive at step " << k;
  }
}

TEST_F(PidGeometricControllerTest, RequiresAtLeastPositionReference) {
  const auto mask = ctrl_->requiredReferenceFields();
  const auto pos  = mpc_examples::framework::ReferenceField::kPosition;
  EXPECT_TRUE(mpc_examples::framework::hasField(mask, pos));
}

// Axis-isolation contract: ENU + FLU at yaw=0, +X hop ⇒ pitch>0 / roll≈0,
// +Y hop ⇒ roll<0 / pitch≈0 (left bank). Each test runs the cascade against
// the hover state and asserts the very first body-rate command tilts the
// drone along the requested cardinal axis with no cross-coupling.

namespace {

double computeRollFromCmd(const PidPositionGeometricController* ctrl,
                          const mav_model::State& state,
                          const Eigen::Vector3d& target_position) {
  const auto refs = horizonAtPosition(target_position, ctrl->referenceHorizonSize());
  Eigen::Vector3d rates = Eigen::Vector3d::Zero();
  for (int k = 0; k < 5; ++k) {
    const auto cmd = const_cast<PidPositionGeometricController*>(ctrl)->computeCommand(state, refs);
    rates = cmd.angular_rate;
  }
  return rates.x();
}

double computePitchFromCmd(const PidPositionGeometricController* ctrl,
                           const mav_model::State& state,
                           const Eigen::Vector3d& target_position) {
  const auto refs = horizonAtPosition(target_position, ctrl->referenceHorizonSize());
  Eigen::Vector3d rates = Eigen::Vector3d::Zero();
  for (int k = 0; k < 5; ++k) {
    const auto cmd = const_cast<PidPositionGeometricController*>(ctrl)->computeCommand(state, refs);
    rates = cmd.angular_rate;
  }
  return rates.y();
}

}  // namespace

TEST_F(PidGeometricControllerTest, HopInPlusXProducesOnlyPitchCommand) {
  // +X hop ⇒ pitch_rate > 0, roll_rate ≈ 0.
  state_ = hoverStateAt({0.0, 0.0, 1.0});
  ctrl_->initialize(state_, sim_cfg_);
  const Eigen::Vector3d target(10.0, 0.0, 1.0);

  const double roll_rate  = computeRollFromCmd(ctrl_.get(), state_, target);
  state_ = hoverStateAt({0.0, 0.0, 1.0});
  ctrl_->initialize(state_, sim_cfg_);
  const double pitch_rate = computePitchFromCmd(ctrl_.get(), state_, target);

  EXPECT_GT(pitch_rate, 0.05)
      << "Expected positive pitch rate to tilt forward for +X hop, got " << pitch_rate;
  EXPECT_NEAR(roll_rate, 0.0, 1e-2)
      << "Cross-coupling: +X hop produced roll_rate=" << roll_rate
      << "; expected ≈ 0 because there is no Y/Z accel setpoint.";
}

TEST_F(PidGeometricControllerTest, HopInPlusYProducesOnlyRollCommand) {
  // +Y hop ⇒ roll_rate < 0 (left bank), pitch_rate ≈ 0.
  state_ = hoverStateAt({0.0, 0.0, 1.0});
  ctrl_->initialize(state_, sim_cfg_);
  const Eigen::Vector3d target(0.0, 10.0, 1.0);

  const double roll_rate  = computeRollFromCmd(ctrl_.get(), state_, target);
  state_ = hoverStateAt({0.0, 0.0, 1.0});
  ctrl_->initialize(state_, sim_cfg_);
  const double pitch_rate = computePitchFromCmd(ctrl_.get(), state_, target);

  EXPECT_LT(roll_rate, -0.05)
      << "Expected negative roll rate to bank left for +Y hop, got " << roll_rate;
  EXPECT_NEAR(pitch_rate, 0.0, 1e-2)
      << "Cross-coupling: +Y hop produced pitch_rate=" << pitch_rate
      << "; expected ≈ 0 because there is no X/Z accel setpoint.";
}

TEST_F(PidGeometricControllerTest, HopInMinusYProducesOnlyRollCommand) {
  // -Y hop ⇒ roll_rate > 0 (right bank), pitch_rate ≈ 0.
  state_ = hoverStateAt({0.0, 0.0, 1.0});
  ctrl_->initialize(state_, sim_cfg_);
  const Eigen::Vector3d target(0.0, -10.0, 1.0);

  const double roll_rate  = computeRollFromCmd(ctrl_.get(), state_, target);
  state_ = hoverStateAt({0.0, 0.0, 1.0});
  ctrl_->initialize(state_, sim_cfg_);
  const double pitch_rate = computePitchFromCmd(ctrl_.get(), state_, target);

  EXPECT_GT(roll_rate, 0.05)
      << "Expected positive roll rate to bank right for -Y hop, got " << roll_rate;
  EXPECT_NEAR(pitch_rate, 0.0, 1e-2)
      << "Cross-coupling: -Y hop produced pitch_rate=" << pitch_rate
      << "; expected ≈ 0 because there is no X/Z accel setpoint.";
}

TEST_F(PidGeometricControllerTest, HopInMinusXProducesOnlyPitchCommand) {
  // -X hop ⇒ pitch_rate < 0, roll_rate ≈ 0.
  state_ = hoverStateAt({0.0, 0.0, 1.0});
  ctrl_->initialize(state_, sim_cfg_);
  const Eigen::Vector3d target(-10.0, 0.0, 1.0);

  const double roll_rate  = computeRollFromCmd(ctrl_.get(), state_, target);
  state_ = hoverStateAt({0.0, 0.0, 1.0});
  ctrl_->initialize(state_, sim_cfg_);
  const double pitch_rate = computePitchFromCmd(ctrl_.get(), state_, target);

  EXPECT_LT(pitch_rate, -0.05)
      << "Expected negative pitch rate to tilt backward for -X hop, got " << pitch_rate;
  EXPECT_NEAR(roll_rate, 0.0, 1e-2)
      << "Cross-coupling: -X hop produced roll_rate=" << roll_rate
      << "; expected ≈ 0 because there is no Y/Z accel setpoint.";
}

// Dynamic-state cross-coupling probe: drone already moving +X at 1 m/s
// toward a +X target must not produce a roll-rate command.
TEST_F(PidGeometricControllerTest, DynamicForwardStateDoesNotInduceRoll) {
  state_ = hoverStateAt({2.0, 0.0, 1.0});
  state_.setLinearVelocityVector({1.0, 0.0, 0.0});
  ctrl_->initialize(state_, sim_cfg_);
  const Eigen::Vector3d target(10.0, 0.0, 1.0);
  const auto refs = horizonAtPosition(target, ctrl_->referenceHorizonSize());

  Eigen::Vector3d rates = Eigen::Vector3d::Zero();
  double thrust         = 0.0;
  for (int k = 0; k < 5; ++k) {
    const auto cmd = ctrl_->computeCommand(state_, refs);
    rates  = cmd.angular_rate;
    thrust = cmd.thrust_n;
  }
  EXPECT_NEAR(rates.x(), 0.0, 1e-2)
      << "In-motion +X probe leaked into roll_rate=" << rates.x();
  EXPECT_GT(thrust, 0.0);
}
