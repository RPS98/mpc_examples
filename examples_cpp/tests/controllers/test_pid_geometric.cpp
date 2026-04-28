// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "controllers/pid_position_geometric_controller.hpp"
#include "framework/controller_base.hpp"
#include "test_helpers.hpp"

using mpc_examples::adapters::PidPositionGeometricController;
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
