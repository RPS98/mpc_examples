// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <memory>

#include "controllers/mpc_position_controller.hpp"
#include "framework/controller_base.hpp"
#include "test_helpers.hpp"

using mpc_examples::adapters::MpcPositionController;
using mpc_examples::testing::hoverStateAt;
using mpc_examples::testing::horizonAtPosition;
using mpc_examples::testing::loadTestSimConfig;
using mpc_examples::testing::repoPath;

class MpcPositionControllerTest : public ::testing::Test {
protected:
  mpc_examples::ExampleConfig sim_cfg_ = loadTestSimConfig();
  mav_model::State            state_   = hoverStateAt({0.0, 0.0, 10.0});
  std::unique_ptr<MpcPositionController> ctrl_;

  void SetUp() override {
    auto cfg = MpcPositionController::loadConfigFromYaml(
        repoPath("configs/controllers/config_mpc.yaml"));
    ctrl_ = std::make_unique<MpcPositionController>(cfg);
    ctrl_->initialize(state_, sim_cfg_);
  }
};

TEST_F(MpcPositionControllerTest, HasValidTimingModel) {
  // MPC-Position drives the horizon progression internally (via
  // setProgressiveReferences) so it only needs one input reference — the
  // current target. What matters is that the timing quantities are strictly
  // positive.
  EXPECT_GE(ctrl_->referenceHorizonSize(), 1);
  EXPECT_GT(ctrl_->referenceHorizonDt(), 0.0);
  EXPECT_GT(ctrl_->controlPeriod(), 0.0);
}

TEST_F(MpcPositionControllerTest, ProducesFiniteCommandOverBriefSimulation) {
  const auto refs = horizonAtPosition({5.0, 0.0, 10.0}, ctrl_->referenceHorizonSize());
  for (int k = 0; k < 5; ++k) {
    const auto cmd = ctrl_->computeCommand(state_, refs);
    ASSERT_TRUE(std::isfinite(cmd.thrust_n)) << "step " << k;
    ASSERT_TRUE(cmd.angular_rate.allFinite()) << "step " << k;
    EXPECT_GT(cmd.thrust_n, 0.0) << "thrust should be positive at step " << k;
  }
  EXPECT_GT(ctrl_->lastSolveTimeMicros(), 0.0);
}

TEST_F(MpcPositionControllerTest, RequiresPositionReference) {
  const auto mask = ctrl_->requiredReferenceFields();
  const auto pos  = mpc_examples::framework::ReferenceField::kPosition;
  EXPECT_TRUE(mpc_examples::framework::hasField(mask, pos));
}
