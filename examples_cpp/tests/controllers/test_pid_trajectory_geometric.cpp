// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "controllers/pid_trajectory_geometric_controller.hpp"
#include "framework/controller_base.hpp"
#include "test_helpers.hpp"

using mpc_examples::adapters::PidTrajectoryGeometricController;
using mpc_examples::framework::hasField;
using mpc_examples::framework::ReferenceField;
using mpc_examples::testing::horizonAtPosition;
using mpc_examples::testing::hoverStateAt;
using mpc_examples::testing::loadTestSimConfig;
using mpc_examples::testing::repoPath;

class PidTrajectoryGeometricControllerTest : public ::testing::Test {
protected:
  mpc_examples::ExampleConfig sim_cfg_ = loadTestSimConfig();
  mav_model::State state_              = hoverStateAt({0.0, 0.0, 10.0});
  std::unique_ptr<PidTrajectoryGeometricController> ctrl_;

  void SetUp() override {
    auto cfg = PidTrajectoryGeometricController::loadConfigFromYaml(
        repoPath("configs/controllers/config_pid_trajectory.yaml"));
    ctrl_ = std::make_unique<PidTrajectoryGeometricController>(cfg);
    ctrl_->initialize(state_, sim_cfg_);
  }
};

TEST_F(PidTrajectoryGeometricControllerTest, HorizonShapeIsSingleStep) {
  EXPECT_EQ(ctrl_->referenceHorizonSize(), 1);
  EXPECT_GT(ctrl_->controlPeriod(), 0.0);
}

TEST_F(PidTrajectoryGeometricControllerTest, RequiresPositionAndVelocityReference) {
  const auto mask = ctrl_->requiredReferenceFields();
  EXPECT_TRUE(hasField(mask, ReferenceField::kPosition));
  EXPECT_TRUE(hasField(mask, ReferenceField::kVelocity));
}

TEST_F(PidTrajectoryGeometricControllerTest, ProducesFiniteCommandOverBriefSimulation) {
  const auto refs = horizonAtPosition({5.0, 0.0, 10.0}, ctrl_->referenceHorizonSize());
  for (int k = 0; k < 50; ++k) {
    const auto cmd = ctrl_->computeCommand(state_, refs);
    ASSERT_TRUE(std::isfinite(cmd.thrust_n)) << "step " << k;
    ASSERT_TRUE(cmd.angular_rate.allFinite()) << "step " << k;
    EXPECT_GT(cmd.thrust_n, 0.0) << "step " << k;
  }
  EXPECT_GT(ctrl_->lastSolveTimeMicros(), 0.0);
}

// Replicate the worst-case waypoint-transition seen in the mav_traj_gen run:
// drone flying diagonally at ~3 m/s, new segment starts with ref_vel = 0.
// Velocity error ≈ [0, 2.8, 2.8] → Kd * error ≈ [0, 16.8, 16.8] m/s².
TEST_F(PidTrajectoryGeometricControllerTest,
       FiniteCommandAtWaypointTransitionWithHighVelocityError) {
  mav_model::State moving = hoverStateAt({0.0, 10.0, 20.0});
  moving.setLinearVelocityVector(Eigen::Vector3d(0.0, -2.8, -2.8));

  mpc_examples::framework::ReferenceSample s;
  s.position     = Eigen::Vector3d(0.0, 10.0, 20.0);
  s.velocity     = Eigen::Vector3d::Zero();
  s.acceleration = Eigen::Vector3d::Zero();
  s.yaw          = 0.0;
  const std::vector<mpc_examples::framework::ReferenceSample> refs(1, s);

  for (int k = 0; k < 50; ++k) {
    const auto cmd = ctrl_->computeCommand(moving, refs);
    ASSERT_TRUE(std::isfinite(cmd.thrust_n)) << "step " << k;
    ASSERT_TRUE(cmd.angular_rate.allFinite()) << "step " << k;
    EXPECT_GT(cmd.thrust_n, 0.0) << "step " << k;
  }
}
