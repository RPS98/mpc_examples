// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <memory>

#include "framework/trajectory_generator_base.hpp"
#include "generators/mav_traj_gen_generator.hpp"
#include "test_helpers.hpp"

using mpc_examples::adapters::MavTrajGenGenerator;
using mpc_examples::testing::hoverStateAt;
using mpc_examples::testing::loadTestSimConfig;
using mpc_examples::testing::repoPath;

class MavTrajGenGeneratorTest : public ::testing::Test {
protected:
  mpc_examples::ExampleConfig sim_cfg_ = loadTestSimConfig();
  mav_model::State state_              = hoverStateAt({0.0, 0.0, 10.0});
  std::unique_ptr<MavTrajGenGenerator> gen_;

  void SetUp() override {
    auto cfg = MavTrajGenGenerator::loadConfigFromYaml(
        repoPath("configs/generators/config_mav_traj_gen.yaml"));
    gen_ = std::make_unique<MavTrajGenGenerator>(cfg);
    gen_->initialize(state_, sim_cfg_);
  }
};

TEST_F(MavTrajGenGeneratorTest, ProducesFiniteReferencesOver50Steps) {
  const Eigen::Vector3d target(5.0, 0.0, 10.0);
  gen_->onWaypointChanged(target, state_, 0.0);

  for (int k = 0; k < 50; ++k) {
    const double t = 0.01 * k;
    gen_->update(t, state_);
    const auto s = gen_->evaluate(t);
    ASSERT_TRUE(s.position.allFinite()) << "step " << k;
    ASSERT_TRUE(s.velocity.allFinite()) << "step " << k;
    ASSERT_TRUE(s.acceleration.allFinite()) << "step " << k;
  }
}

TEST_F(MavTrajGenGeneratorTest, AdvertisesTrajectoryFields) {
  const auto mask = gen_->providedReferenceFields();
  using mpc_examples::framework::hasField;
  using mpc_examples::framework::ReferenceField;
  EXPECT_TRUE(hasField(mask, ReferenceField::kPosition));
  EXPECT_TRUE(hasField(mask, ReferenceField::kVelocity));
  EXPECT_TRUE(hasField(mask, ReferenceField::kAcceleration));
}

TEST_F(MavTrajGenGeneratorTest, FreezesAtTargetPastSegmentEnd) {
  const Eigen::Vector3d start  = state_.getPositionVector();
  const Eigen::Vector3d target = start + Eigen::Vector3d(2.0, 0.0, 0.0);
  gen_->onWaypointChanged(target, state_, 0.0);

  // Sample well past any plausible segment duration.
  const double t_far = 1e3;
  gen_->update(t_far, state_);
  const auto s = gen_->evaluate(t_far);

  EXPECT_NEAR((s.position - target).norm(), 0.0, 1e-6);
  EXPECT_NEAR(s.velocity.norm(), 0.0, 1e-9);
  EXPECT_NEAR(s.acceleration.norm(), 0.0, 1e-9);
}
