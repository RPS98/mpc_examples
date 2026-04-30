// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>

#include "framework/trajectory_generator_base.hpp"
#include "generators/jerk_limited_generator.hpp"
#include "test_helpers.hpp"

using mpc_examples::adapters::JerkLimitedGenerator;
using mpc_examples::testing::hoverStateAt;
using mpc_examples::testing::loadTestSimConfig;
using mpc_examples::testing::repoPath;

class JerkLimitedGeneratorTest : public ::testing::Test {
protected:
  mpc_examples::ExampleConfig sim_cfg_ = loadTestSimConfig();
  mav_model::State state_              = hoverStateAt({0.0, 0.0, 10.0});
  std::unique_ptr<JerkLimitedGenerator> gen_;

  void SetUp() override {
    auto cfg = JerkLimitedGenerator::loadConfigFromYaml(
        repoPath("configs/generators/config_jerk_limited.yaml"));
    gen_ = std::make_unique<JerkLimitedGenerator>(cfg);
    gen_->initialize(state_, sim_cfg_);
  }
};

TEST_F(JerkLimitedGeneratorTest, ProducesFiniteReferencesOver50Steps) {
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

TEST_F(JerkLimitedGeneratorTest, AdvertisesPositionVelocityAcceleration) {
  const auto mask = gen_->providedReferenceFields();
  using mpc_examples::framework::hasField;
  using mpc_examples::framework::ReferenceField;
  EXPECT_TRUE(hasField(mask, ReferenceField::kPosition));
  EXPECT_TRUE(hasField(mask, ReferenceField::kVelocity));
  EXPECT_TRUE(hasField(mask, ReferenceField::kAcceleration));
}

TEST_F(JerkLimitedGeneratorTest, StartsAtInitialPosition) {
  const Eigen::Vector3d p0     = state_.getPositionVector();
  const Eigen::Vector3d target = p0 + Eigen::Vector3d(5.0, 0.0, 0.0);
  gen_->onWaypointChanged(target, state_, 0.0);

  const auto s = gen_->evaluate(0.0);
  EXPECT_NEAR((s.position - p0).norm(), 0.0, 1e-3);
  EXPECT_NEAR(s.velocity.norm(), 0.0, 1e-3);
  EXPECT_NEAR(s.acceleration.norm(), 0.0, 1e-3);
}

TEST_F(JerkLimitedGeneratorTest, ReachesTargetPastDuration) {
  const Eigen::Vector3d target(5.0, 0.0, 10.0);
  gen_->onWaypointChanged(target, state_, 0.0);

  // Far past the segment duration: must clamp to target with zero motion.
  const auto s = gen_->evaluate(1000.0);
  EXPECT_NEAR((s.position - target).norm(), 0.0, 1e-3);
  EXPECT_NEAR(s.velocity.norm(), 0.0, 1e-9);
  EXPECT_NEAR(s.acceleration.norm(), 0.0, 1e-9);
}

TEST_F(JerkLimitedGeneratorTest, NoPlanWhenWaypointEqualsCurrent) {
  const Eigen::Vector3d p0 = state_.getPositionVector();
  gen_->onWaypointChanged(p0, state_, 0.0);

  // No plan was produced; evaluate must return hold_pos == p0 with zero motion.
  for (double t : {0.0, 1.0, 100.0}) {
    const auto s = gen_->evaluate(t);
    EXPECT_NEAR((s.position - p0).norm(), 0.0, 1e-9) << "t=" << t;
    EXPECT_NEAR(s.velocity.norm(), 0.0, 1e-9) << "t=" << t;
    EXPECT_NEAR(s.acceleration.norm(), 0.0, 1e-9) << "t=" << t;
  }
}

TEST_F(JerkLimitedGeneratorTest, ReplanResetsTimeOrigin) {
  const Eigen::Vector3d p0 = state_.getPositionVector();

  gen_->onWaypointChanged(Eigen::Vector3d(5.0, 0.0, 10.0), state_, 0.0);
  // First segment: at t=0 we're at p0.
  EXPECT_NEAR((gen_->evaluate(0.0).position - p0).norm(), 0.0, 1e-3);

  // Second segment starting at t=5: state still at p0 (scheduler may fire
  // before vehicle settles, same as the rest of p2p adapters). The new
  // segment must restart from the current state at t_start=5.
  gen_->onWaypointChanged(Eigen::Vector3d(-3.0, 4.0, 10.0), state_, 5.0);
  EXPECT_NEAR((gen_->evaluate(5.0).position - p0).norm(), 0.0, 1e-3);
}

TEST_F(JerkLimitedGeneratorTest, VelocityProfileApproachesMaxSpeed) {
  // Long hop so the S-curve has room to plateau at v_max.
  const Eigen::Vector3d target(20.0, 0.0, 10.0);
  gen_->onWaypointChanged(target, state_, 0.0);

  const double max_speed = sim_cfg_.max_speed;
  ASSERT_GT(max_speed, 0.0);

  // Sample evaluate(t) over a generous horizon (well past expected duration).
  // The S-curve must reach at least 90% of max_speed and never exceed it.
  double max_observed_speed = 0.0;
  for (int k = 0; k < 6000; ++k) {
    const double t     = 0.01 * k;
    const auto s       = gen_->evaluate(t);
    const double v     = s.velocity.norm();
    max_observed_speed = std::max(max_observed_speed, v);
    EXPECT_LE(v, max_speed * 1.001) << "t=" << t;
  }
  EXPECT_GE(max_observed_speed, 0.9 * max_speed);
}
