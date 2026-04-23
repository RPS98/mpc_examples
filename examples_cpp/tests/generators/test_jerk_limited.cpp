// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

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
  mav_model::State            state_   = hoverStateAt({0.0, 0.0, 10.0});
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
    ASSERT_TRUE(s.position.allFinite())     << "step " << k;
    ASSERT_TRUE(s.velocity.allFinite())     << "step " << k;
    ASSERT_TRUE(s.acceleration.allFinite()) << "step " << k;
  }
}

TEST_F(JerkLimitedGeneratorTest, AdvertisesPositionVelocityAcceleration) {
  const auto mask = gen_->providedReferenceFields();
  using mpc_examples::framework::ReferenceField;
  using mpc_examples::framework::hasField;
  EXPECT_TRUE(hasField(mask, ReferenceField::kPosition));
  EXPECT_TRUE(hasField(mask, ReferenceField::kVelocity));
  EXPECT_TRUE(hasField(mask, ReferenceField::kAcceleration));
}
