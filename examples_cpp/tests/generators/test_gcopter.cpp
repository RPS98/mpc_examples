// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <memory>

#include "framework/trajectory_generator_base.hpp"
#include "generators/gcopter_generator.hpp"
#include "test_helpers.hpp"

using mpc_examples::adapters::GcopterGenerator;
using mpc_examples::testing::hoverStateAt;
using mpc_examples::testing::loadTestSimConfig;
using mpc_examples::testing::repoPath;

class GcopterGeneratorTest : public ::testing::Test {
protected:
  mpc_examples::ExampleConfig sim_cfg_ = loadTestSimConfig();
  mav_model::State            state_   = hoverStateAt({0.0, 0.0, 10.0});
  std::unique_ptr<GcopterGenerator> gen_;

  void SetUp() override {
    auto cfg = GcopterGenerator::loadConfigFromYaml(
        repoPath("configs/generators/config_gcopter.yaml"));
    gen_ = std::make_unique<GcopterGenerator>(cfg);
    gen_->initialize(state_, sim_cfg_);
  }
};

TEST_F(GcopterGeneratorTest, ProducesFiniteReferencesOver50Steps) {
  const Eigen::Vector3d target(5.0, 0.0, 10.0);
  gen_->onWaypointChanged(target, state_, 0.0);

  for (int k = 0; k < 50; ++k) {
    const double t = 0.01 * k;
    gen_->update(t, state_);
    const auto s = gen_->evaluate(t);
    ASSERT_TRUE(s.position.allFinite()) << "step " << k;
    ASSERT_TRUE(s.velocity.allFinite()) << "step " << k;
  }
}

TEST_F(GcopterGeneratorTest, AdvertisesTrajectoryFields) {
  const auto mask = gen_->providedReferenceFields();
  using mpc_examples::framework::ReferenceField;
  using mpc_examples::framework::hasField;
  EXPECT_TRUE(hasField(mask, ReferenceField::kPosition));
  EXPECT_TRUE(hasField(mask, ReferenceField::kVelocity));
}
