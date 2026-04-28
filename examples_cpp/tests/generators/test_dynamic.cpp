// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

// The dynamic_trajectory_generator adapter is no longer used in any runtime
// binary, but the interface lives on so downstream code that consumes it
// stays working. This test exercises the construction + short-simulation
// path so a compiler/config regression is caught in CI.

#include <gtest/gtest.h>

#include <memory>

#include "framework/trajectory_generator_base.hpp"
#include "generators/dynamic_trajectory_generator.hpp"
#include "test_helpers.hpp"

using mpc_examples::adapters::DynamicTrajectoryGenerator;
using mpc_examples::testing::hoverStateAt;
using mpc_examples::testing::loadTestSimConfig;
using mpc_examples::testing::repoPath;

class DynamicTrajectoryGeneratorTest : public ::testing::Test {
protected:
  mpc_examples::ExampleConfig sim_cfg_ = loadTestSimConfig();
  mav_model::State state_              = hoverStateAt({0.0, 0.0, 10.0});
  std::unique_ptr<DynamicTrajectoryGenerator> gen_;

  void SetUp() override {
    auto cfg = DynamicTrajectoryGenerator::loadConfigFromYaml(
        repoPath("configs/generators/config_dynamic.yaml"));
    gen_ = std::make_unique<DynamicTrajectoryGenerator>(cfg);
    gen_->initialize(state_, sim_cfg_);
  }
};

TEST_F(DynamicTrajectoryGeneratorTest, ProducesFiniteReferencesOver50Steps) {
  const Eigen::Vector3d target(5.0, 0.0, 10.0);
  gen_->onWaypointChanged(target, state_, 0.0);

  for (int k = 0; k < 50; ++k) {
    const double t = 0.01 * k;
    gen_->update(t, state_);
    const auto s = gen_->evaluate(t);
    ASSERT_TRUE(s.position.allFinite()) << "step " << k;
  }
}

TEST_F(DynamicTrajectoryGeneratorTest, AdvertisesPositionField) {
  const auto mask = gen_->providedReferenceFields();
  EXPECT_TRUE(
      mpc_examples::framework::hasField(mask, mpc_examples::framework::ReferenceField::kPosition));
}
