// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <memory>

#include "framework/trajectory_generator_base.hpp"
#include "generators/waypoint_reference_generator.hpp"
#include "test_helpers.hpp"

using mpc_examples::adapters::WaypointReferenceGenerator;
using mpc_examples::testing::hoverStateAt;
using mpc_examples::testing::loadTestSimConfig;
using mpc_examples::testing::repoPath;

class WaypointReferenceGeneratorTest : public ::testing::Test {
protected:
  mpc_examples::ExampleConfig sim_cfg_ = loadTestSimConfig();
  mav_model::State state_              = hoverStateAt({0.0, 0.0, 10.0});
  std::unique_ptr<WaypointReferenceGenerator> gen_;

  void SetUp() override {
    auto cfg = WaypointReferenceGenerator::loadConfigFromYaml(
        repoPath("configs/generators/config_waypoints.yaml"));
    gen_ = std::make_unique<WaypointReferenceGenerator>(cfg);
    gen_->initialize(state_, sim_cfg_);
  }
};

TEST_F(WaypointReferenceGeneratorTest, ProducesFiniteReferencesOver50Steps) {
  const Eigen::Vector3d target(10.0, 0.0, 10.0);
  gen_->onWaypointChanged(target, state_, 0.0);

  for (int k = 0; k < 50; ++k) {
    const double t = 0.01 * k;
    gen_->update(t, state_);
    const auto s = gen_->evaluate(t);
    ASSERT_TRUE(s.position.allFinite()) << "step " << k;
    ASSERT_TRUE(std::isfinite(s.yaw)) << "step " << k;
  }
}

TEST_F(WaypointReferenceGeneratorTest, AdvertisesPositionField) {
  const auto mask = gen_->providedReferenceFields();
  EXPECT_TRUE(
      mpc_examples::framework::hasField(mask, mpc_examples::framework::ReferenceField::kPosition));
}
