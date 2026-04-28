// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "framework/factories.hpp"
#include "test_helpers.hpp"

using mpc_examples::framework::ControllerKeys;
using mpc_examples::framework::defaultControllerConfigPath;
using mpc_examples::framework::defaultGeneratorConfigPath;
using mpc_examples::framework::GeneratorKeys;
using mpc_examples::framework::makeController;
using mpc_examples::framework::makeGenerator;

TEST(FactoriesPositionTest, ResolvesDefaultPathsForPositionScope) {
  EXPECT_EQ(std::string(defaultControllerConfigPath(ControllerKeys::kPid)),
            "configs/controllers/config_pid.yaml");
  EXPECT_EQ(std::string(defaultControllerConfigPath(ControllerKeys::kMpcPosition)),
            "configs/controllers/config_mpc.yaml");
  EXPECT_EQ(std::string(defaultGeneratorConfigPath(GeneratorKeys::kWaypoints)),
            "configs/generators/config_waypoints.yaml");
}

TEST(FactoriesPositionTest, RejectsUnknownControllerKey) {
  EXPECT_THROW(defaultControllerConfigPath("does_not_exist"), std::invalid_argument);
  EXPECT_THROW(makeController("does_not_exist", ""), std::invalid_argument);
}

TEST(FactoriesPositionTest, RejectsUnknownGeneratorKey) {
  EXPECT_THROW(defaultGeneratorConfigPath("does_not_exist"), std::invalid_argument);
  EXPECT_THROW(makeGenerator("does_not_exist", ""), std::invalid_argument);
}

TEST(FactoriesPositionTest, BuildsPidAndWaypointsAdaptersFromYaml) {
  using mpc_examples::testing::repoPath;
  auto pid = makeController(ControllerKeys::kPid,
                            repoPath("configs/controllers/config_pid.yaml"));
  ASSERT_NE(pid, nullptr);
  EXPECT_FALSE(pid->name().empty());

  auto wp = makeGenerator(GeneratorKeys::kWaypoints,
                          repoPath("configs/generators/config_waypoints.yaml"));
  ASSERT_NE(wp, nullptr);
  EXPECT_FALSE(wp->name().empty());
}

TEST(FactoriesPositionTest, BuildsMpcPositionFromYaml) {
  using mpc_examples::testing::repoPath;
  auto mpc = makeController(ControllerKeys::kMpcPosition,
                            repoPath("configs/controllers/config_mpc.yaml"));
  ASSERT_NE(mpc, nullptr);
}
