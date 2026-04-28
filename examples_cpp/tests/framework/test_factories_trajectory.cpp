// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "framework/factories.hpp"
#include "test_helpers.hpp"

using mpc_examples::framework::ControllerKeys;
using mpc_examples::framework::defaultControllerConfigPath;
using mpc_examples::framework::defaultGeneratorConfigPath;
using mpc_examples::framework::GeneratorKeys;
using mpc_examples::framework::makeController;
using mpc_examples::framework::makeGenerator;

TEST(FactoriesTrajectoryTest, ResolvesDefaultPathsForTrajectoryScope) {
  EXPECT_EQ(std::string(defaultControllerConfigPath(ControllerKeys::kMpcTrajectory)),
            "configs/controllers/config_mpc_trajectory.yaml");
  EXPECT_EQ(std::string(defaultGeneratorConfigPath(GeneratorKeys::kJerkLimited)),
            "configs/generators/config_jerk_limited.yaml");
  EXPECT_EQ(std::string(defaultGeneratorConfigPath(GeneratorKeys::kGcopter)),
            "configs/generators/config_gcopter.yaml");
  EXPECT_EQ(std::string(defaultGeneratorConfigPath(GeneratorKeys::kDynamic)),
            "configs/generators/config_dynamic.yaml");
  EXPECT_EQ(std::string(defaultGeneratorConfigPath(GeneratorKeys::kMavTrajGen)),
            "configs/generators/config_mav_traj_gen.yaml");
}

TEST(FactoriesTrajectoryTest, RejectsUnknownControllerKey) {
  EXPECT_THROW(makeController("mpc_position", ""), std::invalid_argument);  // out of scope here
  EXPECT_THROW(makeController("does_not_exist", ""), std::invalid_argument);
}

TEST(FactoriesTrajectoryTest, BuildsAllFiveTrajectoryGeneratorsFromYaml) {
  using mpc_examples::testing::repoPath;

  for (const auto& [key, path] : std::vector<std::pair<std::string, std::string>>{
           {GeneratorKeys::kWaypoints, "configs/generators/config_waypoints.yaml"},
           {GeneratorKeys::kJerkLimited, "configs/generators/config_jerk_limited.yaml"},
           {GeneratorKeys::kGcopter, "configs/generators/config_gcopter.yaml"},
           {GeneratorKeys::kDynamic, "configs/generators/config_dynamic.yaml"},
           {GeneratorKeys::kMavTrajGen, "configs/generators/config_mav_traj_gen.yaml"}}) {
    auto gen = makeGenerator(key, repoPath(path));
    ASSERT_NE(gen, nullptr) << "key=" << key;
    EXPECT_FALSE(gen->name().empty()) << "key=" << key;
  }
}

TEST(FactoriesTrajectoryTest, BuildsMpcTrajectoryFromYaml) {
  using mpc_examples::testing::repoPath;
  auto mpc = makeController(ControllerKeys::kMpcTrajectory,
                            repoPath("configs/controllers/config_mpc_trajectory.yaml"));
  ASSERT_NE(mpc, nullptr);
}

// The trajectory factory pulls in acados_trajectory_mpc, dynamic_trajectory_generator
// (with an internal worker thread) and the mav_simulator stack into the same
// binary. Their global destructors race at exit, producing
// "double free or corruption" after every test has already passed. Skip the
// destructor chain via std::_Exit so CTest sees the gtest result, not the
// teardown crash. Custom main mirrors gtest_main.cc.
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const int rc = RUN_ALL_TESTS();
  std::_Exit(rc);
}
