// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

#include "framework/factories.hpp"
#include "framework/unified_mcap_logger.hpp"
#include "framework/waypoints_simulator.hpp"
#include "mav_simulator/simulator_yaml.hpp"
#include "test_helpers.hpp"

using mpc_examples::framework::ControllerKeys;
using mpc_examples::framework::GeneratorKeys;
using mpc_examples::framework::makeController;
using mpc_examples::framework::makeGenerator;
using mpc_examples::framework::RunMetadata;
using mpc_examples::framework::WaypointsSimulator;
using mpc_examples::testing::loadTestSimConfig;
using mpc_examples::testing::repoPath;

namespace {

mav_simulator::SimulatorParameters loadTestSimParams() {
  return mav_simulator::loadSimulatorParametersFromYaml(
      repoPath("configs/simulation/config_simulator.yaml"));
}

RunMetadata makeMeta() {
  RunMetadata m;
  m.controller_name = "pid";
  m.generator_name  = "waypoints";
  m.run_id          = "unit_test";
  m.language        = "cpp";
  return m;
}

}  // namespace

TEST(WaypointsSimulatorTest, RejectsNullController) {
  auto cfg = loadTestSimConfig();
  auto sp  = loadTestSimParams();
  auto gen = makeGenerator(GeneratorKeys::kWaypoints,
                           repoPath("configs/generators/config_waypoints.yaml"));
  EXPECT_THROW(WaypointsSimulator(nullptr, std::move(gen), cfg, sp, "", makeMeta()),
               std::invalid_argument);
}

TEST(WaypointsSimulatorTest, RejectsNullGenerator) {
  auto cfg  = loadTestSimConfig();
  auto sp   = loadTestSimParams();
  auto pid  = makeController(ControllerKeys::kPid,
                             repoPath("configs/controllers/config_pid.yaml"), false);
  EXPECT_THROW(WaypointsSimulator(std::move(pid), nullptr, cfg, sp, "", makeMeta()),
               std::invalid_argument);
}

TEST(WaypointsSimulatorTest, RejectsEmptyWaypointList) {
  auto cfg = loadTestSimConfig();
  cfg.waypoints.clear();
  auto sp  = loadTestSimParams();
  auto pid = makeController(ControllerKeys::kPid,
                            repoPath("configs/controllers/config_pid.yaml"), false);
  auto gen = makeGenerator(GeneratorKeys::kWaypoints,
                           repoPath("configs/generators/config_waypoints.yaml"));
  EXPECT_THROW(WaypointsSimulator(std::move(pid), std::move(gen), cfg, sp, "", makeMeta()),
               std::invalid_argument);
}

TEST(WaypointsSimulatorTest, RunCompletesAndReportsBenchmarkStats) {
  auto cfg          = loadTestSimConfig();
  cfg.sim_time      = 0.5;     // tight; we just check the loop wires up.
  cfg.hover_time    = 0.0;
  cfg.silent        = true;
  cfg.benchmark     = true;    // skip on-disk logging.
  cfg.output_format = "mcap";
  auto sp           = loadTestSimParams();
  auto pid          = makeController(ControllerKeys::kPid,
                                     repoPath("configs/controllers/config_pid.yaml"), false);
  auto gen          = makeGenerator(GeneratorKeys::kWaypoints,
                                    repoPath("configs/generators/config_waypoints.yaml"));

  // Empty output_csv disables logging — we are just exercising the run loop.
  WaypointsSimulator sim(std::move(pid), std::move(gen), cfg, sp, /*output_csv=*/"", makeMeta());
  ASSERT_NO_THROW(sim.run());

  const auto& stats = sim.benchmarkStats();
  EXPECT_GT(stats.simulated_time_s, 0.0);
  EXPECT_GT(stats.controller_steps, 0u);
  EXPECT_GT(stats.indi_steps, 0u);
  EXPECT_GE(stats.controller_mean_us, 0.0);
  EXPECT_TRUE(std::isfinite(stats.tracking_rmse_m));
}

// Same exit-time double-free as in test_factories_trajectory: skip the
// destructor chain to avoid CTest reporting a teardown crash after every
// test passes.
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const int rc = RUN_ALL_TESTS();
  std::_Exit(rc);
}
