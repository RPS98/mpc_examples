// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "framework/waypoint_scheduler.hpp"

using mpc_examples::framework::WaypointScheduler;

namespace {

constexpr double kTolerance = 1e-9;

}  // namespace

TEST(WaypointSchedulerTest, ComputesSwitchTimesFromDistanceAndSpeed) {
  WaypointScheduler scheduler;
  const std::vector<Eigen::Vector3d> wps = {
      Eigen::Vector3d(10.0, 0.0, 0.0),
      Eigen::Vector3d(10.0, 5.0, 0.0),
      Eigen::Vector3d(10.0, 5.0, 5.0),
  };
  const Eigen::Vector3d start(0.0, 0.0, 0.0);
  const double max_speed = 5.0;
  const double margin    = 1.0;

  scheduler.initialize(wps, start, max_speed, margin);

  // First hop covers 10 m at 5 m/s (= 2 s) + 1 s settle margin.
  EXPECT_NEAR(scheduler.switchTime(0), 10.0 / 5.0 + 1.0, kTolerance);
  // Second hop adds 5 m / 5 m/s + 1 s.
  EXPECT_NEAR(scheduler.switchTime(1), scheduler.switchTime(0) + 5.0 / 5.0 + 1.0, kTolerance);
  // Third hop adds 5 m / 5 m/s + 1 s.
  EXPECT_NEAR(scheduler.switchTime(2), scheduler.switchTime(1) + 5.0 / 5.0 + 1.0, kTolerance);
  EXPECT_EQ(scheduler.size(), wps.size());
}

TEST(WaypointSchedulerTest, TickAdvancesIndexAndFlagsTransitionsExactlyOnce) {
  WaypointScheduler scheduler;
  const std::vector<Eigen::Vector3d> wps = {
      Eigen::Vector3d(1.0, 0.0, 0.0),
      Eigen::Vector3d(2.0, 0.0, 0.0),
  };
  scheduler.initialize(wps, Eigen::Vector3d::Zero(), 1.0, 0.0);

  // Before the first switch time, no transition yet (hold the first waypoint).
  auto r0 = scheduler.tick(0.5);
  EXPECT_FALSE(r0.waypoint_changed);
  EXPECT_EQ(r0.active_index, 0);
  EXPECT_FALSE(r0.finished);

  // Crossing the first switch time activates the second waypoint.
  auto r1 = scheduler.tick(scheduler.switchTime(0) + 1e-3);
  EXPECT_TRUE(r1.waypoint_changed);
  EXPECT_EQ(r1.active_index, 1);

  // Re-querying at the same time does NOT re-emit the transition.
  auto r2 = scheduler.tick(scheduler.switchTime(0) + 2e-3);
  EXPECT_FALSE(r2.waypoint_changed);
  EXPECT_EQ(r2.active_index, 1);

  // Past the last switch time the scheduler reports finished.
  auto r3 = scheduler.tick(scheduler.switchTime(1) + 1e-3);
  EXPECT_TRUE(r3.finished);
}

TEST(WaypointSchedulerTest, RejectsInvalidInputs) {
  WaypointScheduler scheduler;
  EXPECT_THROW(scheduler.initialize({}, Eigen::Vector3d::Zero(), 1.0, 0.0), std::invalid_argument);
  EXPECT_THROW(scheduler.initialize({Eigen::Vector3d::UnitX()}, Eigen::Vector3d::Zero(), 0.0, 0.0),
               std::invalid_argument);
  EXPECT_THROW(scheduler.initialize({Eigen::Vector3d::UnitX()}, Eigen::Vector3d::Zero(), 1.0, -1.0),
               std::invalid_argument);
  // scheduler_speed_factor must lie in (0, 1].
  EXPECT_THROW(scheduler.initialize({Eigen::Vector3d::UnitX()}, Eigen::Vector3d::Zero(), 1.0, 0.0,
                                    /*scheduler_speed_factor=*/0.0),
               std::invalid_argument);
  EXPECT_THROW(scheduler.initialize({Eigen::Vector3d::UnitX()}, Eigen::Vector3d::Zero(), 1.0, 0.0,
                                    /*scheduler_speed_factor=*/1.5),
               std::invalid_argument);
  EXPECT_THROW(scheduler.initialize({Eigen::Vector3d::UnitX()}, Eigen::Vector3d::Zero(), 1.0, 0.0,
                                    /*scheduler_speed_factor=*/-0.1),
               std::invalid_argument);
}

TEST(WaypointSchedulerTest, SpeedFactorInflatesSegmentDurations) {
  WaypointScheduler scheduler;
  const std::vector<Eigen::Vector3d> wps = {
      Eigen::Vector3d(10.0, 0.0, 0.0),
      Eigen::Vector3d(10.0, 5.0, 0.0),
  };
  const Eigen::Vector3d start(0.0, 0.0, 0.0);
  const double max_speed = 5.0;
  const double margin    = 1.0;
  const double factor    = 0.5;

  scheduler.initialize(wps, start, max_speed, margin, factor);

  // First hop: 10 m / (5 m/s * 0.5) + 1 s = 5 s.
  EXPECT_NEAR(scheduler.switchTime(0), 10.0 / (5.0 * 0.5) + 1.0, kTolerance);
  // Second hop adds 5 m / (5 m/s * 0.5) + 1 s = 3 s.
  EXPECT_NEAR(scheduler.switchTime(1), scheduler.switchTime(0) + 5.0 / (5.0 * 0.5) + 1.0,
              kTolerance);
}

TEST(WaypointSchedulerTest, DefaultFactorMatchesLegacyHeuristic) {
  // Omitting the factor (or passing 1.0) reproduces the original
  // distance/max_speed heuristic. Guards against accidental ABI change.
  WaypointScheduler s_default;
  WaypointScheduler s_one;
  const std::vector<Eigen::Vector3d> wps = {Eigen::Vector3d(3.0, 4.0, 0.0)};
  s_default.initialize(wps, Eigen::Vector3d::Zero(), 1.0, 0.5);
  s_one.initialize(wps, Eigen::Vector3d::Zero(), 1.0, 0.5, 1.0);
  EXPECT_NEAR(s_default.switchTime(0), s_one.switchTime(0), kTolerance);
  EXPECT_NEAR(s_default.switchTime(0), 5.0 + 0.5, kTolerance);
}

TEST(WaypointSchedulerTest, SingleWaypointMissionIsFinishedAfterFirstSwitchTime) {
  WaypointScheduler scheduler;
  const std::vector<Eigen::Vector3d> wps = {Eigen::Vector3d(2.0, 0.0, 0.0)};
  scheduler.initialize(wps, Eigen::Vector3d::Zero(), 1.0, 0.5);

  auto r_before = scheduler.tick(0.0);
  EXPECT_FALSE(r_before.finished);
  EXPECT_EQ(r_before.active_index, 0);

  auto r_after = scheduler.tick(scheduler.switchTime(0) + 1e-3);
  EXPECT_TRUE(r_after.finished);
  EXPECT_EQ(r_after.active_index, 0);  // The single waypoint stays active.
}
