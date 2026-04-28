// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <string>

#include "framework/parallel_runner.hpp"

using mpc_examples::framework::casePrefix;
using mpc_examples::framework::runScopedCasesParallel;

TEST(ParallelRunnerTest, CasePrefixIsHumanReadableAndOneIndexed) {
  EXPECT_EQ(casePrefix(0, 3, "pid", "waypoints"), "[1/3 pid\xC3\x97waypoints] ");
  EXPECT_EQ(casePrefix(2, 3, "mpc_trajectory", "gcopter"), "[3/3 mpc_trajectory\xC3\x97gcopter] ");
}

TEST(ParallelRunnerTest, RunScopedCasesParallelDispatchesEveryIndex) {
  constexpr std::size_t kNumWorkers = 8;
  std::atomic<std::size_t> calls{0};
  std::atomic<std::size_t> bitmask{0};

  runScopedCasesParallel(kNumWorkers, [&](std::size_t i) {
    calls.fetch_add(1, std::memory_order_relaxed);
    bitmask.fetch_or(static_cast<std::size_t>(1u) << i, std::memory_order_relaxed);
  });

  EXPECT_EQ(calls.load(), kNumWorkers);
  EXPECT_EQ(bitmask.load(), (static_cast<std::size_t>(1u) << kNumWorkers) - 1u);
}

TEST(ParallelRunnerTest, RunScopedCasesParallelHandlesZeroWorkers) {
  std::atomic<int> calls{0};
  runScopedCasesParallel(0, [&](std::size_t) { calls.fetch_add(1); });
  EXPECT_EQ(calls.load(), 0);
}

TEST(ParallelRunnerTest, RunScopedCasesParallelPreservesIndexInsideWorkers) {
  constexpr std::size_t kNumWorkers = 4;
  std::array<std::size_t, kNumWorkers> seen{};

  runScopedCasesParallel(kNumWorkers, [&](std::size_t i) { seen[i] = i + 100; });

  for (std::size_t i = 0; i < kNumWorkers; ++i) {
    EXPECT_EQ(seen[i], i + 100);
  }
}
