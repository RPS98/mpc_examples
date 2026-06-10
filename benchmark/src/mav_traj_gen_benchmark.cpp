// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file mav_traj_gen_benchmark.cpp
 * @brief Google Benchmark of the polynomial trajectory optimisation
 *        (`mav_trajectory_generation_cpp::TrajectoryGenerator::generate()`).
 *
 * As in the showcase adapter, each replan runs generate() on a three-waypoint
 * sequence [current, midpoint, next]; the generator is reusable and resets its
 * internal state per call. The hop heading is rotated slightly each iteration
 * to keep the problem non-degenerate at constant size. The optimisation
 * parameters are parsed directly from config_mav_traj_gen.yaml.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include <benchmark/benchmark.h>

#include <cmath>
#include <vector>

#include "mav_benchmark/bench_common.hpp"
#include "mav_benchmark/bench_loaders_mav_traj_gen.hpp"
#include "mav_trajectory_generation_cpp/trajectory_generator.hpp"
#include "mav_trajectory_generation_cpp/types.hpp"

namespace {

void BM_MavTrajGenGenerate(benchmark::State& state) {
  const mav_trajectory_generation_cpp::GeneratorConfig cfg =
      mav_benchmark::loadMavTrajGenConfig(mav_benchmark::kTrajYaml);
  mav_trajectory_generation_cpp::TrajectoryGenerator generator(cfg);

  long i = 0;
  int failures = 0;
  for (auto _ : state) {
    const double yaw = 0.01 * static_cast<double>(i++);
    const Eigen::Vector3d p0(0.0, 0.0, 1.0);
    const Eigen::Vector3d p1(mav_benchmark::kCarrotDistance * std::cos(yaw),
                             mav_benchmark::kCarrotDistance * std::sin(yaw), 1.0);
    std::vector<mav_trajectory_generation_cpp::Waypoint> wps;
    wps.reserve(3);
    wps.emplace_back(p0);
    wps.emplace_back(Eigen::Vector3d(0.5 * (p0 + p1)));
    wps.emplace_back(p1);

    const bool ok = generator.generate(wps, mav_benchmark::kMaxSpeed);
    benchmark::DoNotOptimize(ok);
    if (!ok) {
      ++failures;
    } else {
      double d = generator.duration();
      benchmark::DoNotOptimize(d);
    }
  }
  if (failures > 0) {
    state.counters["generate_failures"] = failures;
  }
}

BENCHMARK(BM_MavTrajGenGenerate)->Unit(benchmark::kMicrosecond)->Threads(1)->Repetitions(10);

}  // namespace
