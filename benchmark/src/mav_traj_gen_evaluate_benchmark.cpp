// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file mav_traj_gen_evaluate_benchmark.cpp
 * @brief Google Benchmark of the mav_trajectory_generation **sampling** path
 *        (`mav_trajectory_generation_cpp::TrajectoryGenerator::evaluate(t)`).
 *
 * Companion to `BM_MavTrajGenGenerate` (which measures the optimisation
 * cost). Each control tick of a downstream consumer pays one `evaluate`
 * call to fetch (pos, vel, acc); this benchmark isolates that cost.
 *
 * The query time sweeps the segment and wraps; the underlying polynomial
 * clamps so off-range queries are well-defined. Reported in nanoseconds.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include <benchmark/benchmark.h>

#include <vector>

#include "mav_benchmark/bench_common.hpp"
#include "mav_benchmark/bench_loaders_mav_traj_gen.hpp"
#include "mav_trajectory_generation_cpp/trajectory_generator.hpp"
#include "mav_trajectory_generation_cpp/types.hpp"

namespace {

void BM_MavTrajGenEvaluate(benchmark::State& state) {
  const mav_trajectory_generation_cpp::GeneratorConfig cfg =
      mav_benchmark::loadMavTrajGenConfig(mav_benchmark::kTrajYaml);
  mav_trajectory_generation_cpp::TrajectoryGenerator generator(cfg);

  const Eigen::Vector3d p0(0.0, 0.0, 1.0);
  const Eigen::Vector3d p1(mav_benchmark::kCarrotDistance, 0.0, 1.0);
  std::vector<mav_trajectory_generation_cpp::Waypoint> wps;
  wps.reserve(3);
  wps.emplace_back(p0);
  wps.emplace_back(Eigen::Vector3d(0.5 * (p0 + p1)));
  wps.emplace_back(p1);
  if (!generator.generate(wps, mav_benchmark::kMaxSpeed)) {
    state.SkipWithError("mav_traj_gen generate() failed; cannot benchmark evaluate().");
    return;
  }
  const double T = generator.duration();
  if (T <= 0.0) {
    state.SkipWithError("mav_traj_gen trajectory has non-positive duration.");
    return;
  }

  const double dt = 1.0e-3;
  double t = 0.0;
  for (auto _ : state) {
    const mav_trajectory_generation_cpp::TrajectorySample s = generator.evaluate(t);
    benchmark::DoNotOptimize(s.position);
    benchmark::DoNotOptimize(s.velocity);
    benchmark::DoNotOptimize(s.acceleration);
    t += dt;
    if (t > T) {
      t = 0.0;
    }
  }
}

BENCHMARK(BM_MavTrajGenEvaluate)->Unit(benchmark::kNanosecond)->Threads(1)->Repetitions(10);

}  // namespace
