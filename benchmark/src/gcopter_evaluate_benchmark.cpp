// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file gcopter_evaluate_benchmark.cpp
 * @brief Google Benchmark of the GCOPTER trajectory **sampling** path
 *        (`gcopter_lib::TrajectoryGenerator::evaluate(t)`).
 *
 * This is the cost paid every control tick of a trajectory-following
 * controller (T-MPC, PID trajectory): given a pre-generated polynomial,
 * sample the position/velocity/acceleration triple at the current time.
 *
 * Generation cost lives in `BM_GcopterGenerate` — that runs the L-BFGS
 * optimisation; the per-tick evaluation is just a polynomial readout and
 * costs three orders of magnitude less. Reported in nanoseconds.
 *
 * The query time is swept across the segment to defeat constant-time
 * cache effects (`evaluate` clamps so a sweep that overshoots is safe).
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include <benchmark/benchmark.h>

#include <vector>

#include "gcopter_lib/trajectory_generator.hpp"
#include "gcopter_lib/types.hpp"
#include "mav_benchmark/bench_common.hpp"
#include "mav_benchmark/bench_loaders_gcopter.hpp"

namespace {

void BM_GcopterEvaluate(benchmark::State& state) {
  const gcopter_lib::GeneratorConfig cfg =
      mav_benchmark::loadGcopterConfig(mav_benchmark::kGcopterYaml, mav_benchmark::kMaxSpeed);
  gcopter_lib::TrajectoryGenerator generator(cfg);

  std::vector<gcopter_lib::Waypoint> wps(2);
  wps[0].position = Eigen::Vector3d(0.0, 0.0, 1.0);
  wps[1].position = Eigen::Vector3d(mav_benchmark::kCarrotDistance, 0.0, 1.0);
  if (!generator.generate(wps, mav_benchmark::kMaxSpeed)) {
    state.SkipWithError("GCOPTER generate() failed; cannot benchmark evaluate().");
    return;
  }

  const double T = generator.duration();
  if (T <= 0.0) {
    state.SkipWithError("GCOPTER trajectory has non-positive duration.");
    return;
  }

  // Stride that scans the segment in ~1 ms steps (control-tick granularity)
  // and wraps once t > T so the iteration count is unbounded.
  const double dt = 1.0e-3;
  double t = 0.0;
  for (auto _ : state) {
    const gcopter_lib::TrajectorySample s = generator.evaluate(t);
    benchmark::DoNotOptimize(s.position);
    benchmark::DoNotOptimize(s.velocity);
    benchmark::DoNotOptimize(s.acceleration);
    t += dt;
    if (t > T) {
      t = 0.0;
    }
  }
}

BENCHMARK(BM_GcopterEvaluate)->Unit(benchmark::kNanosecond)->Threads(1)->Repetitions(10);

}  // namespace
