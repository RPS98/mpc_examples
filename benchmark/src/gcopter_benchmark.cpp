// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file gcopter_benchmark.cpp
 * @brief Google Benchmark of the GCOPTER point-to-point trajectory
 *        optimisation (`gcopter_lib::TrajectoryGenerator::generate()`, L-BFGS).
 *
 * GCOPTER is stateless between calls, so every generate() runs the full
 * polytope-SFC + L-BFGS optimisation. A representative two-waypoint hop of
 * length `kCarrotDistance` is solved; the heading is rotated slightly each
 * iteration to avoid trivial degeneracy at constant problem size. The config
 * is parsed directly from config_gcopter.yaml to stay independent of the
 * mav_simulator-backed adapter layer.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include <benchmark/benchmark.h>

#include <cmath>
#include <vector>

#include "gcopter_lib/trajectory_generator.hpp"
#include "gcopter_lib/types.hpp"
#include "mav_benchmark/bench_common.hpp"
#include "mav_benchmark/bench_loaders_gcopter.hpp"

namespace {

void BM_GcopterGenerate(benchmark::State& state) {
  const gcopter_lib::GeneratorConfig cfg =
      mav_benchmark::loadGcopterConfig(mav_benchmark::kGcopterYaml, mav_benchmark::kMaxSpeed);
  gcopter_lib::TrajectoryGenerator generator(cfg);

  long i = 0;
  int failures = 0;
  for (auto _ : state) {
    const double yaw = 0.01 * static_cast<double>(i++);
    std::vector<gcopter_lib::Waypoint> wps(2);
    wps[0].position = Eigen::Vector3d(0.0, 0.0, 1.0);
    wps[1].position = Eigen::Vector3d(mav_benchmark::kCarrotDistance * std::cos(yaw),
                                      mav_benchmark::kCarrotDistance * std::sin(yaw), 1.0);
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

BENCHMARK(BM_GcopterGenerate)->Unit(benchmark::kMicrosecond)->Threads(1)->Repetitions(10);

}  // namespace
