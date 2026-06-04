// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file main.cpp
 * @brief Google Benchmark entry point for the mav_examples performance suite.
 *
 * The individual benchmarks (P-MPC / SSA-P-MPC solve, GCOPTER,
 * mav_trajectory_generation, reference adaptation) self-register via the
 * BENCHMARK() macro in their respective translation units. A single binary
 * links them all (the P-MPC `acados_mpc` and SSA `acados_ssa_mpc` types are
 * distinct namespaces / acados models and coexist, exactly as in
 * `position_examples`). We provide our own main instead of BENCHMARK_MAIN()
 * so the benchmarks from several TUs share one executable, while still
 * exposing the full Google Benchmark CLI (--benchmark_filter,
 * --benchmark_repetitions, --benchmark_out, ...).
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include <benchmark/benchmark.h>

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
