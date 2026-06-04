// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file reference_adaptation_benchmark.cpp
 * @brief Google Benchmark of the prediction-horizon reference adaptation that
 *        runs before every MPC solve, parameterised by the maximum speed.
 *
 * Three variants are registered so they can be compared directly:
 *  - BM_RefsPosOnly   : progressive carrot, positions only (mav_examples
 *                       `MpcPositionController::setProgressiveReferences`).
 *  - BM_RefsPosVel    : progressive carrot with the per-stage velocity
 *                       feed-forward v_stage = (s_{k+1}-s_k)/dt_h
 *                       (aerostack2 `as2_position_mpc_plugin` style).
 *  - BM_RefsSsaSetpoint : the SSA constant set-point (a single
 *                       setDesiredPosition broadcast).
 *
 * These are sub-microsecond; Google Benchmark auto-tunes the iteration count.
 * `benchmark::ClobberMemory()` and `DoNotOptimize` prevent the ramp writes
 * from being elided. This TU deliberately mixes `acados_mpc` and
 * `acados_ssa_mpc` — the same combination compiled by
 * `examples_cpp/src/framework/factories_position.cpp`.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include <benchmark/benchmark.h>

#include <array>
#include <cmath>

#include "acados_mpc/acados_mpc.hpp"
#include "acados_mpc/acados_mpc_yaml.hpp"
#include "acados_ssa_mpc/acados_mpc.hpp"
#include "acados_ssa_mpc/acados_mpc_yaml.hpp"
#include "mav_benchmark/bench_common.hpp"

namespace {

/// Derives v_ref and the horizon dimensions from a configured P-MPC instance.
struct PmpcHorizon {
  acados_mpc::MPC mpc;
  int N = 0;
  double dt_h = 0.0;
  double v_ref = 0.0;

  PmpcHorizon() {
    acados_mpc::configureMpcFromYaml(mpc, mav_benchmark::kMpcYaml);
    N = mpc.getPredictionSteps();
    dt_h = mpc.getPredictionTimeStep();
    const double pct = mav_benchmark::readMaxVelPercentage(mav_benchmark::kMpcYaml, 0.95);
    double uh = 0.0;
    if constexpr (acados_mpc::NonlinearConstraintBounds::Nh > 0) {
      uh = mpc.getNonlinearConstraintBounds()->getUhArray()[0];
    }
    v_ref = uh > 0.0 ? std::sqrt(uh) * pct : mav_benchmark::kMaxSpeed * pct;
  }
};

constexpr std::array<double, 3> kCurrent = {0.0, 0.0, 0.0};

/// Carrot goal with a tiny per-iteration perturbation (defeats hoisting).
std::array<double, 3> goalFor(long i) {
  const double dx = 1e-3 * static_cast<double>(i & 1023);
  return {mav_benchmark::kCarrotDistance + dx, 0.0, 0.0};
}

void BM_RefsPosOnly(benchmark::State& state) {
  PmpcHorizon h;
  acados_mpc::MPCData* data = h.mpc.getData();
  long i = 0;
  for (auto _ : state) {
    const std::array<double, 3> goal = goalFor(i++);
    const std::array<double, 3> d = {goal[0] - kCurrent[0], goal[1] - kCurrent[1],
                                     goal[2] - kCurrent[2]};
    const double dist = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    const std::array<double, 3> dir = {d[0] / dist, d[1] / dist, d[2] / dist};
    for (int k = 0; k <= h.N; ++k) {
      const double s_k = std::min((k + 1) * h.v_ref * h.dt_h, dist);
      const std::array<double, 3> stage_pos = {kCurrent[0] + s_k * dir[0],
                                               kCurrent[1] + s_k * dir[1],
                                               kCurrent[2] + s_k * dir[2]};
      data->p_params.setDesiredPosition(stage_pos, k);
    }
    benchmark::DoNotOptimize(data->p_params.getDesiredPosition(h.N));
    benchmark::ClobberMemory();
  }
}

void BM_RefsPosVel(benchmark::State& state) {
  PmpcHorizon h;
  acados_mpc::MPCData* data = h.mpc.getData();
  std::array<double, 3> sink = {0.0, 0.0, 0.0};
  long i = 0;
  for (auto _ : state) {
    const std::array<double, 3> goal = goalFor(i++);
    const std::array<double, 3> d = {goal[0] - kCurrent[0], goal[1] - kCurrent[1],
                                     goal[2] - kCurrent[2]};
    const double dist = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    const std::array<double, 3> dir = {d[0] / dist, d[1] / dist, d[2] / dist};
    for (int k = 0; k <= h.N; ++k) {
      const double s_k = std::min((k + 1) * h.v_ref * h.dt_h, dist);
      const double s_kp1 = std::min((k + 2) * h.v_ref * h.dt_h, dist);
      const std::array<double, 3> stage_pos = {kCurrent[0] + s_k * dir[0],
                                               kCurrent[1] + s_k * dir[1],
                                               kCurrent[2] + s_k * dir[2]};
      const double v_stage = (s_kp1 - s_k) / h.dt_h;
      sink[0] += v_stage * dir[0];
      sink[1] += v_stage * dir[1];
      sink[2] += v_stage * dir[2];
      data->p_params.setDesiredPosition(stage_pos, k);
    }
    benchmark::DoNotOptimize(sink);
    benchmark::ClobberMemory();
  }
}

void BM_RefsSsaSetpoint(benchmark::State& state) {
  acados_ssa_mpc::MPC mpc;
  acados_ssa_mpc::configureMpcFromYaml(mpc, mav_benchmark::kSsaYaml);
  acados_ssa_mpc::MPCData* data = mpc.getData();
  long i = 0;
  for (auto _ : state) {
    const std::array<double, 3> goal = goalFor(i++);
    data->p_params.setDesiredPosition(goal);
    benchmark::DoNotOptimize(data->p_params.getDesiredPosition());
    benchmark::ClobberMemory();
  }
}

BENCHMARK(BM_RefsPosOnly)->Unit(benchmark::kNanosecond)->Threads(1)->Repetitions(10);
BENCHMARK(BM_RefsPosVel)->Unit(benchmark::kNanosecond)->Threads(1)->Repetitions(10);
BENCHMARK(BM_RefsSsaSetpoint)->Unit(benchmark::kNanosecond)->Threads(1)->Repetitions(10);

}  // namespace
