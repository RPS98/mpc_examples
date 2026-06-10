// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file ssa_pmpc_solve_benchmark.cpp
 * @brief Google Benchmark of the SSA-Position-MPC OCP solve
 *        (`acados_ssa_mpc::MPC::solve()`).
 *
 * Same ideal plant-free closed-loop methodology as the P-MPC benchmark: the
 * state is advanced to the solver's one-step-ahead prediction and the
 * set-point is a carrot held `kCarrotDistance` ahead along +x. The
 * steady-state-aware formulation only needs the constant set-point — the
 * admissible, speed-limited approach emerges from the artificial steady-state
 * target. The acados-internal `time_tot` is reported as a custom counter.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include <benchmark/benchmark.h>

#include <array>

#include "acados_ssa_mpc/acados_mpc.hpp"
#include "acados_ssa_mpc/acados_mpc_yaml.hpp"
#include "mav_benchmark/bench_common.hpp"

namespace {

void BM_SsaPmpcSolve(benchmark::State& state) {
  acados_ssa_mpc::MPC mpc;
  acados_ssa_mpc::configureMpcFromYaml(mpc, mav_benchmark::kSsaYaml);

  acados_ssa_mpc::MPCData* data = mpc.getData();
  data->state.setPosition({0.0, 0.0, 0.0});
  data->state.setOrientation({1.0, 0.0, 0.0, 0.0});
  data->state.setLinearVelocity({0.0, 0.0, 0.0});
  data->p_params.setDesiredOrientation({1.0, 0.0, 0.0, 0.0});

  const auto* ptrs = mpc.getAcadosSolverPointers();
  double acados_us_sum = 0.0;
  int failures = 0;

  // Two-phase warm-up. Phase A: reference glued to the state (trivial QP).
  // Phase B: carrot regime used by the timed loop.
  for (int i = 0; i < 500; ++i) {
    const std::array<double, 3> pos = data->state.getPosition();
    data->p_params.setDesiredPosition(pos);
    if (mpc.solve() == 0) {
      data->state.setPosition(data->predicted_state_stage1.getPosition());
      data->state.setOrientation(data->predicted_state_stage1.getOrientation());
      data->state.setLinearVelocity(data->predicted_state_stage1.getLinearVelocity());
    }
  }
  for (int i = 0; i < 500; ++i) {
    const std::array<double, 3> pos = data->state.getPosition();
    data->p_params.setDesiredPosition({pos[0] + mav_benchmark::kCarrotDistance, pos[1], pos[2]});
    if (mpc.solve() == 0) {
      data->state.setPosition(data->predicted_state_stage1.getPosition());
      data->state.setOrientation(data->predicted_state_stage1.getOrientation());
      data->state.setLinearVelocity(data->predicted_state_stage1.getLinearVelocity());
    }
  }

  for (auto _ : state) {
    const std::array<double, 3> pos = data->state.getPosition();
    data->p_params.setDesiredPosition({pos[0] + mav_benchmark::kCarrotDistance, pos[1], pos[2]});

    const int status = mpc.solve();
    benchmark::DoNotOptimize(status);
    if (status != 0) {
      ++failures;
    }

    double time_tot = 0.0;
    ocp_nlp_get(ptrs->nlp_solver, "time_tot", &time_tot);
    acados_us_sum += time_tot * 1e6;

    data->state.setPosition(data->predicted_state_stage1.getPosition());
    data->state.setOrientation(data->predicted_state_stage1.getOrientation());
    data->state.setLinearVelocity(data->predicted_state_stage1.getLinearVelocity());
  }

  state.counters["acados_us"] =
      benchmark::Counter(acados_us_sum, benchmark::Counter::kAvgIterations);
  if (failures > 0) {
    state.counters["solve_failures"] = failures;
  }
}

BENCHMARK(BM_SsaPmpcSolve)->Unit(benchmark::kMicrosecond)->Threads(1)->Repetitions(10);

}  // namespace
