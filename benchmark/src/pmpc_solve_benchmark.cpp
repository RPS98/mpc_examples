// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file pmpc_solve_benchmark.cpp
 * @brief Google Benchmark of the Position-MPC OCP solve
 *        (`acados_mpc::MPC::solve()`).
 *
 * Methodology: an ideal plant-free closed loop. After each solve the state is
 * replaced by the solver's one-step-ahead prediction
 * (`predicted_state_stage1`) and the reference is a carrot held a fixed
 * `kCarrotDistance` ahead of the current position along +x, so the recorded
 * window captures sustained warm-started cruise at a constant, non-trivial
 * tracking error. The per-tick reference rebuild (position+velocity ramp) and
 * the state feedback are sub-microsecond against the millisecond-scale solve,
 * so the whole tick is timed without PauseTiming (which would add comparable
 * overhead). The pure acados solver time (`time_tot`) is reported as a custom
 * counter alongside the harness wall/CPU time.
 *
 * Threading: acados is compiled with OpenMP. The benchmark harness is pinned
 * to one thread; control acados' own threads via OMP_NUM_THREADS in the
 * environment (the launcher sets it to 1 by default).
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include <benchmark/benchmark.h>

#include <array>
#include <cmath>
#include <stdexcept>

#include "acados_mpc/acados_mpc.hpp"
#include "acados_mpc/acados_mpc_yaml.hpp"
#include "mav_benchmark/bench_common.hpp"

namespace {

/// Reads the soft speed bound `constraints.uh[0]` (= max_speed²) or 0.
double readUhDefault(acados_mpc::MPC& mpc) {
  if constexpr (acados_mpc::NonlinearConstraintBounds::Nh == 0) {
    (void)mpc;
    return 0.0;
  } else {
    return mpc.getNonlinearConstraintBounds()->getUhArray()[0];
  }
}

/**
 * @brief Position+velocity reference ramp (as2_position_mpc_plugin style).
 *
 * Stage k: s_k = min((k+1)·v_ref·dt_h, L); the per-stage velocity feed-forward
 * v_stage = (s_{k+1}-s_k)/dt_h is accumulated into @p sink because the
 * mav_examples Position-MPC OCP exposes a position-only reference
 * (`acados_mpc::OnlineParameters` has no setDesiredVelocity).
 */
void setProgressiveReferencesPosVel(acados_mpc::MPCData* data,
                                    const std::array<double, 3>& current,
                                    const std::array<double, 3>& goal, double v_ref, double dt_h,
                                    int N, std::array<double, 3>* sink) {
  const std::array<double, 3> delta = {goal[0] - current[0], goal[1] - current[1],
                                       goal[2] - current[2]};
  const double distance =
      std::sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
  if (distance < 1e-9) {
    data->p_params.setDesiredPosition(goal);
    return;
  }
  const std::array<double, 3> dir = {delta[0] / distance, delta[1] / distance,
                                     delta[2] / distance};
  for (int k = 0; k <= N; ++k) {
    const double s_k = std::min((k + 1) * v_ref * dt_h, distance);
    const double s_kp1 = std::min((k + 2) * v_ref * dt_h, distance);
    const std::array<double, 3> stage_pos = {current[0] + s_k * dir[0], current[1] + s_k * dir[1],
                                             current[2] + s_k * dir[2]};
    const double v_stage = (s_kp1 - s_k) / dt_h;
    (*sink)[0] += v_stage * dir[0];
    (*sink)[1] += v_stage * dir[1];
    (*sink)[2] += v_stage * dir[2];
    data->p_params.setDesiredPosition(stage_pos, k);
  }
}

void BM_PmpcSolve(benchmark::State& state) {
  acados_mpc::MPC mpc;
  acados_mpc::configureMpcFromYaml(mpc, mav_benchmark::kMpcYaml);

  const double pct = mav_benchmark::readMaxVelPercentage(mav_benchmark::kMpcYaml, 0.95);
  const double uh = readUhDefault(mpc);
  const double v_ref = uh > 0.0 ? std::sqrt(uh) * pct : mav_benchmark::kMaxSpeed * pct;
  const int N = mpc.getPredictionSteps();
  const double dt_h = mpc.getPredictionTimeStep();

  acados_mpc::MPCData* data = mpc.getData();
  data->state.setPosition({0.0, 0.0, 0.0});
  data->state.setOrientation({1.0, 0.0, 0.0, 0.0});
  data->state.setLinearVelocity({0.0, 0.0, 0.0});
  data->p_params.setDesiredOrientation({1.0, 0.0, 0.0, 0.0});

  const auto* ptrs = mpc.getAcadosSolverPointers();
  std::array<double, 3> sink = {0.0, 0.0, 0.0};
  double acados_us_sum = 0.0;
  int failures = 0;

  for (auto _ : state) {
    const std::array<double, 3> pos = data->state.getPosition();
    const std::array<double, 3> goal = {pos[0] + mav_benchmark::kCarrotDistance, pos[1], pos[2]};
    setProgressiveReferencesPosVel(data, pos, goal, v_ref, dt_h, N, &sink);

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

  benchmark::DoNotOptimize(sink);
  state.counters["acados_us"] =
      benchmark::Counter(acados_us_sum, benchmark::Counter::kAvgIterations);
  if (failures > 0) {
    state.counters["solve_failures"] = failures;
  }
}

BENCHMARK(BM_PmpcSolve)->Unit(benchmark::kMicrosecond)->Threads(1)->Repetitions(10);

}  // namespace
