// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file tmpc_solve_benchmark.cpp
 * @brief Google Benchmark of the Trajectory-MPC OCP solve
 *        (`acados_mpc::MPC::solve()` from `libs/acados_trajectory_mpc`).
 *
 * Lives in its own binary (`run_benchmarks_trajectory`) because both the
 * trajectory MPC and the position MPC expose `acados_mpc::MPC` from the
 * same `acados_mpc` namespace and their acados C symbols collide at link
 * time. Co-existing them inside `position_examples` works because the
 * showcase does not include the trajectory MPC adapter on that side; we
 * apply the same separation here.
 *
 * Methodology mirrors `BM_PmpcSolve`: plant-free closed loop with
 * one-step-ahead prediction feedback and a moving carrot at
 * `kCarrotDistance` ahead along +x. The trajectory MPC takes a full
 * (position, orientation, velocity) reference per stage, so the
 * benchmark fills the horizon with a constant-velocity ramp toward the
 * carrot and broadcasts a zero-yaw orientation. The acados-internal
 * `time_tot` is reported as a custom counter.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cmath>

#include "acados_mpc/acados_mpc.hpp"
#include "acados_mpc/acados_mpc_yaml.hpp"
#include "mav_benchmark/bench_common.hpp"

namespace {

/// Fill the prediction horizon with a constant-velocity ramp toward `goal`.
void setProgressiveReferencesPosVelOrient(acados_mpc::MPCData* data,
                                          const std::array<double, 3>& current,
                                          const std::array<double, 3>& goal, double v_ref,
                                          double dt_h, int N) {
  const std::array<double, 3> delta = {goal[0] - current[0], goal[1] - current[1],
                                       goal[2] - current[2]};
  const double distance =
      std::sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
  std::array<double, 3> dir{1.0, 0.0, 0.0};
  if (distance > 1e-9) {
    dir = {delta[0] / distance, delta[1] / distance, delta[2] / distance};
  }
  const std::array<double, 4> orient{1.0, 0.0, 0.0, 0.0};
  for (int k = 0; k <= N; ++k) {
    const double s_k = std::min((k + 1) * v_ref * dt_h, distance);
    const std::array<double, 3> stage_pos = {current[0] + s_k * dir[0], current[1] + s_k * dir[1],
                                             current[2] + s_k * dir[2]};
    const std::array<double, 3> stage_vel = (s_k < distance)
                                                ? std::array<double, 3>{v_ref * dir[0],
                                                                        v_ref * dir[1],
                                                                        v_ref * dir[2]}
                                                : std::array<double, 3>{0.0, 0.0, 0.0};
    data->p_params.setDesiredPosition(stage_pos, k);
    data->p_params.setDesiredOrientation(orient, k);
    data->p_params.setDesiredVelocity(stage_vel, k);
  }
}

void BM_TmpcSolve(benchmark::State& state) {
  acados_mpc::MPC mpc;
  acados_mpc::configureMpcFromYaml(mpc, mav_benchmark::kTrajectoryMpcYaml);

  const double pct = mav_benchmark::readMaxVelPercentage(mav_benchmark::kTrajectoryMpcYaml, 0.95);
  const double v_ref = mav_benchmark::kMaxSpeed * pct;
  const int N = mpc.getPredictionSteps();
  const double dt_h = mpc.getPredictionTimeStep();

  acados_mpc::MPCData* data = mpc.getData();
  // Start in steady-state cruise (v = v_ref along +x). A cold start at v=0
  // against a constant-velocity reference is physically infeasible inside one
  // prediction horizon (requires a = v_ref/dt_h ≈ 10 m/s²) and yields
  // ACADOS_MINSTEP failures that snowball through the closed loop.
  data->state.setPosition({0.0, 0.0, 0.0});
  data->state.setOrientation({1.0, 0.0, 0.0, 0.0});
  data->state.setLinearVelocity({v_ref, 0.0, 0.0});

  const auto* ptrs = mpc.getAcadosSolverPointers();
  double acados_us_sum = 0.0;
  int failures = 0;

  // Warm-up the solver so the cost / KKT residual converges to its
  // steady-state regime before the harness starts timing. ~50 ticks is enough
  // for SQP_RTI with this OCP (N ~= 20, dt_h ~= 0.05 s).
  for (int i = 0; i < 50; ++i) {
    const std::array<double, 3> pos = data->state.getPosition();
    const std::array<double, 3> goal = {pos[0] + mav_benchmark::kCarrotDistance, pos[1], pos[2]};
    setProgressiveReferencesPosVelOrient(data, pos, goal, v_ref, dt_h, N);
    if (mpc.solve() == 0) {
      data->state.setPosition(data->predicted_state_stage1.getPosition());
      data->state.setOrientation(data->predicted_state_stage1.getOrientation());
      data->state.setLinearVelocity(data->predicted_state_stage1.getLinearVelocity());
    }
  }

  for (auto _ : state) {
    const std::array<double, 3> pos = data->state.getPosition();
    const std::array<double, 3> goal = {pos[0] + mav_benchmark::kCarrotDistance, pos[1], pos[2]};
    setProgressiveReferencesPosVelOrient(data, pos, goal, v_ref, dt_h, N);

    const int status = mpc.solve();
    benchmark::DoNotOptimize(status);
    if (status != 0) {
      ++failures;
    }

    double time_tot = 0.0;
    ocp_nlp_get(ptrs->nlp_solver, "time_tot", &time_tot);
    acados_us_sum += time_tot * 1e6;

    if (status == 0) {
      data->state.setPosition(data->predicted_state_stage1.getPosition());
      data->state.setOrientation(data->predicted_state_stage1.getOrientation());
      data->state.setLinearVelocity(data->predicted_state_stage1.getLinearVelocity());
    } else {
      // Reset to the steady-state cruise so a transient solver miss does
      // not cascade through the rest of the loop and pollute the timing.
      data->state.setPosition({0.0, 0.0, 0.0});
      data->state.setOrientation({1.0, 0.0, 0.0, 0.0});
      data->state.setLinearVelocity({v_ref, 0.0, 0.0});
    }
  }

  state.counters["acados_us"] =
      benchmark::Counter(acados_us_sum, benchmark::Counter::kAvgIterations);
  if (failures > 0) {
    state.counters["solve_failures"] = failures;
  }
}

BENCHMARK(BM_TmpcSolve)->Unit(benchmark::kMicrosecond)->Threads(1)->Repetitions(10);

}  // namespace
