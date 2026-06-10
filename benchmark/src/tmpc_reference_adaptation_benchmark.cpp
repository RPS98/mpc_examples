// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file tmpc_reference_adaptation_benchmark.cpp
 * @brief Google Benchmark of the Trajectory-MPC per-tick reference adaptation:
 *        sampling a pre-generated trajectory at the N+1 stage timestamps and
 *        writing `(pos, vel, orient)` into the solver, exactly as a
 *        trajectory-following controller does before every `solve()`.
 *
 * Companion to `BM_RefsPosOnly` / `BM_RefsPosVel` (Position-MPC carrot ramp)
 * and `BM_RefsSsaSetpoint` (SSA constant set-point). The T-MPC consumes a
 * full smooth trajectory, so the reference adaptation includes:
 *
 *   1. ``N + 1`` calls to ``gcopter_lib::TrajectoryGenerator::evaluate(t)``
 *      (already isolated in ``BM_GcopterEvaluate``); and
 *   2. ``N + 1`` writes to ``acados_mpc::OnlineParameters::set{Position,
 *      Velocity, Orientation}(value, stage)``.
 *
 * This benchmark measures the **sum** of both — the steady-state per-tick
 * cost between two consecutive `solve()` calls. The orientation reference
 * is held identity (the trajectory MPC OCP weights yaw with Q[5] but the
 * sampling cost is independent of the value).
 *
 * The carrot phase ``t0`` is advanced one ``dt_h`` per iteration so the
 * trajectory is sampled at a different N+1-window every tick (wrapped
 * inside the segment duration). This defeats trivial caching of the
 * polynomial coefficient pieces and reflects the real online use.
 *
 * Reported in nanoseconds.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include <benchmark/benchmark.h>

#include <array>
#include <vector>

#include "acados_mpc/acados_mpc.hpp"
#include "acados_mpc/acados_mpc_yaml.hpp"
#include "gcopter_lib/trajectory_generator.hpp"
#include "gcopter_lib/types.hpp"
#include "mav_benchmark/bench_common.hpp"
#include "mav_benchmark/bench_loaders_gcopter.hpp"

namespace {

void BM_RefsTmpc(benchmark::State& state) {
  // Trajectory MPC: configure once to learn its prediction horizon shape.
  acados_mpc::MPC mpc;
  acados_mpc::configureMpcFromYaml(mpc, mav_benchmark::kTrajectoryMpcYaml);
  acados_mpc::MPCData* data = mpc.getData();

  const int N = mpc.getPredictionSteps();
  const double dt_h = mpc.getPredictionTimeStep();

  // Pre-generate one gcopter polynomial; the reference adaptation samples
  // it without re-generating, exactly like the online controller does.
  const gcopter_lib::GeneratorConfig cfg =
      mav_benchmark::loadGcopterConfig(mav_benchmark::kGcopterYaml, mav_benchmark::kMaxSpeed);
  gcopter_lib::TrajectoryGenerator generator(cfg);
  std::vector<gcopter_lib::Waypoint> wps(2);
  wps[0].position = Eigen::Vector3d(0.0, 0.0, 1.0);
  wps[1].position = Eigen::Vector3d(mav_benchmark::kCarrotDistance, 0.0, 1.0);
  if (!generator.generate(wps, mav_benchmark::kMaxSpeed)) {
    state.SkipWithError("gcopter generate() failed; cannot benchmark T-MPC reference adaptation.");
    return;
  }
  const double T = generator.duration();
  if (T <= 0.0) {
    state.SkipWithError("gcopter trajectory has non-positive duration.");
    return;
  }

  const std::array<double, 4> orient{1.0, 0.0, 0.0, 0.0};
  double t0 = 0.0;
  for (auto _ : state) {
    for (int k = 0; k <= N; ++k) {
      double tk = t0 + k * dt_h;
      // Wrap so out-of-range queries stay realistic (evaluate clamps anyway,
      // but wrapping keeps the sampled stage_pos distributed along the segment).
      while (tk > T) {
        tk -= T;
      }
      const gcopter_lib::TrajectorySample s = generator.evaluate(tk);
      const std::array<double, 3> stage_pos{s.position.x(), s.position.y(), s.position.z()};
      const std::array<double, 3> stage_vel{s.velocity.x(), s.velocity.y(), s.velocity.z()};
      data->p_params.setDesiredPosition(stage_pos, k);
      data->p_params.setDesiredVelocity(stage_vel, k);
      data->p_params.setDesiredOrientation(orient, k);
    }
    benchmark::DoNotOptimize(data->p_params.getDesiredPosition(N));
    benchmark::ClobberMemory();
    t0 += dt_h;
    if (t0 > T) {
      t0 = 0.0;
    }
  }
}

BENCHMARK(BM_RefsTmpc)->Unit(benchmark::kNanosecond)->Threads(1)->Repetitions(10);

}  // namespace
