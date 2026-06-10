# mav_examples — C++ performance benchmarks

Standalone C++ micro-benchmarks, built on **Google Benchmark**, that measure
the compute cost of the performance-critical building blocks of the
`mav_examples` control stack, so they can be characterised on an **NVIDIA
Jetson NX Orin** (or any target). They follow the same framework and
conventions as the per-library `*_benchmark.cpp` tests across this repo (e.g.
`libs/acados_position_mpc/tests/acados_mpc_benchmark.cpp`).

The suite is **ROS-2-free** and self-contained: it links only against the
in-tree `mav_examples` libraries (`acados_position_mpc`,
`acados_ssa_position_mpc`, `acados_trajectory_mpc`, `gcopter_lib`,
`mav_trajectory_generation_cpp`) and the `pid_controllers` +
`geometric_controller` libs vendored under `thirdparty/mav_simulator`, plus
`yaml-cpp` and `benchmark::benchmark`. It does **not** depend on ROS 2, on
the `mav_simulator` dynamics, on the framework adapters, or on anything from
the parent `project_controller_pmpc` workspace.

## What is measured

The suite ships **two binaries** (Google Benchmark, pinned to one harness
thread, 10 repetitions by default for mean/median/stddev/cv):

* **`run_benchmarks`** — covers everything that can co-exist in a single TU:
  PID cascade, P-MPC, SSA-P-MPC, the two trajectory generators
  (generate + evaluate), and the MPC reference-adaptation micro-benchmarks.
* **`run_benchmarks_trajectory`** — Trajectory-MPC solve only. Lives in its
  own binary because `libs/acados_trajectory_mpc` and `libs/acados_position_mpc`
  both expose `acados_mpc::MPC` from the **same** namespace and the underlying
  acados C symbols collide if linked together. Same reason `position_examples`
  does not load the trajectory MPC adapter.

### Controller solve

| Benchmark | Symbol timed | What it represents |
|-----------|--------------|--------------------|
| `BM_PidSolve` | `PositionController` → `VelocityController` → `GeometricController` cascade | Full per-tick PID pipeline (pos → vel → acc → thrust + body rates) |
| `BM_PmpcSolve` | `acados_mpc::MPC::solve()` (position) | Position-MPC OCP solve, warm-started, one per control tick |
| `BM_SsaPmpcSolve` | `acados_ssa_mpc::MPC::solve()` | Steady-state-aware ("MPC for tracking") position-MPC solve |
| `BM_TmpcSolve` | `acados_mpc::MPC::solve()` (trajectory) | Trajectory-MPC solve — lives in `run_benchmarks_trajectory` |

### Trajectory generation (replan) and evaluation (per-tick sampling)

| Benchmark | Symbol timed | What it represents |
|-----------|--------------|--------------------|
| `BM_GcopterGenerate` | `gcopter_lib::TrajectoryGenerator::generate()` | Point-to-point L-BFGS trajectory optimisation (per replan) |
| `BM_GcopterEvaluate` | `gcopter_lib::TrajectoryGenerator::evaluate(t)` | Per-tick sample of the pre-generated polynomial (pos, vel, acc) |
| `BM_MavTrajGenGenerate` | `mav_trajectory_generation_cpp::TrajectoryGenerator::generate()` | Degree-10 polynomial trajectory optimisation (per replan) |
| `BM_MavTrajGenEvaluate` | `mav_trajectory_generation_cpp::TrajectoryGenerator::evaluate(t)` | Per-tick sample of the degree-10 polynomial |

> The `Generate*` cases are typically tens to hundreds of microseconds, the
> `Evaluate*` cases tens to hundreds of nanoseconds — three orders of
> magnitude apart. The generate / evaluate split makes the per-tick polynomial
> readout visible in its own line so the trajectory replan cost (rare, large)
> and the steady-state evaluation cost (per tick, tiny) can be budgeted
> separately.

### MPC reference adaptation

| Benchmark | Binary | What it represents |
|-----------|--------|--------------------|
| `BM_RefsPosOnly` | `run_benchmarks` | progressive carrot, **positions only** (mav_examples `MpcPositionController::setProgressiveReferences`) |
| `BM_RefsPosVel` | `run_benchmarks` | progressive carrot **with the per-stage velocity feed-forward** `v_stage = (s_{k+1} − s_k)/dt_h` (the `as2_position_mpc_plugin` style) |
| `BM_RefsSsaSetpoint` | `run_benchmarks` | the SSA constant set-point (a single `setDesiredPosition` broadcast) |
| `BM_RefsTmpc` | `run_benchmarks_trajectory` | sample a pre-generated gcopter polynomial at `N + 1` stages and write `(pos, vel, orient)` into the trajectory-MPC solver — what a T-MPC controller pays between two `solve()` calls |

> Note: the `mav_examples` Position-MPC OCP exposes a position+orientation
> reference only (`acados_mpc::OnlineParameters` has no `setDesiredVelocity`),
> unlike the aerostack2 plugin's model. In `BM_RefsPosVel` the stage velocity
> is therefore computed and accumulated into a sink rather than written to the
> solver; the benchmark still captures the marginal arithmetic of the velocity
> feed-forward. The Trajectory-MPC OCP does take a per-stage velocity in its
> online parameters, so `BM_TmpcSolve` writes it directly.

### Metrics

- Google Benchmark reports `Time` (real) and `CPU` per iteration, plus the
  iteration count; with `--benchmark_repetitions` it adds `_mean`, `_median`,
  `_stddev` and `_cv` aggregate rows. Solve / generate cases display in
  microseconds, the per-tick evaluate cases and the reference-adaptation cases
  in nanoseconds.
- `BM_PmpcSolve`, `BM_SsaPmpcSolve` and `BM_TmpcSolve` add a custom counter
  **`acados_us`**: the acados-internal `time_tot` averaged per iteration. The
  gap between it and the harness `Time` is the C++ wrapper overhead (state /
  reference marshalling) — usually negligible, confirming the solve dominates.

## Requirements

The benchmark is built **as part of the `mav_examples` build** (enabled by
default via the `BUILD_BENCHMARKS` CMake option). It needs Google Benchmark
(`find_package(benchmark)` — Ubuntu: `libbenchmark-dev`, already present in the
`project_controller_pmpc` Docker). It requires the acados C code
for all three MPC variants (position, trajectory and **SSA position**) to have
been generated — `build.sh` does this automatically when the generated `.so`
files are missing.

At **runtime** the binary links the acados core shared libraries
(`libacados.so`, `libhpipm.so`, `libblasfeo.so`). Because ELF `DT_RUNPATH` is
non-transitive, those must be on `LD_LIBRARY_PATH`. The launcher
`benchmark/run_benchmark.sh` adds the workspace acados lib directory
automatically; if you call the raw binary directly, first source the workspace
environment (`source workspace/.bin/sources.sh`) or export the acados lib path
yourself.

## Build and validate inside the `project_controller_pmpc` Docker

The dev Docker is x86; use it to **validate** that the suite builds and runs.
For real Jetson numbers, build and run on the Orin (see below).

```bash
# On the host:
cd ~/project_controller_pmpc
./docker/scripts/run.sh --dev      # bind-mount the repo over the baked workspace
./docker/scripts/enter.sh          # open a shell in the container

# Inside the container:
cd ~/project_controller_pmpc/workspace/mav_examples
./build.sh                         # acados codegen (if needed) + cmake build (incl. benchmarks)
#   or only the benchmark target (after a previous full build):
cmake --build build --target run_benchmarks -j"$(nproc)"
```

## Run

Always run from the `mav_examples` repository root so the default (relative)
config paths resolve, or use the launcher which does the `cd` for you. All
arguments are standard Google Benchmark flags.

```bash
# Via the launcher (resolves repo root, sets OMP_NUM_THREADS + LD_LIBRARY_PATH):
./benchmark/run_benchmark.sh                                   # run_benchmarks (default)
./benchmark/run_benchmark.sh --benchmark_filter=Solve          # only the controller solves
./benchmark/run_benchmark.sh --benchmark_filter='Gcopter|Traj' # only the trajectory generators

# Trajectory-MPC lives in its own binary (ODR isolation from position MPC).
./benchmark/run_benchmark.sh --target trajectory --benchmark_filter=BM_TmpcSolve

# Run both binaries back-to-back (handy for one-shot characterisation).
./benchmark/run_benchmark.sh --target all --benchmark_repetitions=20

# Stable aggregates + machine-readable output:
./benchmark/run_benchmark.sh --benchmark_repetitions=20 \
    --benchmark_report_aggregates_only=true \
    --benchmark_out=/tmp/orin_bench.json --benchmark_out_format=json

# Raw binaries (must already have the acados libs on LD_LIBRARY_PATH):
./build/benchmark/run_benchmarks            --benchmark_filter=BM_PmpcSolve
./build/benchmark/run_benchmarks_trajectory --benchmark_filter=BM_TmpcSolve
```

> **Threading:** `run_benchmark.sh` exports `OMP_NUM_THREADS=1` by default.
> acados is compiled with OpenMP, but for the small position OCP the thread-pool
> overhead dominates — leaving it unpinned (all cores) was ≈6× slower with ≈10×
> higher variance in the dev container. `BENCHMARK(...)->Threads(1)` only pins
> the *harness*; the acados worker threads are governed by `OMP_NUM_THREADS`.
> Override by exporting it before calling the launcher; on the Orin, sweep it to
> find the best value for the device.

### Useful Google Benchmark flags

```
--benchmark_filter=<regex>            select cases by name (e.g. Pmpc, Refs, Gcopter|Traj)
--benchmark_repetitions=<N>           repeat each case N times (adds mean/median/stddev/cv)
--benchmark_report_aggregates_only=true   print only the aggregate rows
--benchmark_min_time=<seconds>        min wall time per case (double, e.g. 0.5)
--benchmark_out=<path>                write results to a file
--benchmark_out_format=json|csv|console
--benchmark_list_tests=true           list the registered case names
--help                                full flag reference
```

The scenario parameters (config paths, `max_speed`, carrot/hop `distance`) are
compile-time constants in
[include/mav_benchmark/bench_common.hpp](include/mav_benchmark/bench_common.hpp);
edit them there if you need a different operating point.

## Deploying on the NVIDIA Jetson NX Orin

The provided Docker image is x86 (`osrf/ros:humble-desktop`) and is only used
to validate the build. To collect representative Orin numbers:

1. **Build on the device** (native, or an arm64 container). The acados code
   generation already supports arm64 — `docker/teras/t_renderer_arm64` is
   selected automatically by `workspace/.bin/build_workspace.sh` on arm64.
2. **Maximise and pin the clocks** before measuring:
   ```bash
   sudo nvpmodel -m 0     # MAXN power mode (all cores, max frequency)
   sudo jetson_clocks     # lock CPU/GPU/EMC clocks to maximum
   ```
3. **Isolate the process** to reduce jitter:
   ```bash
   OMP_NUM_THREADS=1 taskset -c 4 ./build/benchmark/run_benchmarks \
       --benchmark_repetitions=20 --benchmark_report_aggregates_only=true \
       --benchmark_out=orin.json --benchmark_out_format=json
   ```
   (locking the clocks with `jetson_clocks` also clears Google Benchmark's
   "CPU scaling is enabled" warning). Run with no other significant load.
4. Compare `_median` / `_cv` across runs — the median is more robust than the
   mean for latency budgeting at a fixed control rate, and a low `_cv` (< ~2%)
   means the measurement is stable.

## Interpreting the results

- **`Time` vs `acados_us`**: the gap is the C++ wrapper overhead (state /
  reference marshalling, parameter copies). For real-time budgeting use the
  harness `Time` — that is what the control loop actually pays.
- **Warm-start regime**: the MPC solves are timed exactly as the online
  controller runs them — an ideal plant-free closed loop that feeds the solver's
  own one-step prediction back as the next state, tracking a carrot held a fixed
  distance ahead. This reflects steady-state cruise; a cold start (first solve
  after a large reference jump) is typically more expensive and is not the
  figure reported here. `BM_TmpcSolve` runs a 50-tick warm-up before the harness
  starts timing so SQP_RTI's KKT residual converges to its steady-state regime.
- **PID cascade**: `BM_PidSolve` runs the same three-stage pipeline the
  `PidPositionGeometricController` adapter uses, but it binds the
  `pid_controllers` / `geometric_controller` libraries directly to keep the
  benchmark surface free of `mav_simulator` / framework dependencies. State
  is integrated forward with an open-loop Euler step so the derivative filter,
  anti-windup and saturation paths are exercised in a non-trivial regime.
- **GCOPTER / mav_trajectory_generation** are stateless per call, so every
  `generate()` is a full optimisation. These are the dominant per-replan costs.
  Once a polynomial exists, sampling it with `evaluate(t)` is the per-tick
  cost (`BM_GcopterEvaluate` / `BM_MavTrajGenEvaluate`), three orders of
  magnitude smaller than the optimisation.
- **Reference adaptation** is a few hundred nanoseconds;
  `BM_RefsPosVel − BM_RefsPosOnly` is the cost of the velocity feed-forward,
  negligible next to a `solve()`.
- `BENCHMARK(...)->Threads(1)` pins the harness to one thread; acados' own
  OpenMP threads are controlled separately (see the Threading note above).

## Caveats

- A **Release** build is mandatory for meaningful numbers (`build.sh` uses
  `-DCMAKE_BUILD_TYPE=Release`). Debug builds are not representative.
- Google Benchmark prints a **"CPU scaling is enabled"** warning when the CPU
  governor is not locked; pin the clocks (`jetson_clocks`, or `cpupower
  frequency-set -g performance`) for the most stable numbers.
- This Google Benchmark version (Ubuntu 22.04 / 1.6.x) takes
  `--benchmark_min_time` as a plain number of seconds (e.g. `0.5`), not the
  `0.5s` suffixed form of newer releases.
- Numbers are machine-specific. A run inside the x86 dev container is a
  validation of correctness, **not** an Orin performance figure.
