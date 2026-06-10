# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Standalone C++ performance benchmark suite, built on Google Benchmark (the
# same framework as the per-library `*_benchmark.cpp` tests across this repo,
# e.g. libs/acados_position_mpc/tests/acados_mpc_benchmark.cpp). ROS-2-free and
# self-contained: it reuses only the in-tree mav_examples solver / generator
# libraries (acados_position_mpc, acados_ssa_position_mpc, gcopter_lib,
# mav_trajectory_generation_cpp) plus yaml-cpp. It does NOT link mav_simulator,
# the framework adapters, or anything from aerostack2 / project_controller_pmpc.

cmake_minimum_required(VERSION 3.16)
project(mav_examples_benchmark LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(benchmark REQUIRED)
if(NOT TARGET yaml-cpp)
  find_package(yaml-cpp REQUIRED)
endif()

# --- Primary binary -----------------------------------------------------------
# Bundles every benchmark whose dependencies do not collide with the trajectory
# MPC. main.cpp provides the runner (instead of BENCHMARK_MAIN) so the P-MPC
# (acados_mpc), SSA-P-MPC (acados_ssa_mpc), PID cascade and the two generator
# pipelines all live in one binary. The trajectory MPC ships in a separate
# binary because its `acados_mpc::MPC` collides with the P-MPC one at link
# time (same namespace, distinct acados C symbols).
add_executable(run_benchmarks
  src/main.cpp
  src/pid_solve_benchmark.cpp
  src/pmpc_solve_benchmark.cpp
  src/ssa_pmpc_solve_benchmark.cpp
  src/gcopter_benchmark.cpp
  src/gcopter_evaluate_benchmark.cpp
  src/mav_traj_gen_benchmark.cpp
  src/mav_traj_gen_evaluate_benchmark.cpp
  src/reference_adaptation_benchmark.cpp
)
target_include_directories(run_benchmarks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(run_benchmarks PRIVATE
  benchmark::benchmark
  acados_position_mpc          # acados_mpc::MPC
  acados_ssa_position_mpc      # acados_ssa_mpc::MPC
  gcopter_lib                  # gcopter_lib::TrajectoryGenerator
  mav_trajectory_generation_cpp  # mav_trajectory_generation_cpp::TrajectoryGenerator
  pid_controllers              # PID position + velocity (from mav_simulator)
  geometric_controller         # accelerationToRates pipeline
  yaml-cpp
)
target_compile_features(run_benchmarks PRIVATE cxx_std_17)

# --- Trajectory MPC binary ----------------------------------------------------
# Isolated from `run_benchmarks` because `libs/acados_trajectory_mpc` and
# `libs/acados_position_mpc` both expose `acados_mpc::MPC` and the underlying
# acados C functions share names; linking them together yields ODR/multiple
# definition errors. We reuse `main.cpp` (no benchmarks registered there) so
# users still get the standard Google Benchmark CLI.
add_executable(run_benchmarks_trajectory
  src/main.cpp
  src/tmpc_solve_benchmark.cpp
  src/tmpc_reference_adaptation_benchmark.cpp
)
target_include_directories(run_benchmarks_trajectory PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(run_benchmarks_trajectory PRIVATE
  benchmark::benchmark
  acados_trajectory_mpc        # acados_mpc::MPC (trajectory variant)
  gcopter_lib                  # source of the sampled trajectory in BM_RefsTmpc
  yaml-cpp
)
target_compile_features(run_benchmarks_trajectory PRIVATE cxx_std_17)
