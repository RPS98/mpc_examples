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

# Single executable bundling every benchmark TU. main.cpp provides the
# benchmark runner (instead of BENCHMARK_MAIN()) so the P-MPC (acados_mpc) and
# SSA-P-MPC (acados_ssa_mpc) benchmarks — distinct namespaces / acados models —
# share one binary, never the trajectory MPC (which would collide on
# acados_mpc::MPC).
add_executable(run_benchmarks
  src/main.cpp
  src/pmpc_solve_benchmark.cpp
  src/ssa_pmpc_solve_benchmark.cpp
  src/gcopter_benchmark.cpp
  src/mav_traj_gen_benchmark.cpp
  src/reference_adaptation_benchmark.cpp
)
target_include_directories(run_benchmarks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(run_benchmarks PRIVATE
  benchmark::benchmark
  acados_position_mpc          # acados_mpc::MPC
  acados_ssa_position_mpc      # acados_ssa_mpc::MPC
  gcopter_lib                  # gcopter_lib::TrajectoryGenerator
  mav_trajectory_generation_cpp  # mav_trajectory_generation_cpp::TrajectoryGenerator
  yaml-cpp
)
target_compile_features(run_benchmarks PRIVATE cxx_std_17)
