#!/usr/bin/env bash
# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Convenience launcher for the C++ performance benchmark suite. It resolves the
# mav_examples repository root (so the default config paths in the binary
# resolve), checks the binary has been built, and forwards all arguments to it.
#
# Two binaries are exposed:
#   * run_benchmarks (default) — PID cascade, P-MPC, SSA-P-MPC solves;
#     gcopter / mav_traj_gen generate + evaluate; MPC reference adaptation.
#   * run_benchmarks_trajectory (--target trajectory) — Trajectory-MPC solve.
#     Lives in its own binary because `acados_position_mpc` and
#     `acados_trajectory_mpc` would collide on the shared `acados_mpc::MPC`
#     symbol if linked together.
#
# Usage:
#   benchmark/run_benchmark.sh [options...]
#   benchmark/run_benchmark.sh --target trajectory --benchmark_filter=BM_TmpcSolve
#   benchmark/run_benchmark.sh --target all   # run both binaries back-to-back
#
# Examples:
#   benchmark/run_benchmark.sh --benchmark_filter=Solve --benchmark_repetitions=20
#   benchmark/run_benchmark.sh --target trajectory --benchmark_out=/tmp/tmpc.json \
#       --benchmark_out_format=json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET="primary"
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="$2"
      shift 2
      ;;
    --target=*)
      TARGET="${1#--target=}"
      shift
      ;;
    *)
      PASSTHROUGH+=("$1")
      shift
      ;;
  esac
done

case "${TARGET}" in
  primary)
    BINS=("${REPO_ROOT}/build/benchmark/run_benchmarks")
    ;;
  trajectory|tmpc)
    BINS=("${REPO_ROOT}/build/benchmark/run_benchmarks_trajectory")
    ;;
  all|both)
    BINS=("${REPO_ROOT}/build/benchmark/run_benchmarks"
          "${REPO_ROOT}/build/benchmark/run_benchmarks_trajectory")
    ;;
  *)
    echo "[run_benchmark] ERROR: unknown --target '${TARGET}' (expected: primary | trajectory | all)" >&2
    exit 1
    ;;
esac

for bin in "${BINS[@]}"; do
  if [[ ! -x "${bin}" ]]; then
    name="$(basename "${bin}")"
    echo "[run_benchmark] ERROR: ${bin} not found." >&2
    echo "[run_benchmark] Build it first: (cd '${REPO_ROOT}' && ./build.sh)" >&2
    echo "[run_benchmark]   or: cmake --build '${REPO_ROOT}/build' --target ${name}" >&2
    exit 1
  fi
done

# Pin acados/OpenMP to a single thread by default. acados is compiled with
# OpenMP, but for the small position OCP the thread-pool overhead dominates and
# oversubscription inflates both the median solve time and its variance (≈6x
# slower, ≈10x higher std observed in the dev container). Single-threaded is the
# representative, predictable regime for per-tick latency budgeting. Override by
# exporting OMP_NUM_THREADS yourself before calling this launcher.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# The binary links acados, whose core shared libraries (libacados / libhpipm /
# libblasfeo) must be resolvable at runtime. ELF DT_RUNPATH is non-transitive,
# so libacados.so cannot find its own siblings via the executable's runpath —
# the acados lib directory must be on LD_LIBRARY_PATH. Prepend it from the
# usual workspace location (or the in-build acados copy) so the launcher works
# without manually sourcing the workspace environment.
for acados_lib in \
  "${REPO_ROOT}/../thirdparty_libs/acados/lib" \
  "${REPO_ROOT}/build/_deps/acados-src/lib"; do
  if [[ -f "${acados_lib}/libhpipm.so" ]]; then
    export LD_LIBRARY_PATH="${acados_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    break
  fi
done

# Run from the repo root so the default relative config paths resolve.
cd "${REPO_ROOT}"
for bin in "${BINS[@]}"; do
  echo "[run_benchmark] $(basename "${bin}")"
  "${bin}" "${PASSTHROUGH[@]}"
done
