#!/usr/bin/env bash
# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Convenience launcher for the C++ performance benchmark suite. It resolves the
# mav_examples repository root (so the default config paths in the binary
# resolve), checks the binary has been built, and forwards all arguments to it.
#
# Usage:
#   benchmark/run_benchmark.sh [run_benchmarks options...]
#
# Examples:
#   benchmark/run_benchmark.sh --bench all --csv /tmp/orin_bench.csv
#   benchmark/run_benchmark.sh --bench gcopter --gcopter-iterations 500 --distance 8.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN="${REPO_ROOT}/build/benchmark/run_benchmarks"

if [[ ! -x "${BIN}" ]]; then
  echo "[run_benchmark] ERROR: ${BIN} not found." >&2
  echo "[run_benchmark] Build it first: (cd '${REPO_ROOT}' && ./build.sh)" >&2
  echo "[run_benchmark]   or: cmake --build '${REPO_ROOT}/build' --target run_benchmarks" >&2
  exit 1
fi

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
exec "${BIN}" "$@"
