#!/usr/bin/env bash
# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Convenience launcher for the C++ performance benchmark suite. It resolves the
# mav_examples repository root (so the default config paths in the binary
# resolve), checks the binary has been built, and forwards all arguments to it.
#
# Two binaries are exposed:
#   * run_benchmarks — PID cascade, P-MPC, SSA-P-MPC solves; gcopter /
#     mav_traj_gen generate + evaluate; MPC reference adaptation.
#   * run_benchmarks_trajectory — Trajectory-MPC solve + its reference
#     adaptation. Lives in its own binary because `acados_position_mpc`
#     and `acados_trajectory_mpc` would collide on the shared
#     `acados_mpc::MPC` symbol if linked together.
#
# By default the launcher runs BOTH binaries back-to-back (--target all).
# Pass --target primary or --target trajectory to restrict to one.
#
# Usage:
#   benchmark/run_benchmark.sh [options...]              # both binaries
#   benchmark/run_benchmark.sh --target primary ...      # only run_benchmarks
#   benchmark/run_benchmark.sh --target trajectory ...   # only run_benchmarks_trajectory
#
# Examples:
#   benchmark/run_benchmark.sh --benchmark_filter=Solve --benchmark_repetitions=20
#   benchmark/run_benchmark.sh --target trajectory --benchmark_out=/tmp/tmpc.json \
#       --benchmark_out_format=json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET="all"
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

# When the user does NOT pass --benchmark_out=... themselves, we capture each
# binary's run into a temporary JSON so we can print a compact summary table
# (mean ± stddev, equivalent frequency) at the end. The full Google Benchmark
# console output is still streamed to stdout in real time.
USER_PROVIDED_OUT=0
for arg in "${PASSTHROUGH[@]}"; do
  case "${arg}" in
    --benchmark_out=*|--benchmark_out_format=*) USER_PROVIDED_OUT=1 ;;
  esac
done

TMP_JSON_DIR=""
if [[ "${USER_PROVIDED_OUT}" -eq 0 ]]; then
  TMP_JSON_DIR="$(mktemp -d -t mav_bench_XXXX)"
  trap 'rm -rf "${TMP_JSON_DIR}"' EXIT
fi

JSON_FILES=()
for bin in "${BINS[@]}"; do
  echo "[run_benchmark] $(basename "${bin}")"
  if [[ "${USER_PROVIDED_OUT}" -eq 0 ]]; then
    json_path="${TMP_JSON_DIR}/$(basename "${bin}").json"
    "${bin}" "${PASSTHROUGH[@]}" \
      --benchmark_out="${json_path}" --benchmark_out_format=json
    JSON_FILES+=("${json_path}")
  else
    "${bin}" "${PASSTHROUGH[@]}"
  fi
done

if [[ "${USER_PROVIDED_OUT}" -eq 0 && ${#JSON_FILES[@]} -gt 0 ]]; then
  echo
  echo "==================== run_benchmark summary ===================="
  python3 - "${JSON_FILES[@]}" <<'PY'
import json
import sys
import collections

# Google Benchmark's repetitions suffixes the per-iteration name with
# /repeats:N/threads:1_mean (or _median/_stddev/_cv). We group by the base
# name and pull mean/stddev/unit from the respective aggregate rows.
units_to_us = {'ns': 1e-3, 'us': 1.0, 'ms': 1e3, 's': 1e6}
groups = collections.OrderedDict()

for path in sys.argv[1:]:
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        continue
    for bm in data.get('benchmarks', []):
        name = bm.get('name', '')
        run_type = bm.get('run_type', '')
        agg = bm.get('aggregate_name', '')
        if run_type != 'aggregate':
            continue
        # Strip the "/repeats:N/threads:K_<agg>" suffix to get the base name.
        base = name
        for sep in ('/repeats:', '_mean', '_median', '_stddev', '_cv'):
            if sep in base:
                base = base.split(sep)[0]
        groups.setdefault(base, {})[agg] = bm

if not groups:
    print('(no aggregate data parsed)')
    sys.exit(0)

# Column widths
name_w = max(len(n) for n in groups) + 2
hdr = ('| {n:<' + str(name_w) + 's} | {tm:>10s} | {ts:>10s} | {tcv:>6s} | '
       '{fm:>12s} | {fs:>12s} |')
sep = ('|' + '-' * (name_w + 2) + '|' + '-' * 12 + '|' + '-' * 12 + '|' +
       '-' * 8 + '|' + '-' * 14 + '|' + '-' * 14 + '|')
print(hdr.format(n='Benchmark', tm='Mean (us)', ts='Stddev', tcv='CV (%)',
                 fm='Mean freq Hz', fs='Std freq Hz'))
print(sep)
for base, aggs in groups.items():
    mean_row = aggs.get('mean')
    std_row = aggs.get('stddev')
    cv_row = aggs.get('cv')
    if mean_row is None:
        continue
    unit = mean_row.get('time_unit', 'us')
    scale = units_to_us.get(unit, 1.0)
    t_mean_us = float(mean_row.get('real_time', 0.0)) * scale
    t_std_us = (float(std_row.get('real_time', 0.0)) * scale
                if std_row is not None else 0.0)
    cv_pct = (100.0 * float(cv_row.get('real_time', 0.0))
              if cv_row is not None else float('nan'))
    # Frequency (Hz) and its propagated stddev: f = 1e6/t  →  df ≈ 1e6/t² · dt.
    f_mean = 1.0e6 / t_mean_us if t_mean_us > 0 else float('nan')
    f_std = (1.0e6 * t_std_us / (t_mean_us ** 2)
             if t_mean_us > 0 else float('nan'))
    if t_mean_us >= 1.0:
        t_mean_s = f'{t_mean_us:10.2f}'
        t_std_s = f'{t_std_us:10.2f}'
    else:
        # Sub-microsecond entries (refs, evaluate): show in ns.
        t_mean_s = f'{t_mean_us * 1e3:7.1f} ns'
        t_std_s = f'{t_std_us * 1e3:7.1f} ns'
    print(hdr.format(n=base, tm=t_mean_s, ts=t_std_s,
                     tcv=f'{cv_pct:6.2f}',
                     fm=f'{f_mean:12.1f}', fs=f'{f_std:12.1f}'))
PY
fi
