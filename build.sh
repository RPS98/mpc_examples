#!/bin/bash
# Copyright 2025 Universidad Politécnica de Madrid
# Author: Rafael Perez-Segui <r.psegui@upm.es>
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ── acados code generation ─────────────────────────────────────────────────────
#
# acados-generated C code lives under libs/acados_*_mpc/ and is regenerated
# only on demand (the process requires acados-template and is relatively
# slow). The generated artefacts include both .c/.h sources and the
# companion lib*.so shared libraries consumed at runtime; we check for the
# .so files (not just the directory) so a stale state without compiled
# libraries still triggers regeneration.
ACADOS_POS_SO="libs/acados_position_mpc/mpc_generated_code/mpc_generated_code/libacados_ocp_solver_mpc_position.so"
ACADOS_TRAJ_SO="libs/acados_trajectory_mpc/mpc_generated_code/mpc_generated_code/libacados_ocp_solver_mpc_trajectory.so"
if [ ! -f "${ACADOS_POS_SO}" ] || [ ! -f "${ACADOS_TRAJ_SO}" ]; then
  echo "Generating acados C code (position + trajectory)..."
  bash libs/generate_acados.sh
else
  echo "acados C code already generated — skipping."
fi
# The SSA position MPC wrapper is materialized + its acados C code generated at
# CMake configure time (see libs/CMakeLists.txt); no pre-step is needed here.
echo ""

# ── C++ + Python build ─────────────────────────────────────────────────────────
#
# One CMake invocation builds every C++ target and every pybind11 module of the
# thirdparty submodules, gathering the resulting Python packages under
# build/python/ so a single PYTHONPATH entry exposes all of them. The launch
# scripts under scripts/ prepend build/python/ automatically — no pip install
# step is required.
echo "Building C++ targets and Python bindings..."
mkdir -p build

# pybind11 >= 2.12 is required to link against numpy 2.x ABI. The apt package
# pybind11-dev on Ubuntu 22.04 is 2.9.1, which crashes when numpy 2.x is
# present. Prefer the pybind11 installed as a Python package (typically
# pip-managed >= 2.13), querying its CMake dir at configure time.
PYBIND11_DIR="$(python3 -c 'import pybind11, sys; sys.stdout.write(pybind11.get_cmake_dir())' 2>/dev/null || true)"
if [[ -n "${PYBIND11_DIR}" && -f "${PYBIND11_DIR}/pybind11Config.cmake" ]]; then
  echo "Using pybind11 from: ${PYBIND11_DIR}"
  PYBIND11_CMAKE_ARG="-Dpybind11_DIR=${PYBIND11_DIR}"
else
  echo "Warning: pybind11 Python package not found; falling back to system pybind11 (may be incompatible with numpy 2.x)."
  PYBIND11_CMAKE_ARG=""
fi

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_PYBIND=ON \
  -DBUILD_DEVELOPER_TESTS=ON \
  ${PYBIND11_CMAKE_ARG}
cmake --build build -j"$(nproc)"

echo ""
echo "Build complete."
echo "Python packages are available under: build/python/"
echo "(scripts/run_*.sh automatically prepends this path to PYTHONPATH)"
echo ""
echo "Run every enabled case:     ./scripts/run_all.sh"
echo "Run a single combination:   ./scripts/single/<controller>_<generator>_<lang>.sh"
echo "Run all tests:              ctest --test-dir build --output-on-failure"
