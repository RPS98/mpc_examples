#!/bin/bash
# Copyright 2025 Universidad Politécnica de Madrid
# Author: Rafael Perez-Segui <r.psegui@upm.es>
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ── acados code generation ─────────────────────────────────────────────────────
#
# acados-generated C code lives under examples/acados_*_mpc/ and is regenerated
# only on demand (the process requires acados-template and is relatively slow).
ACADOS_GENERATED_DIR="examples/acados_position_mpc"
if [ ! -d "${ACADOS_GENERATED_DIR}" ]; then
  echo "Generating acados C code..."
  bash generate.sh
else
  echo "acados C code already generated — skipping."
fi
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
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_PYBIND=ON \
  -DBUILD_DEVELOPER_TESTS=ON
cmake --build build -j"$(nproc)"

echo ""
echo "Build complete."
echo "Python packages are available under: build/python/"
echo "(scripts/run_*.sh automatically prepends this path to PYTHONPATH)"
echo ""
echo "Run all 12 unified examples: ./scripts/run_all.sh"
echo "Compare aggregate metrics  : python3 scripts/compare_all.py"
