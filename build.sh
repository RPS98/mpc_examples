#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ── Python dependencies ────────────────────────────────────────────────────────

# Install mav_simulator Python bindings (mavpy) if not already available
if ! python3 -c "import mavpy" &>/dev/null; then
  echo "Installing mav_simulator Python bindings..."
  pip3 install thirdparty/mav_simulator/pybind/
else
  echo "mavpy already installed — skipping."
fi
echo ""  

# Install position_mpc Python package if not already available
if ! python3 -c "import mpc_position" &>/dev/null; then
  echo "Installing position_mpc Python package..."
  pip3 install thirdparty/position_mpc/
else
  echo "mpc_position already installed — skipping."
fi
echo ""  

# ── acados code generation ─────────────────────────────────────────────────────

# If generated code is missing, regenerate it.
ACADOS_GENERATED_DIR="examples/acados_position_mpc"
if [ ! -d "${ACADOS_GENERATED_DIR}" ]; then
  echo "Generating acados C code..."
  bash generate.sh
else
  echo "acados C code already generated — skipping."
fi
echo ""  

# ── C++ build ─────────────────────────────────────────────────────────────────

echo ""
echo "Building C++ example..."
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DBUILD_EXAMPLES=ON -DBUILD_DEVELOPER_TESTS=ON
cmake --build build -j"$(nproc)"

echo ""
echo "Build complete."
echo "Run the C++ example : ./run_example_cpp.sh"
echo "Run the Python example: ./run_example_py.sh"
