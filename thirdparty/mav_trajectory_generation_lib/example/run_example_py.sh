#!/bin/bash
# Launch the Python example using the package mirrored into the build tree.
# Trajectories are written to CSVs in the logs/ subdirectory, same as the C++
# example. A temporary config YAML is generated on the fly with the output_csv
# fields rewritten to match the YAML filenames (which will be placed in logs/).
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "${SCRIPT_DIR}"

PYTHON_PACKAGE_DIR="${PROJECT_DIR}/build/pybind/python"
EXAMPLE_CONFIG="config_example.yaml"
TRAJECTORY_CONFIG="config_trajectory.yaml"

if [ ! -d "${PYTHON_PACKAGE_DIR}/mav_trajectory_generation_py" ]; then
  echo "Python package not found at ${PYTHON_PACKAGE_DIR}/mav_trajectory_generation_py." >&2
  echo "Build the project first, e.g.:" >&2
  echo "  cmake -S ${PROJECT_DIR} -B ${PROJECT_DIR}/build -DBUILD_EXAMPLES=ON -DBUILD_PYBIND=ON" >&2
  echo "  cmake --build ${PROJECT_DIR}/build -j" >&2
  exit 1
fi

mkdir -p logs

# Write a config variant whose output_csv paths live under logs/ with simple filenames.
PY_CONFIG="logs/config_example.yaml"
python3 - "${EXAMPLE_CONFIG}" "${PY_CONFIG}" <<'PY'
import os, sys, yaml
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    doc = yaml.safe_load(f) or {}
for entry in (doc.get("trajectories") or []):
    name = entry.get("output_csv") or ""
    entry["output_csv"] = os.path.basename(name)
with open(dst, "w") as f:
    yaml.safe_dump(doc, f, sort_keys=False)
PY

PYTHONPATH="${PYTHON_PACKAGE_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  python3 run_example.py "${PY_CONFIG}" "${TRAJECTORY_CONFIG}"

echo
echo "Generating plots..."
python3 plot_results.py \
  -c "${PY_CONFIG}" \
  --plot-all \
  --vmax 3.0 \
  --save logs/plot \
  "${@}"
