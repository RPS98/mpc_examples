#!/bin/bash
# Launch the C++ example: plan all trajectories declared in the example YAML
# and plot each that succeeded. Paths are resolved relative to this script's
# directory (example/), so the command can be invoked from anywhere.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "${SCRIPT_DIR}"

BINARY="${PROJECT_DIR}/build/example/run_example"
EXAMPLE_CONFIG="config_example.yaml"
TRAJECTORY_CONFIG="config_trajectory.yaml"

if [ ! -x "${BINARY}" ]; then
  echo "Binary not found at ${BINARY}." >&2
  echo "Build the project first, e.g.:" >&2
  echo "  cmake -S ${PROJECT_DIR} -B ${PROJECT_DIR}/build -DBUILD_EXAMPLES=ON" >&2
  echo "  cmake --build ${PROJECT_DIR}/build -j" >&2
  exit 1
fi

# Create logs directory and remove any stale CSVs so we can detect which
# trajectories actually succeeded this run.
mkdir -p logs
python3 - "${EXAMPLE_CONFIG}" <<'PY'
import sys, os, yaml
with open(sys.argv[1]) as f:
    doc = yaml.safe_load(f) or {}
for entry in (doc.get("trajectories") or []):
    out = entry.get("output_csv")
    if out:
        csv_file = os.path.join("logs", os.path.basename(out))
        if os.path.isfile(csv_file):
            os.remove(csv_file)
PY

"${BINARY}" "${EXAMPLE_CONFIG}" "${TRAJECTORY_CONFIG}"

echo
echo "Generating plots..."
python3 plot_results.py \
  -c "${EXAMPLE_CONFIG}" \
  --plot-all \
  --vmax 3.0 \
  --save logs/plot \
  "${@}"
