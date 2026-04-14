#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

rm -f mpc_log.csv
./build/examples/mpc_examples_run_example \
  -c config_example.yaml \
  -s config_simulator.yaml \
  -m config_mpc.yaml \
  -f mpc_log.csv
python3 examples/utils/plot_results.py -f mpc_log.csv
