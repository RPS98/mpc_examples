#!/usr/bin/env bash
# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Run the single case "pid + waypoints" with the Python position runner.
# Any extra argument is forwarded.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/../_lib/env.sh"
source "${SCRIPT_DIR}/../_lib/run_py.sh"

run_one_py run_position_examples \
  --only-controller pid \
  --only-generator  waypoints \
  "$@"
