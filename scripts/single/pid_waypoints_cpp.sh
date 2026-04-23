#!/usr/bin/env bash
# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Run the single case "pid + waypoints" with the C++ position_examples
# binary. Any extra argument is forwarded.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/../_lib/env.sh"
source "${SCRIPT_DIR}/../_lib/run_cpp.sh"

run_one_cpp position_examples \
  --only-controller pid \
  --only-generator  waypoints \
  "$@"
