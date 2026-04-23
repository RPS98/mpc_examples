#!/usr/bin/env bash
# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Run the single case "pid + jerk_limited" with the Python trajectory runner.
# Any extra argument is forwarded.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/../_lib/env.sh"
source "${SCRIPT_DIR}/../_lib/run_py.sh"

run_one_py run_trajectory_examples \
  --only-controller pid \
  --only-generator  jerk_limited \
  "$@"
