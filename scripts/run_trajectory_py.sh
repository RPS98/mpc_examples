#!/usr/bin/env bash
# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Run the four trajectory-showcase cases with the Python runner
# (pid and mpc_trajectory × gcopter, jerk_limited). Any extra argument is
# forwarded to the runner.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/_lib/env.sh"
source "${SCRIPT_DIR}/_lib/run_py.sh"

run_one_py run_trajectory_examples "$@"
