#!/usr/bin/env bash
# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Run the two position-showcase cases with the C++ binary
# (pid + waypoints, mpc_position + waypoints). Any extra argument is
# forwarded to the executable.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/_lib/env.sh"
source "${SCRIPT_DIR}/_lib/run_cpp.sh"

run_one_cpp position_examples "$@"
