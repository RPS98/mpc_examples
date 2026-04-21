# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

# Unified per-combination example mains (<controller>_<generator>).
# Each subdirectory produces one executable wired from the shared framework
# and two adapter targets. Add new combinations here as additional
# add_subdirectory() calls — no other file needs touching.

add_subdirectory(pid_waypoints)
add_subdirectory(pid_jerk_limited)
add_subdirectory(pid_gcopter)
add_subdirectory(pid_dynamic)
add_subdirectory(mpc_position_waypoints)
add_subdirectory(mpc_position_jerk_limited)
add_subdirectory(mpc_position_gcopter)
add_subdirectory(mpc_position_dynamic)
add_subdirectory(mpc_trajectory_waypoints)
add_subdirectory(mpc_trajectory_jerk_limited)
add_subdirectory(mpc_trajectory_gcopter)
add_subdirectory(mpc_trajectory_dynamic)
