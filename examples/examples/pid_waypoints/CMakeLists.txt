# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

add_executable(${PROJECT_NAME}_run_pid_waypoints run_example.cpp)

target_link_libraries(${PROJECT_NAME}_run_pid_waypoints PRIVATE
  mpc_examples_framework
  mpc_examples_adapter_pid_geometric
  mpc_examples_adapter_waypoint_reference
  mav_simulator
  yaml-cpp
)

set_target_properties(${PROJECT_NAME}_run_pid_waypoints PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples"
)
