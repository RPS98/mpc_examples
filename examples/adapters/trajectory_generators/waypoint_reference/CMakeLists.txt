# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

add_library(mpc_examples_adapter_waypoint_reference STATIC
  src/waypoint_reference_generator.cpp
)

target_include_directories(mpc_examples_adapter_waypoint_reference PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mpc_examples_adapter_waypoint_reference PUBLIC
  mpc_examples_framework
  yaml-cpp
)

target_compile_features(mpc_examples_adapter_waypoint_reference PUBLIC cxx_std_17)
