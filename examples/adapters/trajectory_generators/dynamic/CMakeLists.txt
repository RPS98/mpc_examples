# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

add_library(mpc_examples_adapter_dynamic STATIC
  src/dynamic_trajectory_generator.cpp
)

target_include_directories(mpc_examples_adapter_dynamic PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mpc_examples_adapter_dynamic PUBLIC
  mpc_examples_framework
  dynamic_trajectory_generator
  mav_simulator
  yaml-cpp
)

target_compile_features(mpc_examples_adapter_dynamic PUBLIC cxx_std_17)
