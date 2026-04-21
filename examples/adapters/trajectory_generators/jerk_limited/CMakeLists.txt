# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

add_library(mpc_examples_adapter_jerk_limited STATIC
  src/jerk_limited_generator.cpp
)

target_include_directories(mpc_examples_adapter_jerk_limited PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mpc_examples_adapter_jerk_limited PUBLIC
  mpc_examples_framework
  trajectory_generator_jerk_limited
  mav_simulator
  yaml-cpp
)

target_compile_features(mpc_examples_adapter_jerk_limited PUBLIC cxx_std_17)
