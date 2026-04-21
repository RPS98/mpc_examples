# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

add_library(mpc_examples_adapter_gcopter STATIC
  src/gcopter_generator.cpp
)

target_include_directories(mpc_examples_adapter_gcopter PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mpc_examples_adapter_gcopter PUBLIC
  mpc_examples_framework
  gcopter_lib
  mav_simulator
  yaml-cpp
)

target_compile_features(mpc_examples_adapter_gcopter PUBLIC cxx_std_17)
