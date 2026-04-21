# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

add_library(mpc_examples_adapter_pid_geometric STATIC
  src/pid_geometric_controller.cpp
)

target_include_directories(mpc_examples_adapter_pid_geometric PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mpc_examples_adapter_pid_geometric PUBLIC
  mpc_examples_framework
  mav_simulator   # INTERFACE: exposes pid_controllers/, geometric_controller/ headers
  yaml-cpp
)

target_compile_features(mpc_examples_adapter_pid_geometric PUBLIC cxx_std_17)
