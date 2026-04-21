# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

# Adapter that bridges the acados trajectory-MPC solver behind the IController
# interface. Links only against acados_trajectory_mpc (NOT acados_position_mpc)
# to avoid ODR violations — both libraries expose the symbol acados_mpc::MPC.

add_library(mpc_examples_adapter_mpc_trajectory STATIC
  src/mpc_trajectory_controller.cpp
)

target_include_directories(mpc_examples_adapter_mpc_trajectory PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mpc_examples_adapter_mpc_trajectory PUBLIC
  mpc_examples_framework
  acados_trajectory_mpc
  mav_simulator
  yaml-cpp
)

target_compile_features(mpc_examples_adapter_mpc_trajectory PUBLIC cxx_std_17)
