# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# One gtest binary per controller / generator adapter. Each test loads the
# real YAML from configs/{controllers,generators,simulation}/, performs a
# short simulation, and asserts the adapter behaves sanely.

enable_testing()

find_package(GTest QUIET)
if(NOT GTest_FOUND)
  message(STATUS "examples_cpp/tests: GTest not found, skipping C++ adapter tests.")
  return()
endif()

set(_repo_root_def "MPC_EXAMPLES_REPO_ROOT=\"${CMAKE_SOURCE_DIR}\"")

function(_add_adapter_test name src lib)
  add_executable(${name} ${src})
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}   # for test_helpers.hpp next to this CMakeLists
  )
  target_link_libraries(${name} PRIVATE
    ${lib}
    mpc_examples_framework
    GTest::gtest
    GTest::gtest_main
  )
  target_compile_definitions(${name} PRIVATE ${_repo_root_def})
  add_test(NAME ${name} COMMAND ${name})
endfunction()

# Controllers
_add_adapter_test(test_pid_geometric
  controllers/test_pid_geometric.cpp
  mpc_examples_controller_pid)
_add_adapter_test(test_mpc_position
  controllers/test_mpc_position.cpp
  mpc_examples_controller_mpc_position)
_add_adapter_test(test_mpc_trajectory
  controllers/test_mpc_trajectory.cpp
  mpc_examples_controller_mpc_trajectory)

# Generators
_add_adapter_test(test_waypoint_reference
  generators/test_waypoint_reference.cpp
  mpc_examples_generators)
_add_adapter_test(test_jerk_limited
  generators/test_jerk_limited.cpp
  mpc_examples_generators)
_add_adapter_test(test_gcopter
  generators/test_gcopter.cpp
  mpc_examples_generators)
_add_adapter_test(test_dynamic
  generators/test_dynamic.cpp
  mpc_examples_generators)
