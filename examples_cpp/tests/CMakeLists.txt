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

# Variant for tests that ship their own main() (e.g. to call std::_Exit and
# bypass exit-time crashes from racing global destructors in shared libs).
function(_add_adapter_test_custom_main name src lib)
  add_executable(${name} ${src})
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${name} PRIVATE
    ${lib}
    mpc_examples_framework
    GTest::gtest
  )
  target_compile_definitions(${name} PRIVATE ${_repo_root_def})
  add_test(NAME ${name} COMMAND ${name})
endfunction()

# Framework tests link against the framework library plus, when needed, the
# trajectory factory (which transitively pulls in every adapter the trajectory
# binary supports). We reuse the trajectory factory because its scope is the
# superset (PID + MPC-trajectory + every generator) — sufficient for the
# integration smoke test.
function(_add_framework_test name src)
  _add_adapter_test(${name} ${src} mpc_examples_factories_trajectory)
endfunction()

# Controllers
_add_adapter_test(test_pid_geometric
  controllers/test_pid_geometric.cpp
  mpc_examples_controller_pid)
_add_adapter_test(test_pid_trajectory_geometric
  controllers/test_pid_trajectory_geometric.cpp
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
_add_adapter_test(test_mav_traj_gen
  generators/test_mav_traj_gen.cpp
  mpc_examples_generators)

# Framework
_add_framework_test(test_waypoint_scheduler  framework/test_waypoint_scheduler.cpp)
_add_framework_test(test_delay_buffer        framework/test_delay_buffer.cpp)
_add_framework_test(test_parallel_runner     framework/test_parallel_runner.cpp)
# test_waypoints_simulator and the trajectory-factory test ship their own
# main() that calls std::_Exit() — the trajectory factory transitively links
# acados, the dynamic-trajectory worker thread and the simulator stack, whose
# global destructors race at exit and produce a benign double-free after every
# test has already passed.
_add_adapter_test_custom_main(test_waypoints_simulator
  framework/test_waypoints_simulator.cpp
  mpc_examples_factories_trajectory)
_add_adapter_test_custom_main(test_factories_trajectory
  framework/test_factories_trajectory.cpp
  mpc_examples_factories_trajectory)
# test_factories_position is fine with the default gtest_main (no
# dynamic-trajectory worker thread on its link line).
_add_adapter_test(test_factories_position
  framework/test_factories_position.cpp
  mpc_examples_factories_position)
