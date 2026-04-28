# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Consolidated build for the C++ half of the mpc_examples showcase. Organises
# the interfaces (framework/), concrete adapters (controllers/, generators/)
# and the two entry-point executables under a single subdirectory.

cmake_minimum_required(VERSION 3.16)
project(mpc_examples_cpp LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)  # std::thread in run_*_examples.cpp parallel mode

# Helper variable so every target shares the same include path root. The
# ``include/`` subtree is organised by layer (framework/, controllers/,
# generators/, utils/) matching the ``#include "controllers/..."`` and
# ``#include "generators/..."`` conventions used across the codebase.
set(_inc ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# framework: IController, ITrajectoryGenerator, WaypointsSimulator,
#            WaypointScheduler, UnifiedMcapLogger (facade over mav_flight_review).
# ---------------------------------------------------------------------------
add_library(mpc_examples_framework STATIC
  src/framework/waypoints_simulator.cpp
  src/framework/waypoint_scheduler.cpp
  src/framework/unified_mcap_logger.cpp
)

target_include_directories(mpc_examples_framework PUBLIC
  ${_inc}
)

target_link_libraries(mpc_examples_framework PUBLIC
  mav_simulator        # INTERFACE: provides mav_model / simulator headers + Eigen
  yaml-cpp             # consumed via utils/example_config_utils.hpp
  mav_flight_review    # MCAP / CSV logger backing unified_mcap_logger.hpp
)

target_compile_features(mpc_examples_framework PUBLIC cxx_std_17)

# ---------------------------------------------------------------------------
# Controllers (one STATIC per adapter; MPC variants stay separate so their
# corresponding acados library only ends up in its own executable).
# ---------------------------------------------------------------------------
add_library(mpc_examples_controller_pid STATIC
  src/controllers/pid_position_geometric_controller.cpp
  src/controllers/pid_trajectory_geometric_controller.cpp
)
target_include_directories(mpc_examples_controller_pid PUBLIC ${_inc})
target_link_libraries(mpc_examples_controller_pid PUBLIC
  mpc_examples_framework
  mav_simulator
  yaml-cpp
)
target_compile_features(mpc_examples_controller_pid PUBLIC cxx_std_17)

add_library(mpc_examples_controller_mpc_position STATIC
  src/controllers/mpc_position_controller.cpp
)
target_include_directories(mpc_examples_controller_mpc_position PUBLIC ${_inc})
target_link_libraries(mpc_examples_controller_mpc_position PUBLIC
  mpc_examples_framework
  acados_position_mpc
  mav_simulator
  yaml-cpp
)
target_compile_features(mpc_examples_controller_mpc_position PUBLIC cxx_std_17)

add_library(mpc_examples_controller_mpc_trajectory STATIC
  src/controllers/mpc_trajectory_controller.cpp
)
target_include_directories(mpc_examples_controller_mpc_trajectory PUBLIC ${_inc})
target_link_libraries(mpc_examples_controller_mpc_trajectory PUBLIC
  mpc_examples_framework
  acados_trajectory_mpc
  mav_simulator
  yaml-cpp
)
target_compile_features(mpc_examples_controller_mpc_trajectory PUBLIC cxx_std_17)

# ---------------------------------------------------------------------------
# Generators (all five bundled in a single STATIC; they have no ODR issues).
# ---------------------------------------------------------------------------
add_library(mpc_examples_generators STATIC
  src/generators/waypoint_reference_generator.cpp
  src/generators/jerk_limited_generator.cpp
  src/generators/gcopter_generator.cpp
  src/generators/dynamic_trajectory_generator.cpp
  src/generators/mav_traj_gen_generator.cpp
)
target_include_directories(mpc_examples_generators PUBLIC ${_inc})
target_link_libraries(mpc_examples_generators PUBLIC
  mpc_examples_framework
  trajectory_generator_jerk_limited
  gcopter_lib
  dynamic_trajectory_generator
  mav_trajectory_generation_cpp
  mav_simulator
  yaml-cpp
)
target_compile_features(mpc_examples_generators PUBLIC cxx_std_17)

# ---------------------------------------------------------------------------
# Factories split per binary. Each TU only ``#include``s the controllers and
# generators its binary supports, preventing acados ODR collisions.
# ---------------------------------------------------------------------------
add_library(mpc_examples_factories_position STATIC
  src/framework/factories_position.cpp
)
target_include_directories(mpc_examples_factories_position PUBLIC ${_inc})
target_link_libraries(mpc_examples_factories_position PUBLIC
  mpc_examples_framework
  mpc_examples_controller_pid
  mpc_examples_controller_mpc_position
  mpc_examples_generators
)
target_compile_features(mpc_examples_factories_position PUBLIC cxx_std_17)

add_library(mpc_examples_factories_trajectory STATIC
  src/framework/factories_trajectory.cpp
)
target_include_directories(mpc_examples_factories_trajectory PUBLIC ${_inc})
target_link_libraries(mpc_examples_factories_trajectory PUBLIC
  mpc_examples_framework
  mpc_examples_controller_pid
  mpc_examples_controller_mpc_trajectory
  mpc_examples_generators
)
target_compile_features(mpc_examples_factories_trajectory PUBLIC cxx_std_17)

# ---------------------------------------------------------------------------
# Executables (one per family). Each links against only the factory matching
# its scope — never both.
# ---------------------------------------------------------------------------
add_executable(position_examples src/run_position_examples.cpp)
target_link_libraries(position_examples PRIVATE
  mpc_examples_factories_position
  mav_simulator
  yaml-cpp
  Threads::Threads
)

add_executable(trajectory_examples src/run_trajectory_examples.cpp)
target_link_libraries(trajectory_examples PRIVATE
  mpc_examples_factories_trajectory
  mav_simulator
  yaml-cpp
  Threads::Threads
)

# ---------------------------------------------------------------------------
# Tests (only when the test sources exist to avoid a stale CMakeLists from
# tripping up incremental rebuilds during the migration).
# ---------------------------------------------------------------------------
if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/controllers/test_pid_geometric.cpp")
  add_subdirectory(tests)
endif()
