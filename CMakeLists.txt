# Copyright 2025 Universidad Politécnica de Madrid
# Author: Rafael Perez-Segui <r.psegui@upm.es>
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.16)
set(PROJECT_NAME mpc_examples)
project(${PROJECT_NAME} LANGUAGES CXX)

# --- C++ standard ---
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# --- Default build type ---
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
# This umbrella project ships both C++ binaries and Python mirrors. Pybind is
# ON by default (the showcase needs it for the Python twins), but the user can
# disable it for a slimmer build.
option(BUILD_PYBIND   "Build Python bindings in thirdparty submodules" ON)
option(BUILD_TESTING  "Enable CTest integration"                       ON)

# ---------------------------------------------------------------------------
# Dependencies (resolved up-front so every subdirectory sees them)
# ---------------------------------------------------------------------------
find_package(yaml-cpp REQUIRED)

# fastcdr is consumed transitively by mav_flight_review's MCAP writer. Resolve
# it here so the IMPORTED target is visible to every consumer (find_package
# inside helper functions only exposes the target locally). Prefer ROS 2
# Humble's package; fall back to system CMake paths.
find_package(fastcdr CONFIG QUIET PATHS /opt/ros/humble /opt/ros/iron)
if(NOT fastcdr_FOUND)
  find_package(fastcdr CONFIG QUIET)
endif()

# Enable CTest early so add_test() calls inside add_subdirectory()'d targets
# (e.g. examples_cpp/tests/) register correctly.
include(CTest)

# ---------------------------------------------------------------------------
# Centralized Python mirror root
# ---------------------------------------------------------------------------
# Every thirdparty submodule that ships a Python package mirrors its compiled
# .so and/or pure-Python tree under PYBIND_PY_MIRROR_ROOT, so a single
# PYTHONPATH entry exposes the full namespace without needing pip install.
# Defining it here as CACHE INTERNAL pre-empts the per-submodule fallback so
# every sub-pybind agrees on a single root.
set(PYBIND_PY_MIRROR_ROOT "${CMAKE_BINARY_DIR}/python" CACHE INTERNAL
    "Shared Python mirror root for all pybind sub-packages and pure-Python mirrors")
file(MAKE_DIRECTORY "${PYBIND_PY_MIRROR_ROOT}")
message(STATUS "${PROJECT_NAME}: Python packages will be gathered under ${PYBIND_PY_MIRROR_ROOT}")

# ---------------------------------------------------------------------------
# Thirdparty submodules
# ---------------------------------------------------------------------------
# mav_flight_review: MCAP + CSV telemetry backend (replaces the legacy
# mav_flight_logger). The submodule's own examples and tests are disabled —
# this repo ships its own under examples_cpp/.
set(MAV_FLIGHT_REVIEW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MAV_FLIGHT_REVIEW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(MAV_FLIGHT_REVIEW_BUILD_PYBIND   ${BUILD_PYBIND} CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/mav_flight_review)

# Expose mav_flight_review's Python package under PYBIND_PY_MIRROR_ROOT so
# `import mav_flight_review` works with a single PYTHONPATH entry. Matches
# the symlink pattern used by mpc / mav_simulator.
if(BUILD_PYBIND)
  file(MAKE_DIRECTORY "${PYBIND_PY_MIRROR_ROOT}")
  set(_mfr_src "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/mav_flight_review/pybind/python/mav_flight_review")
  set(_mfr_dst "${PYBIND_PY_MIRROR_ROOT}/mav_flight_review")
  if(EXISTS "${_mfr_dst}" OR IS_SYMLINK "${_mfr_dst}")
    file(REMOVE_RECURSE "${_mfr_dst}")
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink "${_mfr_src}" "${_mfr_dst}"
    RESULT_VARIABLE _symlink_rc
  )
  if(NOT _symlink_rc EQUAL 0)
    message(WARNING "mav_flight_review: failed to symlink ${_mfr_src} -> ${_mfr_dst}")
  else()
    message(STATUS "mav_flight_review: mirrored -> ${_mfr_dst}")
  endif()
endif()

# Quadrotor simulator (mav_model + mav_controllers + mav_sensors composition).
add_subdirectory(thirdparty/mav_simulator)

# Trajectory generators. Save/restore BUILD_TESTING for submodules that pull
# in test-only dependencies through their own CMakeLists.
set(_mpc_examples_build_testing_saved ${BUILD_TESTING})
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/dynamic_trajectory_generator)
set(BUILD_TESTING ${_mpc_examples_build_testing_saved} CACHE BOOL "" FORCE)

add_subdirectory(thirdparty/trajectory_generator_jerk_limited)
add_subdirectory(thirdparty/gcopter_lib)
add_subdirectory(thirdparty/mav_trajectory_generation_lib)

# MPC controllers (acados-generated code, position + trajectory variants).
add_subdirectory(thirdparty/mpc)

# ---------------------------------------------------------------------------
# Optional MPCC dependency (mpcc_acados + spline_trajectory_generator).
# ---------------------------------------------------------------------------
# `mav_examples` is standalone (upstream RPS98/mpc_examples), but the MPCC
# adapter and the closed-loop circuit_examples executable need the C++ side
# of the mpcc_acados solver and the spline library, neither of which is
# vendored here. When the consumer project (e.g. mpcc_v2) makes them
# available via MPCC_SOURCE_DIR (or the MPCC_REPO_ROOT env var), the
# circuit_examples target is built; otherwise it is silently skipped.
set(MPCC_SOURCE_DIR "" CACHE PATH
    "Path to the mpcc/ source root (e.g. <mpcc_v2>/workspace/thirdparty_libs/mpcc).")
if(NOT MPCC_SOURCE_DIR AND DEFINED ENV{MPCC_REPO_ROOT})
  set(MPCC_SOURCE_DIR "$ENV{MPCC_REPO_ROOT}/workspace/thirdparty_libs/mpcc"
      CACHE PATH "" FORCE)
endif()
set(MPC_EXAMPLES_HAVE_MPCC OFF)
if(MPCC_SOURCE_DIR AND
   EXISTS "${MPCC_SOURCE_DIR}/mpcc_acados_example/mpcc_interface/CMakeLists.txt" AND
   EXISTS "${MPCC_SOURCE_DIR}/mpcc_acados_example/mpcc_interface/mpc_generated_code/mpc_generated_code/libacados_ocp_solver_mpcc.so")
  if(NOT TARGET mpcc_acados)
    add_subdirectory("${MPCC_SOURCE_DIR}/mpcc_acados_example/mpcc_interface"
                     "${CMAKE_BINARY_DIR}/_mpcc_interface" EXCLUDE_FROM_ALL)
  endif()
  if(NOT TARGET spline_trajectory_generator)
    add_subdirectory("${MPCC_SOURCE_DIR}/mpcc_acados_example/examples/spline_trajectory_generator"
                     "${CMAKE_BINARY_DIR}/_spline_trajectory_generator" EXCLUDE_FROM_ALL)
    # Upstream upstream's CMakeLists lists both arc_length_reparametrization.cpp
    # and trajectory_generator.cpp as SOURCES. The corresponding header
    # arc_length_reparametrization.hpp defines its free functions without the
    # `inline` keyword, so each TU that #includes it emits its own copy of
    # the symbols — two TUs → multiple-definition at link. trajectory_generator
    # already pulls in the header, so dropping the now-redundant
    # arc_length_reparametrization.cpp leaves a single set of definitions.
    # Self-contained patch (no upstream edit needed).
    get_target_property(_spline_srcs spline_trajectory_generator SOURCES)
    list(FILTER _spline_srcs EXCLUDE REGEX "arc_length_reparametrization\\.cpp$")
    set_target_properties(spline_trajectory_generator PROPERTIES SOURCES "${_spline_srcs}")
  endif()
  set(MPC_EXAMPLES_HAVE_MPCC ON)
  message(STATUS "mav_examples: MPCC available (mpcc_acados + spline_trajectory_generator).")
else()
  message(STATUS "mav_examples: MPCC unavailable; circuit_examples target skipped. "
                 "Set -DMPCC_SOURCE_DIR=<path-to-thirdparty_libs/mpcc> or the "
                 "MPCC_REPO_ROOT env var to enable.")
endif()

# ---------------------------------------------------------------------------
# Project targets
# ---------------------------------------------------------------------------
# Generated acados MPC targets (autogenerated code isolated under libs/).
add_subdirectory(libs)

# C++ half of the showcase: interfaces + adapters + entry-point executables.
add_subdirectory(examples_cpp)

# Python half of the showcase: pure-Python mirrors + runs + tests. The
# subdirectory only manipulates the PYBIND_PY_MIRROR_ROOT symlink and
# registers the pytest suite; the actual bindings come from the thirdparty
# submodules above.
if(EXISTS "${CMAKE_SOURCE_DIR}/examples_py/CMakeLists.txt")
  add_subdirectory(examples_py)
endif()

# ---------------------------------------------------------------------------
# Pytest suites (one per thirdparty submodule that ships Python tests)
# ---------------------------------------------------------------------------
# Specific to this umbrella: register every thirdparty pytest tree so a single
# `ctest` invocation covers both C++ (gtest) and Python (pytest). Each suite
# imports its bindings from ${PYBIND_PY_MIRROR_ROOT}, so no `pip install` is
# needed — only a successful `cmake --build` beforehand.
if(BUILD_TESTING)
  find_program(PYTEST_EXECUTABLE NAMES pytest pytest-3 py.test-3)
  if(NOT PYTEST_EXECUTABLE)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
      set(PYTEST_EXECUTABLE ${Python3_EXECUTABLE})
      set(_pytest_extra_args -m pytest)
    endif()
  else()
    set(_pytest_extra_args "")
  endif()

  if(PYTEST_EXECUTABLE)
    # register_pytest_suite(<ctest_name> <tests_dir>)
    # Runs pytest on <tests_dir> with PYTHONPATH pointing to PYBIND_PY_MIRROR_ROOT
    # so the compiled bindings + pure-Python mirrors resolve transparently.
    function(register_pytest_suite suite_name tests_dir)
      if(NOT IS_DIRECTORY "${tests_dir}")
        message(WARNING "register_pytest_suite: ${tests_dir} does not exist, skipping.")
        return()
      endif()
      add_test(
        NAME ${suite_name}
        COMMAND ${PYTEST_EXECUTABLE} ${_pytest_extra_args}
                -ra -q "${tests_dir}"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      )
      set_tests_properties(${suite_name} PROPERTIES
        ENVIRONMENT "PYTHONPATH=${PYBIND_PY_MIRROR_ROOT}:$ENV{PYTHONPATH}"
        LABELS "pytest"
      )
    endfunction()

    # mav_simulator (and its libs)
    register_pytest_suite(pytest_mav_simulator
      "${CMAKE_SOURCE_DIR}/thirdparty/mav_simulator/pybind/tests")
    register_pytest_suite(pytest_mav_model
      "${CMAKE_SOURCE_DIR}/thirdparty/mav_simulator/libs/mav_model/pybind/tests")
    register_pytest_suite(pytest_mav_model_definition
      "${CMAKE_SOURCE_DIR}/thirdparty/mav_simulator/libs/mav_model/model_definition/pytests")
    register_pytest_suite(pytest_mav_controllers
      "${CMAKE_SOURCE_DIR}/thirdparty/mav_simulator/libs/mav_controllers/pybind/tests")
    register_pytest_suite(pytest_mav_pid_controllers
      "${CMAKE_SOURCE_DIR}/thirdparty/mav_simulator/libs/mav_controllers/controllers/pid_controllers/pybind/tests")
    register_pytest_suite(pytest_mav_geometric_controller
      "${CMAKE_SOURCE_DIR}/thirdparty/mav_simulator/libs/mav_controllers/controllers/geometric_controller/pybind/tests")
    register_pytest_suite(pytest_mav_indi_controller
      "${CMAKE_SOURCE_DIR}/thirdparty/mav_simulator/libs/mav_controllers/controllers/indi_controller/pybind/tests")
    register_pytest_suite(pytest_mav_pid_controller_lib
      "${CMAKE_SOURCE_DIR}/thirdparty/mav_simulator/libs/mav_controllers/libs/pid_controller/pybind/tests")
    register_pytest_suite(pytest_mav_sensors
      "${CMAKE_SOURCE_DIR}/thirdparty/mav_simulator/libs/mav_sensors/pybind/tests")

    # MCAP / CSV telemetry backend
    register_pytest_suite(pytest_mav_flight_review
      "${CMAKE_SOURCE_DIR}/thirdparty/mav_flight_review/pybind/tests")

    # examples_py — Python half of the showcase, tests per adapter and framework.
    if(EXISTS "${CMAKE_SOURCE_DIR}/examples_py/tests")
      register_pytest_suite(pytest_examples_py
        "${CMAKE_SOURCE_DIR}/examples_py/tests")
      # Tests rely on MPC_EXAMPLES_REPO_ROOT to find configs/ from anywhere.
      if(TEST pytest_examples_py)
        set_property(TEST pytest_examples_py APPEND PROPERTY ENVIRONMENT
          "MPC_EXAMPLES_REPO_ROOT=${CMAKE_SOURCE_DIR}")
      endif()
    endif()

    # Trajectory generators
    register_pytest_suite(pytest_trajectory_generator_jerk_limited
      "${CMAKE_SOURCE_DIR}/thirdparty/trajectory_generator_jerk_limited/pybind/tests")
    register_pytest_suite(pytest_gcopter_lib
      "${CMAKE_SOURCE_DIR}/thirdparty/gcopter_lib/pybind/tests")
    register_pytest_suite(pytest_dynamic_trajectory_generator
      "${CMAKE_SOURCE_DIR}/thirdparty/dynamic_trajectory_generator/pybind/tests")
    register_pytest_suite(pytest_mav_trajectory_generation_lib
      "${CMAKE_SOURCE_DIR}/thirdparty/mav_trajectory_generation_lib/pybind/tests")
  else()
    message(WARNING
      "pytest not found — Python test suites will NOT be registered in ctest. "
      "Install it via `sudo apt install python3-pytest` or `pip install pytest`.")
  endif()
endif()
