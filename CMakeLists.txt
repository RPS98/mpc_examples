# Copyright 2025 Universidad Politécnica de Madrid
# Author: Rafael Perez-Segui <r.psegui@upm.es>
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.16)
set(PROJECT_NAME mpc_examples)
project(${PROJECT_NAME})

# --- C++ standard ---
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# ---------------------------------------------------------------------------
# Centralized Python package root
# ---------------------------------------------------------------------------
# Every thirdparty submodule that ships a Python package mirrors its compiled
# .so and/or pure-Python tree under PYBIND_PY_MIRROR_ROOT, so a single
# PYTHONPATH entry exposes the full namespace without needing pip install.
# Defining it here as CACHE INTERNAL pre-empts the fallback in
# thirdparty/mav_simulator/CMakeLists.txt (which otherwise falls back to
# ${CMAKE_BINARY_DIR}/pybind/python) so every sub-pybind agrees on a single
# root.
set(PYBIND_PY_MIRROR_ROOT "${CMAKE_BINARY_DIR}/python" CACHE INTERNAL
    "Shared Python mirror root for all pybind sub-packages and pure-Python mirrors")
file(MAKE_DIRECTORY "${PYBIND_PY_MIRROR_ROOT}")
message(STATUS "mpc_examples: Python packages will be gathered under ${PYBIND_PY_MIRROR_ROOT}")

# Enable pybind11 modules in all participating submodules by default.
set(BUILD_PYBIND ON CACHE BOOL "Build Python bindings in thirdparty submodules" FORCE)

# Enable CTest early so add_test() calls inside add_subdirectory()'d targets
# (e.g. examples_cpp/tests/) register correctly. The individual pytest
# suite registrations still live at the bottom of this file.
include(CTest)

# ---------------------------------------------------------------------------
# Thirdparty submodules
# ---------------------------------------------------------------------------
# mav_flight_logger: self-contained CSV logger + metrics + plotter reused by
# both the C++ and Python example stacks. Kept first because it has no
# dependencies on other thirdparty submodules.
# TODO: remove once MCAP pipeline validated (replaced by mav_flight_mcap below).
add_subdirectory(thirdparty/mav_flight_logger)

# mav_flight_mcap: replaces mav_flight_logger as the primary telemetry backend.
# Writes ROS 2-compatible MCAPs and ships a matplotlib viewer. Tests and the
# internal C++ examples are disabled here — this repo ships its own examples
# under examples_cpp/.
set(MAV_FLIGHT_MCAP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MAV_FLIGHT_MCAP_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(MAV_FLIGHT_MCAP_BUILD_PYBIND   ON  CACHE BOOL "" FORCE)
# Resolve fastcdr at the top-level scope so the IMPORTED target is visible to
# every target that links against mav_flight_mcap (find_package inside the
# subdirectory's helper function only exposes it locally). Prefer the ROS 2
# Humble package; fall back to the system CMake paths.
find_package(fastcdr CONFIG QUIET PATHS /opt/ros/humble /opt/ros/iron)
if(NOT fastcdr_FOUND)
  find_package(fastcdr CONFIG QUIET)
endif()
add_subdirectory(thirdparty/mav_flight_mcap)

# Expose the Python packages shipped by mav_flight_mcap under
# ${PYBIND_PY_MIRROR_ROOT} so `import mav_flight_mcap` and
# `import mav_flight_viewer` work with a single PYTHONPATH entry. Matches the
# symlink pattern used by mav_flight_logger / mpc / mav_simulator.
if(DEFINED PYBIND_PY_MIRROR_ROOT)
  file(MAKE_DIRECTORY "${PYBIND_PY_MIRROR_ROOT}")
  foreach(_pair
      "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/mav_flight_mcap/pybind/python/mav_flight_mcap|mav_flight_mcap"
      "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/mav_flight_mcap/viewer/src/mav_flight_viewer|mav_flight_viewer")
    string(REPLACE "|" ";" _parts "${_pair}")
    list(GET _parts 0 _src)
    list(GET _parts 1 _name)
    set(_dst "${PYBIND_PY_MIRROR_ROOT}/${_name}")
    if(EXISTS "${_dst}" OR IS_SYMLINK "${_dst}")
      file(REMOVE_RECURSE "${_dst}")
    endif()
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E create_symlink "${_src}" "${_dst}"
      RESULT_VARIABLE _symlink_rc
    )
    if(NOT _symlink_rc EQUAL 0)
      message(WARNING "mav_flight_mcap: failed to symlink ${_src} -> ${_dst}")
    else()
      message(STATUS "mav_flight_mcap: mirrored -> ${_dst}")
    endif()
  endforeach()
endif()

add_subdirectory(thirdparty/mav_simulator)

# Dynamic trajectory generator (submodule). Save/restore BUILD_TESTING so the
# submodule's CMake does not try to pull in test-only dependencies.
set(MPC_EXAMPLES_BUILD_TESTING_SAVED ${BUILD_TESTING})
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/dynamic_trajectory_generator)
set(BUILD_TESTING ${MPC_EXAMPLES_BUILD_TESTING_SAVED} CACHE BOOL "" FORCE)

# Jerk-limited trajectory generator (pybind11 + pure-C++ library).
add_subdirectory(thirdparty/trajectory_generator_jerk_limited)

# GCOPTER trajectory optimizer (pybind11 + pure-C++ library).
add_subdirectory(thirdparty/gcopter_lib)

# MPC (pure-Python core + controllers mirrored into PYBIND_PY_MIRROR_ROOT).
add_subdirectory(thirdparty/mpc)

# ---------------------------------------------------------------------------
# Project targets
# ---------------------------------------------------------------------------
find_package(yaml-cpp REQUIRED)

# Generated acados MPC targets (autogenerated code isolated under libs/).
add_subdirectory(libs)

# C++ half of the showcase: interfaces + adapters + entry-point executables.
add_subdirectory(examples_cpp)

# Python half of the showcase: pure-Python mirrors + runs + tests. The
# subdirectory only manipulates the PYBIND_PY_MIRROR_ROOT symlink and
# registers the pytest suite; the actual bindings are produced by the
# thirdparty submodules above.
if(EXISTS "${CMAKE_SOURCE_DIR}/examples_py/CMakeLists.txt")
  add_subdirectory(examples_py)
endif()

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
# CTest is already included at the top of this file. This section registers
# every thirdparty pytest suite so `ctest` from the repo root covers
# both C++ (gtest) and Python (pytest) tests in a single invocation. Each
# suite imports its bindings from ${PYBIND_PY_MIRROR_ROOT} (i.e. build/python),
# so no `pip install` is needed — only a successful `cmake --build` beforehand.
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

    # mav_flight_logger (self-contained Python logger / metrics / plotter)
    # TODO: remove once MCAP pipeline validated.
    register_pytest_suite(pytest_mav_flight_logger
      "${CMAKE_SOURCE_DIR}/thirdparty/mav_flight_logger/tests")

    # mav_flight_mcap bindings + viewer (replacement for mav_flight_logger).
    register_pytest_suite(pytest_mav_flight_mcap
      "${CMAKE_SOURCE_DIR}/thirdparty/mav_flight_mcap/pybind/tests")
    register_pytest_suite(pytest_mav_flight_viewer
      "${CMAKE_SOURCE_DIR}/thirdparty/mav_flight_mcap/viewer/tests")

    # examples_py — Python half of the showcase, tests per adapter.
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
  else()
    message(WARNING
      "pytest not found — Python test suites will NOT be registered in ctest. "
      "Install it via `sudo apt install python3-pytest` or `pip install pytest`.")
  endif()
endif()
