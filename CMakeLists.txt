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

# ---------------------------------------------------------------------------
# Thirdparty submodules
# ---------------------------------------------------------------------------
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

# Generated acados MPC targets used by the controller examples.
add_subdirectory(examples/acados_position_mpc)
add_subdirectory(examples/acados_trajectory_mpc)

# Shared OOP framework (IController, ITrajectoryGenerator, WaypointsSimulator)
# used by the unified per-combination example mains.
add_subdirectory(examples/framework)

# Adapter targets that bridge the framework interfaces to the concrete
# third-party controllers and trajectory generators.
add_subdirectory(examples/adapters)

# Unified per-combination example mains (one executable per (controller, generator)).
add_subdirectory(examples/examples)

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
include(CTest)

# Register every thirdparty pytest suite so `ctest` from the repo root covers
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
