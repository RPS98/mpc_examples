# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# mav_flight_logger: self-contained C++ + Python package providing a CSV
# writer, metrics helpers and plotting utilities for MAV flight telemetry.
# No dependency on any other thirdparty submodule or on Eigen / yaml-cpp.

cmake_minimum_required(VERSION 3.16)
project(mav_flight_logger LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# ---------------------------------------------------------------------------
# C++ library
# ---------------------------------------------------------------------------
add_library(mav_flight_logger STATIC
  src/csv_logger.cpp
)

target_include_directories(mav_flight_logger PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_features(mav_flight_logger PUBLIC cxx_std_17)

# Position-independent code so the STATIC target can be linked into SHARED
# libraries (e.g. when the consumer ships plug-ins or pybind11 modules).
set_target_properties(mav_flight_logger PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Pure-Python mirror: expose python/mav_flight_logger under
# ${PYBIND_PY_MIRROR_ROOT}/mav_flight_logger via a live symlink. Mirrors the
# pattern used by examples/framework and thirdparty/mpc.
# ---------------------------------------------------------------------------
if(DEFINED PYBIND_PY_MIRROR_ROOT)
  file(MAKE_DIRECTORY "${PYBIND_PY_MIRROR_ROOT}")

  set(_mfl_src "${CMAKE_CURRENT_SOURCE_DIR}/python/mav_flight_logger")
  set(_mfl_dst "${PYBIND_PY_MIRROR_ROOT}/mav_flight_logger")

  if(EXISTS "${_mfl_dst}" OR IS_SYMLINK "${_mfl_dst}")
    file(REMOVE_RECURSE "${_mfl_dst}")
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink "${_mfl_src}" "${_mfl_dst}"
    RESULT_VARIABLE _mfl_symlink_rc
  )
  if(NOT _mfl_symlink_rc EQUAL 0)
    message(WARNING
      "mav_flight_logger: failed to symlink ${_mfl_src} -> ${_mfl_dst} (rc=${_mfl_symlink_rc})")
  else()
    message(STATUS "mav_flight_logger: mirrored -> ${_mfl_dst}")
  endif()
endif()

# ---------------------------------------------------------------------------
# Tests (gtest + pytest)
# ---------------------------------------------------------------------------
if(BUILD_TESTING)
  # Enable testing locally so add_test() calls below register with CTest even
  # when this subdirectory is processed before the root calls include(CTest).
  enable_testing()

  find_package(GTest QUIET)
  if(GTest_FOUND)
    add_executable(test_mav_flight_logger_csv tests/test_csv_logger.cpp)
    target_link_libraries(test_mav_flight_logger_csv PRIVATE
      mav_flight_logger
      GTest::gtest
      GTest::gtest_main
    )
    add_test(NAME mav_flight_logger_csv COMMAND test_mav_flight_logger_csv)
  else()
    message(STATUS "mav_flight_logger: GTest not found, skipping C++ tests.")
  endif()

  # Python tests are registered by the top-level CMakeLists via
  # register_pytest_suite(pytest_mav_flight_logger <tests dir>). We don't
  # self-register here to avoid needing to replicate the helper.
endif()
