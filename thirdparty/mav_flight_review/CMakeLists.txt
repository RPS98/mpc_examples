# Copyright 2025 Universidad Politécnica de Madrid
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.16)
set(PROJECT_NAME mav_flight_mcap)
project(${PROJECT_NAME} LANGUAGES CXX VERSION 0.2.0)

# --- Defaults -----------------------------------------------------------------
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(mfm_require_dep)

# --- Options ------------------------------------------------------------------
option(MAV_FLIGHT_MCAP_BUILD_EXAMPLES "Build C++ examples"          ON)
option(MAV_FLIGHT_MCAP_BUILD_TESTS    "Build C++ gtest suite"       ON)
option(MAV_FLIGHT_MCAP_BUILD_PYBIND   "Build pybind11 bindings"     OFF)

# --- Eigen3 -------------------------------------------------------------------
mfm_require_dep(
  NAME          Eigen3
  SYSTEM_PKG    Eigen3
  VENDORED_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/eigen
)

# --- lz4 / zstd (MCAP compression) -------------------------------------------
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(LZ4  QUIET liblz4)
  pkg_check_modules(ZSTD QUIET libzstd)
endif()
if(NOT LZ4_FOUND)
  message(FATAL_ERROR
    "liblz4 not found. Install: 'sudo apt install liblz4-dev'. "
    "See README.md.")
endif()
if(NOT ZSTD_FOUND)
  message(FATAL_ERROR
    "libzstd not found. Install: 'sudo apt install libzstd-dev'. "
    "See README.md.")
endif()

# --- MCAP (header-only, vendored preferred) -----------------------------------
set(MCAP_VENDORED_INC ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/mcap/include)
if(NOT EXISTS ${MCAP_VENDORED_INC}/mcap/mcap.hpp)
  find_path(MCAP_SYSTEM_INC mcap/mcap.hpp)
  if(NOT MCAP_SYSTEM_INC)
    message(FATAL_ERROR
      "MCAP headers not found at '${MCAP_VENDORED_INC}' nor on the system. "
      "See README.md.")
  endif()
  set(MCAP_VENDORED_INC ${MCAP_SYSTEM_INC})
endif()

# Compile MCAP implementation in a dedicated TU (header-only STB-style lib).
set(_mcap_impl ${CMAKE_BINARY_DIR}/mcap_impl.cpp)
file(WRITE ${_mcap_impl}
  "#define MCAP_IMPLEMENTATION\n#include <mcap/mcap.hpp>\n")

add_library(mav_flight_mcap_vendored STATIC ${_mcap_impl})
target_include_directories(mav_flight_mcap_vendored PUBLIC ${MCAP_VENDORED_INC})
target_include_directories(mav_flight_mcap_vendored PUBLIC
  ${LZ4_INCLUDE_DIRS} ${ZSTD_INCLUDE_DIRS})
target_link_libraries(mav_flight_mcap_vendored PUBLIC
  ${LZ4_LIBRARIES} ${ZSTD_LIBRARIES})
target_link_directories(mav_flight_mcap_vendored PUBLIC
  ${LZ4_LIBRARY_DIRS} ${ZSTD_LIBRARY_DIRS})
set_target_properties(mav_flight_mcap_vendored PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

# --- Fast-CDR (REQUIRED for ROS 2 CDR serialization) --------------------------
mfm_require_dep(
  NAME          fastcdr
  SYSTEM_PKG    fastcdr
  VENDORED_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/fastcdr
  EXTRA_PATHS   /opt/ros/humble /opt/ros/iron
  CONFIG
)

# --- mav_flight_mcap core library ---------------------------------------------
add_library(${PROJECT_NAME} STATIC
  src/time_source.cpp
  src/channel_registry.cpp
  src/cdr_impl.cpp
  src/mcap_logger.cpp
)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME}
  PUBLIC
    Eigen3::Eigen
    mav_flight_mcap_vendored
    fastcdr
)
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# --- Subdirectories -----------------------------------------------------------
if(MAV_FLIGHT_MCAP_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(MAV_FLIGHT_MCAP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(MAV_FLIGHT_MCAP_BUILD_PYBIND)
  add_subdirectory(pybind)
endif()

# --- Install ------------------------------------------------------------------
install(TARGETS ${PROJECT_NAME} mav_flight_mcap_vendored
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)
install(DIRECTORY include/${PROJECT_NAME} DESTINATION include)
