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

set(PYBIND_TARGET _logger_cpp)
project(${PYBIND_TARGET} LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)

# Pull in the core library (it registers the target mav_flight_mcap).
if(NOT TARGET mav_flight_mcap)
  set(MAV_FLIGHT_MCAP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(MAV_FLIGHT_MCAP_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/..
                   ${CMAKE_BINARY_DIR}/_parent_mav_flight_mcap)
endif()

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(${PYBIND_TARGET}
  src/module.cpp
  src/logger_pybind.cpp
)
target_include_directories(${PYBIND_TARGET} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(${PYBIND_TARGET} PRIVATE
  mav_flight_mcap::mav_flight_mcap
)

# --- Install target ----------------------------------------------------------
# scikit-build-core provides SKBUILD_PROJECT_NAME etc. when building a wheel.
if(DEFINED SKBUILD_PROJECT_NAME)
  install(TARGETS ${PYBIND_TARGET} LIBRARY DESTINATION mav_flight_mcap)
else()
  # Direct CMake build: mirror the .so into python/mav_flight_mcap/ so a
  # developer `pip install -e .` sees it alongside the pure-Python wrappers.
  set_target_properties(${PYBIND_TARGET} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY
      "${CMAKE_CURRENT_SOURCE_DIR}/python/mav_flight_mcap")
endif()
